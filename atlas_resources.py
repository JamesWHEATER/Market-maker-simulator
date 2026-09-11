"""Destination checks and sampled process-tree memory accounting for Atlas.

The watchdog is a cooperative guard, not an OS memory reservation. Native
allocations can fail between samples; the report never certifies an unseen run.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path

GIB = 1024 ** 3
DEFAULT_ATLAS_DIR = Path("D:/Atlas/atlas_1000000x2")


def memory_snapshot():
    """Return physical availability and summed RSS/private bytes of our tree."""
    if os.name != "nt":
        # Linux support also makes the accounting testable on CI without psutil.
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            info[name] = int(value.split()[0]) * 1024
        processes = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                processes[int(entry.name)] = (int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError, IndexError):
                continue
        tree = {os.getpid()}
        while True:
            added = {pid for pid, (ppid, _) in processes.items() if ppid in tree} - tree
            if not added:
                break
            tree.update(added)
        rss = sum(processes.get(pid, (0, 0))[1] for pid in tree)
        return dict(total=info["MemTotal"], available=info["MemAvailable"], rss=rss,
                    private=rss, os_peak_rss=rss, processes=len(tree))

    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    parents = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    tree = {os.getpid()}
    while True:
        added = {pid for pid, ppid in parents.items() if ppid in tree} - tree
        if not added:
            break
        tree.update(added)
    rss = private = peak = 0
    for pid in tree:
        handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            if pid == os.getpid():
                raise ctypes.WinError(ctypes.get_last_error())
            continue  # A worker can exit between the snapshot and this query.
        try:
            counts = Counters()
            counts.cb = ctypes.sizeof(counts)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counts), counts.cb):
                rss += counts.WorkingSetSize
                private += counts.PrivateUsage
                peak += counts.PeakWorkingSetSize
            elif pid == os.getpid():
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
    return dict(total=status.total, available=status.available, rss=rss, private=private,
                os_peak_rss=peak, processes=len(tree))


def require_d_destination(path: Path) -> Path:
    """Resolve junctions as well as drive letters; never silently fall back to C:."""
    path = Path(path).resolve()
    if os.name != "nt" or path.drive.upper() != "D:":
        raise ValueError(f"Atlas output and cache must be on D:, received {path}")
    if not Path("D:/").is_dir():
        raise FileNotFoundError("D: is unavailable; Atlas will not fall back to another drive")
    # Check all existing descendants, including a cache symlink from an older run.
    for name in ("data", "data/shards", "models", "cache", "xgboost_cache", "tmp"):
        if (path / name).resolve().drive.upper() != "D:":
            raise ValueError(f"Atlas {name} resolves outside D:")
    return path


def atomic_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class ResourceMonitor:
    def __init__(self, output_dir, *, memory_budget_gib=None, reserve_ram_gib=1.0,
                 min_free_disk_gib=5.0, sample_seconds=0.5):
        for name, value in (("reserve_ram_gib", reserve_ram_gib), ("min_free_disk_gib", min_free_disk_gib),
                            ("sample_seconds", sample_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if memory_budget_gib is not None and (not math.isfinite(memory_budget_gib) or memory_budget_gib <= 0):
            raise ValueError("memory_budget_gib must be finite and positive")
        self.root = Path(output_dir)
        self.initial = memory_snapshot()
        conservative = min(6 * GIB, self.initial["total"] * 0.5,
                           max(self.initial["rss"], self.initial["private"]) + self.initial["available"] * 0.65)
        self.budget_bytes = int(conservative if memory_budget_gib is None else memory_budget_gib * GIB)
        if self.budget_bytes > conservative:
            raise ValueError(f"Requested memory budget exceeds current conservative limit ({conservative / GIB:.2f} GiB)")
        self.reserve_bytes = int(reserve_ram_gib * GIB)
        self.disk_floor = int(min_free_disk_gib * GIB)
        self.interval = sample_seconds
        self.stage = "setup"
        self.failure = None
        self.stages = {}
        self.peak_rss = self.peak_private = self.peak_os_rss = 0
        self.min_available = self.initial["available"]
        self.max_processes = 0
        self.started = time.monotonic()
        self.last_sample = 0.0
        self.last_progress = self.started
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    @staticmethod
    def _stage_group(stage):
        lowered = stage.lower()
        for name in ("generation", "preprocess", "logistic", "mlp", "xgboost", "validation", "test", "explanation", "capacity"):
            if name in lowered:
                return name
        return "other"

    def _sample(self):
        with self.lock:
            values = memory_snapshot()
            now = time.monotonic()
            self.peak_rss = max(self.peak_rss, values["rss"])
            self.peak_private = max(self.peak_private, values["private"])
            self.peak_os_rss = max(self.peak_os_rss, values["os_peak_rss"])
            self.min_available = min(self.min_available, values["available"])
            self.max_processes = max(self.max_processes, values["processes"])
            record = self.stages.setdefault(self._stage_group(self.stage), {"peak_rss_bytes": 0, "peak_private_bytes": 0, "samples": 0})
            record["peak_rss_bytes"] = max(record["peak_rss_bytes"], values["rss"])
            record["peak_private_bytes"] = max(record["peak_private_bytes"], values["private"])
            record["samples"] += 1
            free = shutil.disk_usage(self.root).free
            if max(values["rss"], values["private"]) > self.budget_bytes:
                self.failure = self.failure or f"Process-tree memory exceeded {self.budget_bytes / GIB:.2f} GiB budget during {self.stage}"
            if values["available"] < self.reserve_bytes:
                self.failure = self.failure or f"Available physical RAM fell below {self.reserve_bytes / GIB:.2f} GiB during {self.stage}"
            if free < self.disk_floor:
                self.failure = self.failure or f"D: free space fell below {self.disk_floor / GIB:.2f} GiB during {self.stage}"
            self.last_sample = now

    def _watch(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except Exception as exc:
                self.failure = self.failure or f"Resource accounting failed: {exc}"

    def check(self, stage="other"):
        self.stage = stage
        if self._stage_group(stage) not in self.stages or time.monotonic() - self.last_sample >= self.interval:
            self._sample()
        if self.failure:
            raise MemoryError(self.failure)
        if time.monotonic() - self.last_progress >= 30:
            self.last_progress = time.monotonic()
            atomic_json(self.root / "resource_report.json", self.report())
            # Generation owns its 1,000-run console updates; keep resource
            # snapshots and guard checks active without extra timer messages.
            if self._stage_group(stage) != "generation":
                print(f"Atlas progress: {stage}; peak tree RSS {self.peak_rss / GIB:.2f} GiB, "
                      f"private memory {self.peak_private / GIB:.2f} GiB", flush=True)

    def __enter__(self):
        self._sample()
        self.check("setup")
        self.thread = threading.Thread(target=self._watch, name="atlas-resource-monitor", daemon=True)
        self.thread.start()
        print(f"Atlas RAM budget: {self.budget_bytes / GIB:.2f} GiB; available at start: {self.initial['available'] / GIB:.2f} GiB", flush=True)
        return self

    def report(self):
        return {"memory_budget_bytes": self.budget_bytes, "reserve_ram_bytes": self.reserve_bytes,
                "disk_reserve_bytes": self.disk_floor, "initial_memory": self.initial,
                "peak_process_tree_rss_bytes": self.peak_rss, "peak_process_tree_private_bytes": self.peak_private,
                "max_sum_live_process_os_peak_rss_bytes": self.peak_os_rss,
                "minimum_available_ram_bytes": self.min_available, "max_processes": self.max_processes,
                "elapsed_seconds": time.monotonic() - self.started, "sample_interval_seconds": self.interval,
                "stages": self.stages, "resource_guard_failure": self.failure,
                "scope": "This invocation only. Sampled process-tree RSS/private memory; shared RSS may be counted twice.",
                "limitation": "Cooperative guard, not a hard allocation cap; native allocations may fail between checks. Full-scale feasibility is unproven."}

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self._sample()
        atomic_json(self.root / "resource_report.json", self.report())
        if exc_type is None and self.failure:
            raise MemoryError(self.failure)
