"""Read-only capacity screening from pilot artifacts; does not certify training fit."""
import argparse
import ctypes
import json
import shutil
from pathlib import Path

import pandas as pd


def streaming_capacity(pilot_dir, structures, seeds_per_structure, storage_dir):
    """Extrapolate a compressed pilot without opening its observations/test labels."""
    from atlas_resources import GIB, memory_snapshot
    from atlas_streaming_data import StreamingDataset, MANIFEST_FILENAME
    root = Path(pilot_dir)
    data_root = root if (root / MANIFEST_FILENAME).exists() else root / "data"
    dataset = StreamingDataset(data_root)
    manifest = dataset.manifest
    if data_root == root:
        root = root.parent
    runs = manifest["generated_runs"]
    scale = structures * seeds_per_structure / runs
    projected_rows = manifest["rows"] * scale
    compressed = (manifest["compressed_bytes"] + manifest["catalog_bytes"]) * scale
    predictions = sum(path.stat().st_size for path in root.glob("test_*.csv.gz")) * scale
    cache = 0
    feature_counts = {}
    for feature_set in ("observable", "oracle"):
        path = root / f"streaming_training_{feature_set}.json"
        if path.exists():
            training = json.loads(path.read_text(encoding="utf-8"))
            # Feature sets train sequentially, so reserve the larger live cache.
            cache = max(cache, training["xgboost"]["cache_bytes_observed"] * scale)
            feature_counts[feature_set] = training["preprocessing"]["feature_count"]
    free = shutil.disk_usage(storage_dir).free
    memory = memory_snapshot()
    report_file = root / "resource_report.json"
    measured = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else None
    status_file = root / "atlas_status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
    required_stages = {"generation", "xgboost", "mlp", "test"}
    observed_stages = {name for name, values in (measured or {}).get("stages", {}).items()
                       if values.get("samples", 0) > 0}
    training_complete = (status.get("status") == "EVALUATION_COMPLETE"
                         and status.get("command") == "all" and len(feature_counts) == 2
                         and required_stages.issubset(observed_stages))
    # Data/catalog/predictions/cache have measured per-run rates; small model files
    # are not multiplied by the number of target simulations.
    allowance = (compressed + predictions + cache) * 1.5 + 5 * GIB
    row_state = 32 * (manifest["rows_by_split"]["train"] + manifest["rows_by_split"]["validation"]) * scale
    generation_seconds = manifest.get("generation_seconds", 0.0) * scale
    measured_peak = max((measured or {}).get("peak_process_tree_rss_bytes", 0),
                        (measured or {}).get("peak_process_tree_private_bytes", 0))
    projected_ram = measured_peak + row_state
    current_budget = min(6 * GIB, memory["total"] * 0.5,
                         max(memory["rss"], memory["private"]) + memory["available"] * 0.65)
    ram_screen = bool(measured and projected_ram < current_budget and memory["available"] > GIB)
    result = {
        "target_structures": structures, "target_seeds_per_structure": seeds_per_structure,
        "pilot_runs": runs, "pilot_steps_per_run": manifest["experiment"]["steps_per_run"],
        "projected_pattern_rows": projected_rows,
        "projected_compressed_data_and_catalog_bytes": compressed,
        "projected_test_exports_bytes": predictions, "projected_live_xgboost_cache_bytes": cache,
        "planning_disk_bytes_with_50_percent_headroom_and_5_gib_reserve": allowance,
        "destination_free_bytes": free,
        "xgboost_row_state_32_bytes_per_row_planning_bytes": row_state,
        "projected_peak_plus_row_state_planning_bytes": projected_ram,
        "current_conservative_memory_budget_bytes": current_budget,
        "memory_screen_passed": ram_screen,
        "projected_generation_seconds_linear_pilot_rate": generation_seconds,
        "pilot_measured_memory": measured, "both_feature_workflow_measured": training_complete,
        "required_memory_stages": sorted(required_stages), "observed_memory_stages": sorted(observed_stages),
        "screen_passed": bool(training_complete and ram_screen and allowance < free and not (measured or {}).get("resource_guard_failure")),
        "full_scale_certified": False,
        "assumptions": ["Same path length, sampling settings, feature vocabulary, shards and training configuration as pilot",
                        "Compressed sizes and cache sizes extrapolated per run; small pilots can miss rare/denser structures",
                        "Generation time is a linear rate estimate, excludes failed/replayed shards and machine contention",
                        "XGBoost row-state allowance is a heuristic, not a bound; peak RAM is not constant in row count",
                        "Training/evaluation runtime must be measured at increasing sizes; external-memory I/O can dominate"]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def windows_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total",
             "virtual_available", "extended")
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.total, status.available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--structures", type=int, required=True)
    parser.add_argument("--seeds-per-structure", type=int, required=True)
    parser.add_argument("--storage-dir", type=Path, default=Path("D:/"))
    args = parser.parse_args()
    if min(args.structures, args.seeds_per_structure) < 1:
        parser.error("structure and seed counts must be positive")
    if ((args.pilot_dir / "atlas_dataset_manifest.json").exists()
            or (args.pilot_dir / "data" / "atlas_dataset_manifest.json").exists()):
        result = streaming_capacity(args.pilot_dir, args.structures, args.seeds_per_structure, args.storage_dir)
        return 0 if result["screen_passed"] else 2
    manifest = json.loads((args.pilot_dir / "ai_research_manifest.json").read_text())
    experiment = manifest["experiment"]
    artifact = manifest["dataset_artifacts"]["pattern_training_dataset.csv"]
    dataset = args.pilot_dir / "pattern_training_dataset.csv"
    if dataset.stat().st_size != artifact["bytes"]:
        raise ValueError("Pilot dataset size differs from its manifest")
    sample = pd.read_csv(dataset, nrows=10000)
    scale = args.structures * args.seeds_per_structure / (
        experiment["structures"] * experiment["seeds_per_structure"])
    rows = artifact["rows"] * scale
    raw_bytes = artifact["bytes"] * scale
    frame_bytes = sample.memory_usage(index=True, deep=True).sum() / len(sample) * rows
    total, available = windows_memory()
    free = shutil.disk_usage(args.storage_dir).free
    gib = 1024 ** 3
    print(f"Target: {args.structures:,} structures x {args.seeds_per_structure} seeds")
    print(f"Assumes pilot path length ({experiment['steps_per_run']}) and sampling settings.")
    print(f"Estimated pattern rows: {rows:,.0f}")
    print(f"Estimated raw CSV: {raw_bytes / gib:,.1f} GiB")
    print(f"Estimated single DataFrame: {frame_bytes / gib:,.1f} GiB")
    print(f"RAM: {total / gib:.1f} GiB total; {available / gib:.1f} GiB available")
    print(f"Disk free on destination: {free / gib:.1f} GiB")
    print(f"Planning allowances: {6 * frame_bytes / gib:.1f} GiB RAM; {4 * raw_bytes / gib:.1f} GiB disk")
    print("Allowances are heuristics, not measured peaks or guaranteed upper bounds.")
    print("Disk allowance includes checkpoint storage, CSV copies and working space.")
    if frame_bytes > available * .7 or raw_bytes > free:
        print("NO-GO: even the estimated base dataset exceeds the available resource budget.")
        return 2
    if 6 * frame_bytes > available * .7 or 4 * raw_bytes > free * .8:
        print("NO-GO: insufficient planning headroom for the current in-memory pipeline.")
        return 2
    print("SCREEN PASSED, NOT CERTIFIED: measure a full both-feature training run's process-tree peak")
    print("on independent capacity data at the intended size before claiming training fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
