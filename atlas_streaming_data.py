"""Bounded, resumable Atlas generation and disk readers.

Only one structure-aligned shard of specifications and bounded worker results is
resident. SQLite owns the growing design, seed-uniqueness index and shard catalog.
Pattern observations are stored once in gzip CSV shards. A committed shard and its
sampler state form one checkpoint; interrupted shards are regenerated on resume.

Sampling version 1 deliberately replaces the legacy shuffled null-flag vector by
sequential selection with an independent RNG. It preserves the exact null count,
but is a new frozen design, not a continuation of a legacy in-memory experiment.
Split assignment uses only structure identity and the frozen seed; the first three
structures reserve train/validation/test so small fixtures have all partitions.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sqlite3
import time
import zlib
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import ai_pattern_research_pipeline as ai
from atlas_progress import GenerationProgress


FORMAT_VERSION = "atlas-streaming-1"
SAMPLING_VERSION = "sequential-exact-null-sqlite-unique-1"
SPLIT_METHOD = "seeded-sha256-structure-first-three-reserved-1"
PARTITIONS = ("train", "validation", "test")
CATALOG_FILENAME = "atlas_catalog.sqlite3"
MANIFEST_FILENAME = "atlas_dataset_manifest.json"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _packed(value: Any) -> bytes:
    return zlib.compress(_json(value).encode("utf-8"))


def _unpacked(value: bytes) -> Any:
    return json.loads(zlib.decompress(value))


def _check(monitor: Any, stage: str) -> None:
    if monitor is not None:
        monitor.check(stage)


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    # SQLite caches and ORDER BY spill files must not grow with the experiment.
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri if readonly else str(path), uri=readonly)
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")
    if not readonly:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA journal_mode=DELETE")
    return connection


@contextmanager
def _generation_lock(root: Path):
    """OS lock releases after a crash; the persistent lock file is harmless."""
    path = root / ".atlas-generation.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another generator is writing this dataset") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def structure_split(structure_id: str, structure_index: int, experiment: ai.AIExperimentConfig) -> str:
    """Assign all seeds before observing outcomes, in constant memory."""
    if structure_index < 3:
        return PARTITIONS[structure_index]
    value = int.from_bytes(hashlib.sha256(
        f"{SPLIT_METHOD}|{experiment.base_seed}|{structure_id}".encode("utf-8")
    ).digest()[:8], "big") / 2**64
    if value < experiment.train_fraction:
        return "train"
    if value < experiment.train_fraction + experiment.validation_fraction:
        return "validation"
    return "test"


def _contract(experiment, costs, ranges) -> dict[str, Any]:
    paths = {
        "streaming_data": Path(__file__), "ai_pipeline": Path(ai.__file__),
        "simulator": ai._module_path(ai.SyntheticMarketConfig),
        "candle_builder": ai._module_path(ai.build_candles),
        "pattern_detector": ai._module_path(ai.detect_pattern_universe),
        "execution_costs": ai._module_path(ai.net_trade_return),
    }
    return {
        "format_version": FORMAT_VERSION, "sampling_version": SAMPLING_VERSION,
        "split_method": SPLIT_METHOD, "experiment": asdict(experiment),
        "costs": asdict(costs), "ranges": asdict(ranges),
        "source_sha256": {name: ai._file_sha256(path) for name, path in paths.items()},
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "split_frozen_before_outcomes": True,
        "split_unit": "complete structure including every repeated seed",
        "null_assignment": "exact count, sequential sampling without replacement",
    }


def _initialize(connection, contract, experiment) -> dict[str, Any]:
    connection.executescript("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE structures (
            structure_index INTEGER PRIMARY KEY, structure_id TEXT UNIQUE NOT NULL,
            split TEXT NOT NULL, world_family TEXT NOT NULL, config BLOB NOT NULL
        );
        CREATE INDEX structures_split ON structures(split, structure_index);
        CREATE TABLE seeds (seed INTEGER PRIMARY KEY);
        CREATE TABLE shards (
            shard_id INTEGER PRIMARY KEY, filename TEXT UNIQUE NOT NULL,
            first_simulation_id INTEGER NOT NULL, run_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL
        );
        CREATE TABLE runs (
            simulation_id INTEGER PRIMARY KEY, run_id TEXT UNIQUE NOT NULL,
            structure_index INTEGER NOT NULL REFERENCES structures(structure_index),
            seed INTEGER NOT NULL UNIQUE, split TEXT NOT NULL,
            shard_id INTEGER NOT NULL REFERENCES shards(shard_id),
            row_count INTEGER NOT NULL, summary BLOB NOT NULL
        );
        CREATE INDEX runs_shard ON runs(shard_id, simulation_id);
        CREATE INDEX runs_split ON runs(split, simulation_id);
    """)
    master = np.random.default_rng(experiment.base_seed)
    null_rng = np.random.default_rng(np.random.SeedSequence([experiment.base_seed, 981723]))
    state = {
        "next_structure": 0, "next_simulation": 0, "next_shard": 0,
        "null_remaining": int(round(experiment.structures * experiment.null_world_fraction)),
        "master_state": master.bit_generator.state, "null_state": null_rng.bit_generator.state,
        "columns": [], "rows": 0, "compressed_bytes": 0,
        "uncompressed_bytes": 0, "generation_seconds": 0.0,
        "rows_by_split": dict.fromkeys(PARTITIONS, 0),
        "structures_by_split": dict.fromkeys(PARTITIONS, 0),
        "runs_by_split": dict.fromkeys(PARTITIONS, 0),
        "skipped_incomplete_horizon": 0, "skipped_unresolved_direction": 0,
    }
    connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
        ("contract", _json(contract)), ("state", _json(state)),
    ])
    connection.commit()
    return state


def _sample_shard(connection, state, experiment, ranges, structure_count):
    """SQLite, rather than Python sets, guarantees global structure/seed uniqueness."""
    master = np.random.default_rng()
    master.bit_generator.state = state["master_state"]
    null_rng = np.random.default_rng()
    null_rng.bit_generator.state = state["null_state"]
    specs = []
    splits = []
    for index in range(state["next_structure"], min(
        experiment.structures, state["next_structure"] + structure_count,
    )):
        remaining = experiment.structures - index
        is_null = int(null_rng.integers(remaining)) < state["null_remaining"]
        state["null_remaining"] -= int(is_null)
        for _attempt in range(100):
            template, family = ai._sample_structure_template(index, is_null, experiment, ranges, master)
            template = ai.canonical_behavior_config(template)
            sid = ai._structure_id(template)
            split = structure_split(sid, index, experiment)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO structures VALUES (?, ?, ?, ?, ?)",
                (index, sid, split, family, _packed(asdict(template))),
            ).rowcount
            if inserted:
                break
        else:
            raise RuntimeError("could not generate a unique synthetic structure")
        seed_rng = np.random.default_rng(experiment.base_seed + 1_000_003 * (index + 1))
        for _replicate in range(experiment.seeds_per_structure):
            while True:
                seed = int(seed_rng.integers(0, 2**31 - 1))
                if connection.execute("INSERT OR IGNORE INTO seeds VALUES (?)", (seed,)).rowcount:
                    break
            specs.append(ai.RunSpec(state["next_simulation"], index, sid, family, replace(template, seed=seed)))
            splits.append(split)
            state["next_simulation"] += 1
            state["runs_by_split"][split] += 1
        state["structures_by_split"][split] += 1
        state["next_structure"] = index + 1
    state["master_state"] = master.bit_generator.state
    state["null_state"] = null_rng.bit_generator.state
    return specs, splits


def _manifest(contract, state, status):
    contract_hash = hashlib.sha256(_json(contract).encode("utf-8")).hexdigest()
    return {
        **contract, "status": status, "contract_sha256": contract_hash,
        "dataset_id": f"ATLAS-{contract_hash[:24]}", "columns": state["columns"],
        "rows": state["rows"], "rows_by_split": state["rows_by_split"],
        "generated_runs": state["next_simulation"], "generated_structures": state["next_structure"],
        "runs_by_split": state["runs_by_split"], "structures_by_split": state["structures_by_split"],
        "shards": state["next_shard"], "compressed_bytes": state["compressed_bytes"],
        "uncompressed_bytes": state["uncompressed_bytes"], "generation_seconds": state["generation_seconds"],
        "skipped_incomplete_horizon": state["skipped_incomplete_horizon"],
        "skipped_unresolved_direction": state["skipped_unresolved_direction"],
        "catalog_filename": CATALOG_FILENAME, "models_trained": False,
    }


def _verify_checkpoint(connection, state):
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("Atlas catalog integrity check failed")
    for table, field, expected in (
        ("structures", "structure_index", state["next_structure"]),
        ("runs", "simulation_id", state["next_simulation"]),
        ("shards", "shard_id", state["next_shard"]),
    ):
        count, low, high = connection.execute(
            f"SELECT COUNT(*), MIN({field}), MAX({field}) FROM {table}"
        ).fetchone()
        if count != expected or (count and (low != 0 or high != count - 1)):
            raise ValueError("Atlas checkpoint is not a contiguous committed prefix")
    if connection.execute("SELECT COUNT(*) FROM seeds").fetchone()[0] != state["next_simulation"]:
        raise ValueError("Atlas seed index differs from the committed run count")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("Atlas catalog contains broken design references")
    if connection.execute("SELECT COALESCE(SUM(row_count),0) FROM runs").fetchone()[0] != state["rows"]:
        raise ValueError("Atlas checkpoint row count differs from the run catalog")


def generate_streaming_dataset(
    output_dir: Path, experiment: ai.AIExperimentConfig, *, costs: ai.CostModel = ai.CostModel(),
    ranges: ai.WorldSamplingRanges = ai.WorldSamplingRanges(), jobs: int = 1,
    runs_per_shard: int = 32, monitor=None, resume: bool = False,
) -> "StreamingDataset":
    """Generate compressed shards without constructing a total-sized run design.

    The caller selects and validates the disk target (the Atlas CLI requires D:).
    Changing jobs or runs_per_shard on resume is safe: design and observations are
    independent of execution batch boundaries. A failed shard is never reused.
    """
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("streaming jobs must be a positive explicit worker count")
    if isinstance(runs_per_shard, bool) or not isinstance(runs_per_shard, int) or runs_per_shard < experiment.seeds_per_structure:
        raise ValueError("runs_per_shard must be an integer >= seeds_per_structure")
    if experiment.total_runs > 2**31 - 1:
        raise ValueError("requested runs exceed the unique simulator seed space")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not resume and any(root.iterdir()):
        raise FileExistsError(f"use an empty output directory or resume=True: {root}")
    catalog = root / CATALOG_FILENAME
    if resume and not catalog.is_file():
        raise FileNotFoundError(f"no Atlas checkpoint to resume: {catalog}")
    contract = _contract(experiment, costs, ranges)
    with _generation_lock(root):
        connection = _connect(catalog)
        generation_started = False
        try:
            if resume:
                saved = connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
                if saved is None or saved[0] != _json(contract):
                    raise ValueError("Atlas checkpoint settings/code differ; use the frozen procedure or a new directory")
                state = json.loads(connection.execute("SELECT value FROM metadata WHERE key='state'").fetchone()[0])
                _verify_checkpoint(connection, state)
                # Validate committed files before doing any further simulations.
                for filename, size, digest in connection.execute("SELECT filename, bytes, sha256 FROM shards ORDER BY shard_id"):
                    _verified_shard(root, filename, size, digest)
                    _check(monitor, "resume integrity")
                previous = root / MANIFEST_FILENAME
                if previous.is_file():
                    manifest = json.loads(previous.read_text(encoding="utf-8"))
                    if manifest.get("status") == "COMPLETE":
                        dataset = StreamingDataset(root)
                        GenerationProgress(experiment.total_runs, completed_runs=state["next_simulation"],
                                           elapsed_seconds=state["generation_seconds"]).finish()
                        return dataset
            else:
                state = _initialize(connection, contract, experiment)
            (root / "shards").mkdir(exist_ok=True)
            _write_json(root / MANIFEST_FILENAME, _manifest(contract, state, "RUNNING"))
            generation_started = True
            progress = GenerationProgress(experiment.total_runs, completed_runs=state["next_simulation"],
                                          elapsed_seconds=state["generation_seconds"])
            while state["next_structure"] < experiment.structures:
                _check(monitor, "generation shard start")
                shard_started = time.perf_counter()
                # Roll back both design uniqueness and RNG checkpoint on failure.
                next_state = json.loads(_json(state))
                connection.execute("BEGIN IMMEDIATE")
                specs, splits = _sample_shard(connection, next_state, experiment, ranges,
                                              runs_per_shard // experiment.seeds_per_structure)
                shard_id = state["next_shard"]
                relative = f"shards/part-{shard_id:08d}.csv.gz"
                target = root / relative
                temporary = target.with_name(target.name + ".building")
                connection.execute("INSERT INTO shards VALUES (?, ?, ?, ?, 0, 0, '')",
                                   (shard_id, relative, specs[0].simulation_id, len(specs)))
                shard_rows = 0
                result_count = 0
                with gzip.open(temporary, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
                    writer = None
                    if next_state["columns"]:
                        writer = csv.DictWriter(handle, fieldnames=next_state["columns"])
                        writer.writeheader()
                    for result_count, (rows, raw_summary) in enumerate(ai.iter_pattern_runs(
                        specs, experiment, costs=costs, jobs=jobs, git_commit=None,
                    ), start=1):
                        if result_count > len(specs):
                            raise ValueError("simulation iterator returned extra runs")
                        spec = specs[result_count - 1]
                        split = splits[result_count - 1]
                        expected_run = ai.simulator_run_id(spec.config)
                        if raw_summary.get("simulation_id") != spec.simulation_id or raw_summary.get("run_id") != expected_run:
                            raise ValueError("simulation summary differs from frozen run design")
                        if raw_summary.get("structure_id") != spec.structure_id:
                            raise ValueError("simulation structure differs from frozen run design")
                        summary = {**raw_summary, "split": split, "structure_index": spec.structure_index,
                                   "seed": spec.config.seed, "world_family": spec.world_family,
                                   "training_rows": len(rows)}
                        for source in rows:
                            if source.get("run_id") != expected_run or source.get("structure_id") != spec.structure_id:
                                raise ValueError("pattern row differs from frozen run design")
                            net = float(source["net_directional_return"])
                            if not math.isfinite(net) or int(source["profitable"]) != int(net > 0):
                                raise ValueError("invalid finite net-return label")
                            row = {**source, "simulation_id": spec.simulation_id,
                                   "structure_index": spec.structure_index, "seed": spec.config.seed,
                                   "world_family": spec.world_family, "split": split,
                                   "label_horizon_bars": experiment.label_horizon_candles,
                                   "profitable_after_costs": int(net > 0)}
                            if "entry_mid" in row and "exit_mid" in row:
                                row[f"future_{experiment.label_horizon_candles}_bar_return"] = float(row["exit_mid"]) / float(row["entry_mid"]) - 1
                            if writer is None:
                                next_state["columns"] = list(row)
                                writer = csv.DictWriter(handle, fieldnames=next_state["columns"])
                                writer.writeheader()
                            if set(row) != set(next_state["columns"]):
                                raise ValueError("pattern schema changed during generation")
                            writer.writerow({name: "" if isinstance(value, (float, np.floating)) and math.isnan(value) else value
                                             for name, value in row.items()})
                            shard_rows += 1
                            next_state["rows_by_split"][split] += 1
                        connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                            spec.simulation_id, expected_run, spec.structure_index, spec.config.seed,
                            split, shard_id, len(rows), _packed(summary),
                        ))
                        for name in ("skipped_incomplete_horizon", "skipped_unresolved_direction"):
                            next_state[name] += int(summary.get(name, 0))
                        _check(monitor, "generation run complete")
                        progress.update(state["next_simulation"] + result_count)
                    if result_count != len(specs):
                        raise ValueError("simulation iterator omitted planned runs")
                    handle.flush()
                    uncompressed_bytes = handle.buffer.tell()
                # Flush the actual compressed bytes before committing their catalog entry.
                with temporary.open("r+b") as handle:
                    os.fsync(handle.fileno())
                temporary.replace(target)
                size, digest = target.stat().st_size, ai._file_sha256(target)
                connection.execute("UPDATE shards SET row_count=?, bytes=?, sha256=? WHERE shard_id=?",
                                   (shard_rows, size, digest, shard_id))
                next_state["next_shard"] += 1
                next_state["rows"] += shard_rows
                next_state["compressed_bytes"] += size
                next_state["uncompressed_bytes"] += uncompressed_bytes
                next_state["generation_seconds"] += time.perf_counter() - shard_started
                connection.execute("UPDATE metadata SET value=? WHERE key='state'", (_json(next_state),))
                connection.commit()
                state = next_state
                del specs, splits, rows
                _write_json(root / MANIFEST_FILENAME, _manifest(contract, state, "RUNNING"))
                _check(monitor, "generation shard committed")
            _verify_checkpoint(connection, state)
        except BaseException as exc:
            connection.rollback()
            if generation_started:
                failed = _manifest(contract, state, "FAILED")
                failed["error"] = f"{type(exc).__name__}: {exc}"
                _write_json(root / MANIFEST_FILENAME, failed)
            raise
        finally:
            connection.close()
        complete = _manifest(contract, state, "COMPLETE")
        complete["catalog_sha256"] = ai._file_sha256(catalog)
        complete["catalog_bytes"] = catalog.stat().st_size
        _write_json(root / MANIFEST_FILENAME, complete)
    _check(monitor, "dataset complete")
    dataset = StreamingDataset(root)
    progress.finish()
    return dataset


def _verified_shard(root, filename, size, digest):
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or path.parent != (root / "shards").resolve():
        raise ValueError("invalid shard path in catalog")
    if not path.is_file() or path.stat().st_size != size or ai._file_sha256(path) != digest:
        raise ValueError(f"Atlas shard changed after generation: {filename}")
    return path


class StreamingDataset:
    """Read a frozen dataset through bounded iterators; no full-data load method."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        manifest = self.manifest
        if manifest.get("format_version") != FORMAT_VERSION or manifest.get("status") != "COMPLETE":
            raise ValueError("Atlas dataset is not a complete supported frozen dataset; resume generation first")
        catalog = self.root / CATALOG_FILENAME
        if not catalog.is_file() or ai._file_sha256(catalog) != manifest.get("catalog_sha256"):
            raise ValueError("Atlas catalog changed after generation")
        with closing(_connect(catalog, readonly=True)) as connection:
            contract = json.loads(connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0])
            state = json.loads(connection.execute("SELECT value FROM metadata WHERE key='state'").fetchone()[0])
            expected = _manifest(contract, state, "COMPLETE")
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("Atlas manifest differs from its frozen catalog")

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    def _shards(self, connection, split):
        if split is not None and split not in PARTITIONS:
            raise ValueError(f"unknown split: {split}")
        if split is None:
            return connection.execute("SELECT shard_id, filename, row_count, bytes, sha256 FROM shards ORDER BY shard_id")
        return connection.execute("""SELECT shard_id, filename, row_count, bytes, sha256 FROM shards s
            WHERE EXISTS (SELECT 1 FROM runs r WHERE r.shard_id=s.shard_id AND r.split=?)
            ORDER BY shard_id""", (split,))

    def iter_batches(self, split=None, batch_rows: int = 4096, columns=None) -> Iterator[pd.DataFrame]:
        """Read <= batch_rows observations, including when filtering partitions."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        schema = self.manifest["columns"]
        wanted = list(schema) if columns is None else list(columns)
        if not set(wanted).issubset(schema):
            raise ValueError("requested columns are absent from the dataset schema")
        read_columns = list(dict.fromkeys(wanted + (["split"] if split is not None else [])))
        connection = _connect(self.root / CATALOG_FILENAME, readonly=True)
        try:
            for _, filename, count, size, digest in self._shards(connection, split):
                path = _verified_shard(self.root, filename, size, digest)
                if not count:
                    continue
                observed = 0
                with pd.read_csv(path, chunksize=batch_rows, usecols=read_columns) as reader:
                    for frame in reader:
                        observed += len(frame)
                        if split is not None:
                            frame = frame.loc[frame["split"] == split]
                        if not frame.empty:
                            yield frame.loc[:, wanted].reset_index(drop=True)
                if observed != count:
                    raise ValueError("Atlas shard row count differs from its catalog")
        finally:
            connection.close()

    def iter_runs(self, split=None) -> Iterator[tuple[pd.DataFrame, dict[str, Any]]]:
        """Yield complete runs in structure/seed order, retaining zero-event runs.

        At most one run plus one 4096-row CSV buffer is resident. The run bound is
        controlled by steps_per_run, not the total number of structures.
        """
        columns = self.manifest["columns"]
        connection = _connect(self.root / CATALOG_FILENAME, readonly=True)
        try:
            for shard_id, filename, count, size, digest in self._shards(connection, split):
                path = _verified_shard(self.root, filename, size, digest)
                reader = pd.read_csv(path, chunksize=4096) if count else None
                chunks = iter(reader) if reader is not None else iter(())
                frame = pd.DataFrame(columns=columns)
                position = 0
                observed = 0
                try:
                    for expected, packed in connection.execute(
                        "SELECT row_count, summary FROM runs WHERE shard_id=? ORDER BY simulation_id", (shard_id,),
                    ):
                        summary = _unpacked(packed)
                        pieces = []
                        remaining = expected
                        while remaining:
                            if position == len(frame):
                                frame = next(chunks, None)
                                if frame is None:
                                    raise ValueError("Atlas shard ended before its catalog run count")
                                position = 0
                            take = min(remaining, len(frame) - position)
                            if split is None or summary["split"] == split:
                                pieces.append(frame.iloc[position:position + take])
                            remaining -= take
                            position += take
                            observed += take
                        if split is None or summary["split"] == split:
                            rows = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
                            if len(rows) and (not rows["run_id"].eq(summary["run_id"]).all()
                                              or not rows["structure_id"].eq(summary["structure_id"]).all()
                                              or not rows["split"].eq(summary["split"]).all()):
                                raise ValueError("Atlas run rows differ from their frozen identity")
                            yield rows, summary
                    if observed != count or position < len(frame) or next(chunks, None) is not None:
                        raise ValueError("Atlas shard contains rows outside its run catalog")
                finally:
                    if reader is not None:
                        reader.close()
        finally:
            connection.close()

    def verify_integrity(self, monitor=None) -> None:
        """Verify all compressed files in constant memory, including zero-row shards."""
        connection = _connect(self.root / CATALOG_FILENAME, readonly=True)
        try:
            for _, filename, _, size, digest in self._shards(connection, None):
                _verified_shard(self.root, filename, size, digest)
                _check(monitor, "dataset integrity")
        finally:
            connection.close()
