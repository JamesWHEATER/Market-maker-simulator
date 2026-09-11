"""Atlas orchestration: bounded generation, disk training and streamed evaluation."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from atlas_resources import DEFAULT_ATLAS_DIR, ResourceMonitor, atomic_json, require_d_destination


def add_streaming_arguments(parser):
    parser.add_argument("--legacy-in-memory", action="store_true",
                        help="Historical small-data implementation; never use for Atlas.")
    parser.add_argument("--runs-per-shard", type=int, default=32)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--logistic-epochs", type=int, default=10)
    parser.add_argument("--xgb-rounds", type=int, default=300)
    parser.add_argument("--explanation-rows", type=int, default=512)
    parser.add_argument("--memory-budget-gib", type=float, default=None,
                        help="Optional lower budget; default adapts to available RAM, capped at 6 GiB.")
    parser.add_argument("--reserve-ram-gib", type=float, default=1.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume verified committed generation shards; model training restarts.")
    return parser


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze_training_protocol(root, dataset, experiment, args):
    names = ("atlas_pipeline.py", "atlas_streaming_models.py", "atlas_streaming_evaluation.py",
             "atlas_streaming_data.py", "ai_pattern_research_pipeline.py")
    protocol = {
        "dataset_contract_sha256": dataset.manifest["contract_sha256"],
        "experiment": asdict(experiment), "feature_set": args.feature_set,
        "batch_rows": args.batch_rows, "logistic_epochs": args.logistic_epochs,
        "xgb_rounds": args.xgb_rounds, "explanation_rows": args.explanation_rows,
        "sources": {name: _source_hash(Path(__file__).with_name(name)) for name in names},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost")},
        "method_changes": ["train-only mean imputation with streaming scale statistics",
                           "incremental SGD logistic regression; validate optimizer against a bounded reference",
                           "CPU MLP minibatches read anew each epoch",
                           "CPU XGBoost ExtMemQuantileDMatrix with disk feature cache",
                           "streamed structure-cluster inference; histogram rank metrics are approximate"],
        "test_policy": "Freeze model thresholds and primary model using validation before reading test outcomes.",
    }
    path = root / "streaming_training_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen streaming training protocol differs; preserve this run and use a fresh experiment directory")
    atomic_json(path, protocol)


def run_streaming_cli(args):
    """Lock the result directory and invalidate old results before fallible setup."""
    from atlas_streaming_data import _generation_lock
    root = require_d_destination(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with _generation_lock(root):
        started = time.monotonic()
        status_path = root / "atlas_status.json"
        atomic_json(status_path, {"status": "RUNNING", "confirmatory_results_valid": False,
                                 "command": args.command, "error": None})
        try:
            _run_streaming_cli_impl(args)
        except BaseException as exc:
            atomic_json(status_path, {
                "status": "INSUFFICIENT_VALIDATION_SUPPORT" if "INSUFFICIENT_VALIDATION_SUPPORT" in str(exc) else "FAILED",
                "confirmatory_results_valid": False, "command": args.command, "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "updated_utc": datetime.now(timezone.utc).isoformat()})
            raise


def _run_streaming_cli_impl(args):
    # Import only after CLI parsing so --help needs no optional model runtimes.
    import ai_pattern_research_pipeline as ai
    from atlas_streaming_data import StreamingDataset, generate_streaming_dataset
    from atlas_streaming_models import train_streaming_models
    from atlas_streaming_evaluation import evaluate_streaming_models
    from market_structure_experiment import CostModel

    root = require_d_destination(args.output_dir)
    if args.jobs not in (1, 2):
        raise ValueError("On the 16 GB Atlas profile use --jobs 1 or 2; unbounded worker counts are disabled")
    for name in ("batch_rows", "runs_per_shard", "logistic_epochs", "xgb_rounds", "explanation_rows"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.batch_rows > 65536 or args.runs_per_shard > 256 or args.explanation_rows > 4096:
        raise ValueError("Atlas limits: batch-rows <= 65536, runs-per-shard <= 256, explanation-rows <= 4096")
    root.mkdir(parents=True, exist_ok=True)
    # All library scratch data inherits D:, including tempfile's cached directory.
    scratch = root / "tmp"
    scratch.mkdir(exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(scratch)
    tempfile.tempdir = str(scratch)
    data_root = root / "data"
    status_path = root / "atlas_status.json"
    started = time.monotonic()
    monitor = ResourceMonitor(root, memory_budget_gib=args.memory_budget_gib,
                              reserve_ram_gib=args.reserve_ram_gib, min_free_disk_gib=args.min_free_disk_gib)
    def status(value, error=None):
        atomic_json(status_path, {"status": value, "error": error, "command": args.command,
                                 "confirmatory_results_valid": value == "EVALUATION_COMPLETE",
                                 "updated_utc": datetime.now(timezone.utc).isoformat(),
                                 "elapsed_seconds": time.monotonic() - started})
    status("RUNNING")
    try:
        with monitor:
            if args.command == "train":
                dataset = StreamingDataset(data_root)
                experiment = ai.AIExperimentConfig(**dataset.manifest["experiment"])
                costs = CostModel(**dataset.manifest["costs"])
            else:
                experiment = ai.experiment_from_args(args)
                costs = CostModel(brokerage_bps_per_side=args.brokerage_bps,
                                  slippage_bps_per_side=args.slippage_bps)
                if args.reference_procedure_manifest:
                    reference = ai.validate_reference_procedure_manifest(args.reference_procedure_manifest, experiment, costs)
                    atomic_json(root / "reference_statistical_procedure.json", reference)
                print(f"Atlas design: {experiment.structures:,} structures x {experiment.seeds_per_structure} seeds; output {root}", flush=True)
                generation_started = time.monotonic()
                dataset = generate_streaming_dataset(
                    data_root, experiment, costs=costs, ranges=ai.WorldSamplingRanges(), jobs=args.jobs,
                    runs_per_shard=args.runs_per_shard, monitor=monitor,
                    resume=args.resume or (args.checkpoint and (data_root / "atlas_catalog.sqlite3").exists()))
                atomic_json(root / "generation_timing.json", {
                    "elapsed_seconds": time.monotonic() - generation_started,
                    "generated_runs": dataset.manifest["generated_runs"],
                    "resume_invocation": bool(args.resume or args.checkpoint),
                    "note": "Resume timing covers this invocation only; do not extrapolate as a complete generation pilot."})
            if args.command in ("all", "train"):
                if args.reference_procedure_manifest:
                    ai.validate_reference_procedure_manifest(args.reference_procedure_manifest, experiment, costs)
                _freeze_training_protocol(root, dataset, experiment, args)
                comparisons = []
                sets = ("observable", "oracle") if args.feature_set == "both" else (args.feature_set,)
                for feature_set in sets:
                    monitor.check("preprocess")
                    models = train_streaming_models(
                        dataset, experiment, root, feature_set=feature_set, batch_rows=args.batch_rows,
                        logistic_epochs=args.logistic_epochs, xgb_rounds=args.xgb_rounds, monitor=monitor)
                    comparison = evaluate_streaming_models(
                        dataset, models, experiment, root, feature_set=feature_set, batch_rows=args.batch_rows,
                        monitor=monitor, explanation_rows=args.explanation_rows)
                    comparisons.append(comparison)
                    del models
                    gc.collect()
                import pandas as pd
                # This table has at most six model-summary rows, never observations.
                pd.concat(comparisons, ignore_index=True).to_csv(root / "model_comparison_all.csv", index=False)
                monitor.check("test_complete")
        status("EVALUATION_COMPLETE" if args.command in ("all", "train") else "DATASET_COMPLETE")
    except BaseException as exc:
        status("INSUFFICIENT_VALIDATION_SUPPORT" if "INSUFFICIENT_VALIDATION_SUPPORT" in str(exc) else "FAILED", str(exc))
        atomic_json(root / "resource_report.json", monitor.report())
        raise
    print(f"Atlas {args.command} complete: {root}", flush=True)


def main():
    from ai_pattern_research_pipeline import build_argument_parser
    parser = build_argument_parser()
    parser.set_defaults(output_dir=DEFAULT_ATLAS_DIR, seeds_per_structure=2)
    args = parser.parse_args()
    if args.legacy_in_memory:
        parser.error("Use the historical research entry point for --legacy-in-memory")
    run_streaming_cli(args)


if __name__ == "__main__":
    main()
