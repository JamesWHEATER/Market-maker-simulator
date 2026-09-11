"""Backtest regression tests using known windows and independent path observations."""
import csv
import json
import math
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_limits

import market_structure_experiment as ms
from chart_renderer import build_candles
from market_maker_simulator import SyntheticMarketConfig, simulate_market
from pattern_detector import DetectionResult, PatternDetection, RunContext, detect_pattern_universe


def event(entry=45, observation=44, event_id="provenance-a"):
    return PatternDetection(
        pattern_name="double_top", start_index=entry-20, end_index=entry-5,
        expected_direction="bearish", geometry_fit_score=0.8, available_at_index=entry-1,
        event_id=event_id, metadata={"observation_window_length": observation,
            "breakout_confirmed_by_availability": True,
            "breakout_direction_by_availability": "bearish"},
    )


def detection(events):
    return DetectionResult(tuple(events), tuple(events), (), "test", "test", "test", "test", "test")


def candles(count=180):
    rng = np.random.default_rng(88)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, count)))
    return [SimpleNamespace(open=c, high=c*1.01, low=c*0.99, close=c, volume=100) for c in closes]


def evaluate(result, bars, procedure, **kwargs):
    arrays = dict(entry_spreads=[0.001]*len(bars), exit_spreads=[0.001]*len(bars), entry_mids=None, exit_mids=None)
    cache = ms._prepare_path_evaluation_cache(result=result, candles=bars, procedure=procedure, **arrays)
    observations = ms._evaluate_hypothesis_on_path(
        result=result, candles=bars, procedure=procedure, cache=cache,
        pattern_name="double_top", horizon=procedure.horizons[0], run_identity="display-run", **arrays, **kwargs,
    )[0]
    return cache, observations


def metric(seed, observations):
    return ms._metric_from_observations(partition="validation", structure=ms.ScenarioSpec("S", "S", ()),
        scenario_id="S", run_id=f"run-{seed}", seed=seed, pattern_name="double_top", horizon=20,
        observations=observations, skipped_overlap=0, untradeable=0)


class MatchedNullTests(unittest.TestCase):
    def test_identical_real_csv_bytes_different_filenames_have_identical_economics(self):
        with tempfile.TemporaryDirectory() as root, threadpool_limits(limits=1):
            path = Path(root) / "original.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
                for index, bar in enumerate(candles(300)):
                    writer.writerow([(datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)).isoformat(),
                        bar.open, bar.high, bar.low, bar.close, bar.volume])
            renamed = Path(root) / "renamed.csv"
            renamed.write_bytes(path.read_bytes())
            proc = ms.ProcedureConfig(horizons=(5,))
            spec = ms.RealDatasetSpec("asset-a", "venue", "1h", "raw", "none_in_window")
            first = ms.run_real_validation([path], [(ms.ALL_PATTERNS_LABEL, 5)], proc, dataset_specs=[spec], git_commit="a")
            second = ms.run_real_validation([renamed], [(ms.ALL_PATTERNS_LABEL, 5)], proc,
                dataset_specs=[replace(spec, asset="display-b", timeframe="60m")], git_commit="b")
            self.assertGreater(first[0].event_count, 0)
            self.assertGreater(first[0].matched_null_event_count, 0)
            self.assertEqual(ms._result_jsonable(asdict(replace(first[0], structure_label=""))),
                             ms._result_jsonable(asdict(replace(second[0], structure_label=""))))

    def test_ai_labels_equal_statistical_quote_execution_returns(self):
        import ai_pattern_research_pipeline as ai
        cfg = SyntheticMarketConfig(steps=1500, seed=17, world_weights={"momentum": 1.0})
        with threadpool_limits(limits=1):
            history = simulate_market(cfg, record_step_diagnostics=False)
            bars = build_candles(history, 5, include_partial=False)
            result = detect_pattern_universe(bars)
            entries, exits, entry_spreads, exit_spreads = ms.synthetic_execution_arrays(history, bars)
            experiment = ai.AIExperimentConfig(steps_per_run=1500, label_horizon_candles=5)
            costs = ms.CostModel(brokerage_bps_per_side=3, slippage_bps_per_side=2)
            spec = ai.RunSpec(0, 0, ai._structure_id(cfg), "structured", cfg)
            checked = 0
            for signal in result.confirmed_breakout_events:
                row = ai.create_pattern_row(spec, history, bars, signal, entries, exits, entry_spreads, exit_spreads, experiment, costs)
                if row is None:
                    continue
                expected = ms._evaluate_index_return(bars, entry_spreads, exit_spreads, signal.earliest_execution_index,
                    5, ms.trade_direction(signal), costs, entry_mids=entries, exit_mids=exits)
                self.assertEqual(row["gross_directional_return"], expected[0])
                self.assertEqual(row["total_transaction_cost"], expected[3])
                self.assertEqual(row["net_directional_return"], expected[4])
                checked += 1
            self.assertGreater(checked, 0)

    def test_controls_exclude_entire_return_windows_plus_embargo(self):
        chosen = ms._candidate_null_indices(50, range(10, 200), [0.1]*220, [0.001]*220,
            [50, 130], 5, 1000, random.Random(1), horizon=20)
        self.assertTrue(chosen)
        for entry in chosen:
            for signal in (50, 130):
                self.assertTrue(entry + 19 < signal - 5 or entry > signal + 19 + 5)
        self.assertIn(25, chosen)
        self.assertNotIn(26, chosen)
        self.assertNotIn(74, chosen)
        self.assertIn(75, chosen)

    def test_event_scale_controls_and_unconditional_population_share_warmup(self):
        result = detection([event(), event(110, 96, "large")])
        cache, observations = evaluate(result, candles(), ms.ProcedureConfig(horizons=(20,)))
        self.assertEqual(cache.eligible_by_horizon[(20, 44)][0], 44)
        self.assertEqual(cache.eligible_by_horizon[(20, 96)][0], 96)
        self.assertEqual([o.entry_index for o in observations], [45, 110])

    def test_metadata_and_run_labels_do_not_change_control_sampling(self):
        bars = candles()
        procedure = ms.ProcedureConfig(horizons=(20,), matched_nulls_per_event=5)
        _, first = evaluate(detection([event(event_id="filename-and-commit-a")]), bars, procedure)
        _, second = evaluate(detection([event(event_id="filename-and-commit-b")]), bars, procedure)
        self.assertGreater(first[0].matched_null_count, 0)
        self.assertEqual(replace(first[0], event_id=""), replace(second[0], event_id=""))

    def test_sampling_seed_uses_only_causal_economic_prefix(self):
        bars = candles()
        prefix = ms._economic_prefix_fingerprints(bars[:60])
        full = ms._economic_prefix_fingerprints(bars)
        self.assertEqual(prefix, full[:61])
        self.assertEqual(ms._null_seed(prefix[45], "double_top", 45, 20),
                         ms._null_seed(full[45], "double_top", 45, 20))

    def test_no_signal_free_controls_preserves_strategy_but_suppresses_matched_inference(self):
        result = detection([event(entry=i, observation=1, event_id=str(i)) for i in range(25, 150, 10)])
        _, observations = evaluate(result, candles(150), ms.ProcedureConfig(horizons=(20,)))
        self.assertTrue(observations)
        self.assertTrue(all(math.isnan(o.matched_null_mean_net_return) for o in observations))
        row = ms.aggregate_metrics([metric(1, observations)], ms.ProcedureConfig())[0]
        self.assertEqual(row["raw_p_value_primary_edge"], 1.0)

    def test_real_synthetic_detections_ignore_git_label_in_matched_results(self):
        cfg = SyntheticMarketConfig(steps=1500, seed=17, world_weights={"momentum": 1.0})
        with threadpool_limits(limits=1):
            bars = build_candles(simulate_market(cfg, record_step_diagnostics=False), 5, include_partial=False)
            results = [detect_pattern_universe(bars, run_context=RunContext(git_commit=label)) for label in ("a", "b")]
            self.assertTrue(results[0].confirmed_breakout_events)
            for result in results:
                # Include every signal family while comparing their economic observations.
                self.assertTrue(all(e.metadata.get("competing_pattern") is not None for e in result.confirmed_breakout_events))
            proc = ms.ProcedureConfig(horizons=(5,))
            all_observations = []
            for result in results:
                arrays = dict(entry_spreads=[0.001]*len(bars), exit_spreads=[0.001]*len(bars), entry_mids=None, exit_mids=None)
                cache = ms._prepare_path_evaluation_cache(result=result, candles=bars, procedure=proc, **arrays)
                observed = ms._evaluate_hypothesis_on_path(result=result, candles=bars, procedure=proc, cache=cache,
                    pattern_name=ms.ALL_PATTERNS_LABEL, horizon=5, run_identity="display-only", **arrays)[0]
                all_observations.append(ms._result_jsonable([asdict(replace(o, event_id="")) for o in observed]))
            self.assertTrue(all_observations[0])
            self.assertEqual(*all_observations)


class InferenceContractTests(unittest.TestCase):
    def test_real_cli_applies_frozen_procedure_and_pools_temporal_clusters(self):
        with tempfile.TemporaryDirectory() as root, threadpool_limits(limits=1):
            directory = Path(root)
            paths = []
            for year in (2022, 2024, 2026):
                path = directory / f"fixture-{year}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
                    for index, bar in enumerate(candles(150)):
                        writer.writerow([(datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)).isoformat(),
                            bar.open, bar.high, bar.low, bar.close, bar.volume])
                paths.append(path)
            procedure = ms.ProcedureConfig(horizons=(5,))
            # Artificial selected hypothesis solely exercises the real-validation CLI contract.
            manifest = ms._selected_payload([{"structure_id":"fixture", "pattern_name":ms.ALL_PATTERNS_LABEL,
                "horizon":5}], procedure, synthetic_design_hash="0"*64, status="synthetic_validation_complete")
            manifest_path = directory / "fixture-validated.json"
            ms._write_json(manifest, manifest_path)
            metadata_path = directory / "datasets.json"
            ms._write_json({"datasets":[{"path":str(path), **asdict(ms.RealDatasetSpec(
                "asset", "venue", "1h", "raw", "none_in_window"))} for path in paths]}, metadata_path)
            output = directory / "output"
            args = SimpleNamespace(command="real", output_dir=output, validated_manifest=manifest_path,
                input=paths, dataset_manifest=metadata_path, timeframe=None, git_commit="fixture")
            with mock.patch.object(ms, "parse_args", return_value=args):
                ms.main()
            record = json.loads((output / "real_validation_manifest.json").read_text())
            self.assertEqual(record["dependence_cluster_count"], 3)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["procedure_hash"], ms.procedure_hash(procedure))
            with (output / "real_pooled_validation.csv").open() as handle:
                pooled = next(csv.DictReader(handle))
            self.assertEqual(pooled["seed_count_total"], "3")
            self.assertIn("net_return_points_sum_across_seeds", pooled)

    def test_failed_real_rerun_invalidates_previous_results(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(command="real", output_dir=Path(root))
            status_path = Path(root) / "experiment_run_status.json"
            status_path.write_text('{"status":"COMPLETE","confirmatory_results_valid":true}')
            with mock.patch.object(ms, "parse_args", return_value=args), \
                    mock.patch.object(ms, "_real_main", side_effect=ValueError("invalid new data")):
                with self.assertRaises(ValueError):
                    ms.main()
            status = json.loads(status_path.read_text())
        self.assertEqual(status["status"], "FAILED")
        self.assertFalse(status["confirmatory_results_valid"])

    def test_direction_interval_counts_runs_not_events(self):
        base = metric(1, [])
        a = replace(base, event_count=1000, direction_correct_count=1000, direction_accuracy=1.0)
        b = replace(base, run_id="run-2", seed=2, event_count=1, direction_accuracy=0.0)
        row = ms.aggregate_metrics([a, b], ms.ProcedureConfig())[0]
        self.assertEqual(row["direction_accuracy"], 0.5)
        self.assertEqual((row["direction_accuracy_ci_low"], row["direction_accuracy_ci_high"]), (0, 1))
        self.assertGreater(row["direction_accuracy_event_weighted_descriptive"], 0.99)

    def test_direction_lift_uses_same_matched_event_subset(self):
        base = replace(metric(1, []), event_count=10, direction_accuracy=0.9, matched_null_event_count=1,
            mean_effect_vs_matched=0.01, matched_event_direction_accuracy=0.0, matched_null_direction_accuracy=0.5)
        row = ms.aggregate_metrics([base], ms.ProcedureConfig())[0]
        self.assertEqual(row["direction_accuracy_lift_vs_matched"], -0.5)

    def test_duplicate_runs_and_mixed_partitions_are_rejected(self):
        base = metric(1, [])
        for rows in ([base, base], [base, replace(base, run_id="different")],
                     [base, replace(base, seed=2, run_id="2", partition="discovery")]):
            with self.assertRaises(ValueError):
                ms.aggregate_metrics(rows, ms.ProcedureConfig())

    def test_missing_selected_hypothesis_cannot_shrink_holm_family(self):
        with self.assertRaisesRegex(ValueError, "every pre-selected"):
            ms.validate_selected_hypotheses([], [{"structure_id":"S", "pattern_name":"double_top", "horizon":20}], ms.ProcedureConfig())

    def test_manifest_does_not_coerce_or_default_invalid_settings(self):
        valid = {"procedure": asdict(ms.ProcedureConfig())}
        self.assertEqual(ms._procedure_from_manifest(valid), ms.ProcedureConfig())
        for field, value in (("steps_per_candle", 5.5), ("horizons", [1, True]), ("alpha", "0.05")):
            altered = {"procedure": {**valid["procedure"], field: value}}
            with self.assertRaises(ValueError):
                ms._procedure_from_manifest(altered)
        altered = {"procedure": dict(valid["procedure"])}
        del altered["procedure"]["minimum_matched_nulls_per_event"]
        with self.assertRaises(ValueError):
            ms._procedure_from_manifest(altered)

    def test_additive_outputs_and_rankings_use_return_point_names(self):
        base = replace(metric(1, []), pattern_name=ms.ALL_PATTERNS_LABEL, total_net_return=0.5)
        row = ms.aggregate_metrics([base], ms.ProcedureConfig())[0]
        row.update(validated=True, validation_holm_p_value=0.01)
        ranked = ms.structure_rankings([row])[0]
        self.assertEqual(ranked["best_portfolio_mean_net_return_points_per_seed"], 0.5)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metrics.csv"
            ms._write_run_metrics([base], path)
            with path.open() as handle:
                record = next(csv.DictReader(handle))
        self.assertEqual(record["net_return_points"], "0.5")
        self.assertNotIn("total_net_return", record)

    def test_explicit_malformed_close_quotes_never_take_legacy_fallback(self):
        for bid, ask in ((float("nan"), 101), (102, 101), (0, 101)):
            history = [SimpleNamespace(step=1, bid=99, ask=101, post_trade_bid=bid, post_trade_ask=ask)]
            with self.assertRaisesRegex(ValueError, "closing quote"):
                ms.synthetic_execution_arrays(history, [SimpleNamespace(start_step=1, end_step=1)])


if __name__ == "__main__":
    unittest.main()
