"""Real-data backtest contracts: chronology, executable quotes, and independent units."""
import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import market_structure_experiment as experiment


class RealValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.procedure = experiment.ProcedureConfig(horizons=(1, 20))

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, name="bars.csv", *, rows=None, extras=(), start=None, hours=(0, 1, 2)):
        path = self.root / name
        if rows is None:
            beginning = start or self.start
            rows = [
                [(beginning + timedelta(hours=h)).isoformat(), 100, 102, 98, 101, *extras]
                for h in hours
            ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "open", "high", "low", "close", *getattr(self, "extra_headers", ())])
            writer.writerows(rows)
        return path

    def spec(self, asset="BTC", **kwargs):
        return experiment.RealDatasetSpec(
            asset=asset, venue="TEST", timeframe="1h", price_basis="raw",
            corporate_action_policy="none_in_window", **kwargs,
        )

    def load(self, path, **kwargs):
        return experiment.load_real_csv(path, 5.0, **kwargs)

    def test_preserves_boundary_quote_midpoints_and_separate_spreads(self):
        self.extra_headers = ("open_bid", "open_ask", "close_bid", "close_ask")
        data = self.load(self.write_csv(extras=(89, 91, 108, 112)), timeframe="1h")
        self.assertEqual(data.entry_mids, [90.0] * 3)
        self.assertEqual(data.exit_mids, [110.0] * 3)
        self.assertAlmostEqual(data.entry_spreads[0], 2 / 90)
        self.assertAlmostEqual(data.exit_spreads[0], 4 / 110)
        costs = experiment.CostModel(brokerage_bps_per_side=0)
        actual = experiment._evaluate_index_return(
            data.candles, data.entry_spreads, data.exit_spreads, 0, 1, 1, costs,
            entry_mids=data.entry_mids, exit_mids=data.exit_mids,
        )
        expected = experiment.net_trade_return(90, 110, 1, 2 / 90, 4 / 110, costs)
        self.assertEqual(actual, expected)
        self.assertEqual(data.execution_source, "explicit_open_close_quotes")

    def test_fallback_is_explicit_modeled_execution(self):
        data = self.load(self.write_csv(), timeframe="60m")
        self.assertEqual(data.entry_mids, [100.0] * 3)
        self.assertEqual(data.exit_mids, [101.0] * 3)
        self.assertEqual(data.entry_spreads, [0.0005] * 3)
        self.assertEqual(data.execution_source, "modeled_ohlc_mid_with_frozen_spread")

    def test_ambiguous_and_partial_quotes_are_rejected(self):
        for headers, values in ((('bid', 'ask'), (99, 101)), (('open_bid',), (99,))):
            with self.subTest(headers=headers):
                self.extra_headers = headers
                with self.assertRaisesRegex(ValueError, "bid/ask timing|all four"):
                    self.load(self.write_csv(extras=values))

    def test_incomplete_adjusted_ohlc_never_replaces_raw_close(self):
        path = self.write_csv()
        path.write_text(path.read_text().replace("low,close", "low,adj_close"))
        with self.assertRaisesRegex(ValueError, "missing required CSV column"):
            self.load(path)
        with self.assertRaisesRegex(ValueError, "missing required CSV column"):
            self.load(path, price_basis="adjusted")

    def test_complete_adjusted_ohlc_is_supported(self):
        path = self.write_csv()
        path.write_text(path.read_text().replace("open,high,low,close", "adj_open,adj_high,adj_low,adj_close"))
        self.assertEqual(self.load(path, price_basis="adjusted").candles[0].close, 101)

    def test_regular_timezone_aware_timestamps_are_required(self):
        for hours in ((0, 1, 3), (0, 1, 1), (2, 1, 0)):
            with self.subTest(hours=hours):
                with self.assertRaisesRegex(ValueError, "regularly spaced"):
                    self.load(self.write_csv(hours=hours))
        for timestamp in ("2024-01-01", "1704067200", "2024-01-01T00:00:00", ""):
            with self.subTest(timestamp=timestamp):
                rows = [[timestamp, 100, 102, 98, 101]] * 3
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    self.load(self.write_csv(rows=rows))
        with self.assertRaisesRegex(ValueError, "does not match timeframe"):
            self.load(self.write_csv(), timeframe="5m")
        with self.assertRaisesRegex(ValueError, "explicit fixed duration"):
            self.load(self.write_csv(), timeframe="pre_registered")

    def test_missing_and_duplicate_timestamp_columns_fail_closed(self):
        path = self.write_csv()
        original = path.read_text()
        path.write_text(original.replace("timestamp", "row_number"))
        with self.assertRaisesRegex(ValueError, "missing required CSV column"):
            self.load(path)
        path.write_text(original.replace("timestamp,open", "timestamp,Timestamp"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load(path)

    def test_cluster_overlapping_correlated_assets_and_merge_transitively(self):
        datasets = [self.load(self.write_csv(str(i) + ".csv", start=self.start + timedelta(hours=i * 24))) for i in range(3)]
        clusters = experiment._real_dataset_clusters(datasets, [self.spec("BTC"), self.spec("ETH"), self.spec("SOL")], self.procedure)
        self.assertEqual(clusters, [0, 0, 0])
        far = self.load(self.write_csv("far.csv", start=self.start + timedelta(days=365)))
        self.assertEqual(experiment._real_dataset_clusters([datasets[0], far], [self.spec(), self.spec()], self.procedure), [0, 1])
        self.assertEqual(experiment._real_dataset_clusters([datasets[0], far], [self.spec(dependence_group="shared"), self.spec(dependence_group="shared")], self.procedure), [0, 0])

    def test_same_asset_overlap_and_mixed_durations_are_rejected(self):
        first = self.load(self.write_csv())
        with self.assertRaisesRegex(ValueError, "same asset"):
            experiment._real_dataset_clusters([first, first], [self.spec(), self.spec()], self.procedure)
        other = self.load(self.write_csv("other.csv", hours=(0, 4, 8)))
        with self.assertRaisesRegex(ValueError, "different bar durations"):
            experiment._real_dataset_clusters([first, other], [self.spec(), self.spec("ETH")], self.procedure)

    def test_explicit_manifest_matches_input_files_and_enforces_price_policy(self):
        path = self.write_csv()
        manifest = self.root / "datasets.json"
        manifest.write_text(json.dumps({"datasets": [{"path": path.name, "asset": "BTC", "venue": "TEST", "timeframe": "1h", "price_basis": "raw", "corporate_action_policy": "none_in_window"}]}))
        self.assertEqual(experiment._load_real_dataset_manifest(manifest, [path]), [self.spec()])
        with self.assertRaisesRegex(ValueError, "exactly"):
            experiment._load_real_dataset_manifest(manifest, [])
        with self.assertRaisesRegex(ValueError, "corporate_action_policy"):
            experiment.RealDatasetSpec("BTC", "TEST", "1h", "raw", "unknown")
        with self.assertRaisesRegex(ValueError, "explicit RealDatasetSpec"):
            experiment.run_real_validation([path], [(experiment.ALL_PATTERNS_LABEL, 1)], self.procedure, timeframe="1h")

    def test_real_evaluation_passes_exact_arrays_and_provenance_free_identity(self):
        self.extra_headers = ("open_bid", "open_ask", "close_bid", "close_ask")
        first = self.write_csv(extras=(89, 91, 108, 112))
        renamed = self.root / "renamed.csv"
        renamed.write_bytes(first.read_bytes())
        records = []
        with patch.object(experiment, "_evaluate_hypothesis_on_path", return_value=([], 0, 0)) as evaluate:
            experiment.run_real_validation([first], [(experiment.ALL_PATTERNS_LABEL, 1)], self.procedure, dataset_specs=[self.spec()], dataset_records=records, git_commit="first")
            first_call = evaluate.call_args.kwargs
            experiment.run_real_validation([renamed], [(experiment.ALL_PATTERNS_LABEL, 1)], self.procedure, dataset_specs=[replace(self.spec("label"), timeframe="60m")], git_commit="second")
            second_call = evaluate.call_args.kwargs
        self.assertEqual(first_call["entry_mids"], [90] * 3)
        self.assertEqual(first_call["exit_mids"], [110] * 3)
        self.assertEqual(first_call["run_identity"], second_call["run_identity"])
        self.assertEqual(records[0]["bar_duration_seconds"], 3600)
        self.assertEqual(records[0]["dependence_cluster"], 0)

    def test_pooling_collapses_dependence_clusters_before_inference(self):
        base = experiment._metric_from_observations(
            partition="real_validation", structure=experiment.ScenarioSpec("one", "one", ()),
            scenario_id="one", run_id="one", seed=0, pattern_name=experiment.ALL_PATTERNS_LABEL,
            horizon=1, observations=[], skipped_overlap=0, untradeable=0,
        )
        metric = replace(base, event_count=2, matched_null_event_count=1, total_net_return=0.4, total_gross_return=0.5, total_transaction_cost=0.1, mean_net_return=0.2, mean_effect_vs_matched=0.1, mean_matched_null_net_return=0.1, mean_effect_vs_unconditional=0.1, mean_unconditional_net_return=0.1, direction_correct_count=2, direction_accuracy=1.0, matched_null_direction_accuracy=0.5, matched_event_direction_accuracy=1.0)
        second = replace(metric, structure_id="two", run_id="two", event_count=1, total_net_return=-0.1, mean_net_return=-0.1, direction_correct_count=0, direction_accuracy=0.0, matched_event_direction_accuracy=0.0)
        pooled = experiment._pooled_real_cluster_metrics([metric, second])
        self.assertEqual(len(pooled), 1)
        self.assertAlmostEqual(pooled[0].total_net_return, 0.3)
        self.assertEqual(pooled[0].event_count, 3)
        self.assertAlmostEqual(pooled[0].mean_net_return, 0.1)
        self.assertEqual(pooled[0].matched_event_direction_accuracy, 0.5)
        self.assertTrue(math.isnan(pooled[0].median_net_return))
        rows = experiment.aggregate_metrics(pooled, self.procedure)
        self.assertEqual(rows[0]["seed_count_total"], 1)
        self.assertEqual(rows[0]["raw_p_value_primary_edge"], 1.0)


if __name__ == "__main__":
    unittest.main()
