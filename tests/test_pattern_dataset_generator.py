"""Leakage, label-boundary, split and streaming regressions without model fitting."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import ai_pattern_dataset_generator as generator
import ai_pattern_research_pipeline as ai
from chart_renderer import build_candles
from market_maker_simulator import SyntheticMarketConfig, run_id, simulate_market
from market_structure_experiment import CostModel, synthetic_execution_arrays
from pattern_detector import PatternDetection, detect_pattern_universe


class CausalRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SyntheticMarketConfig(steps=1500, seed=17, world_weights={"momentum": 1.0})
        with threadpool_limits(limits=1):
            cls.history = simulate_market(cls.config, record_step_diagnostics=False)
            cls.bars = build_candles(cls.history, 5)
            cls.detections = detect_pattern_universe(cls.bars)
        cls.spec = ai.RunSpec(0, 0, ai._structure_id(cls.config), "structured", cls.config)
        cls.experiment = ai.AIExperimentConfig(steps_per_run=1500)
        cls.arrays = synthetic_execution_arrays(cls.history, cls.bars)
        cls.signal = next(e for e in cls.detections.confirmed_breakout_events
                          if e.earliest_execution_index + 19 < len(cls.bars))

    def row(self, *, signal=None, history=None, bars=None, arrays=None, costs=CostModel(), spec=None):
        return ai.create_pattern_row(
            spec or self.spec, history if history is not None else self.history,
            bars if bars is not None else self.bars, signal or self.signal,
            *(arrays or self.arrays), self.experiment, costs,
        )

    def assert_features_equal(self, first, second, feature_set="oracle", *, tolerance=0):
        numeric, categorical = ai.feature_columns(feature_set)
        np.testing.assert_allclose([first[n] for n in numeric], [second[n] for n in numeric],
                                   rtol=tolerance, atol=tolerance, equal_nan=True)
        self.assertEqual([first[n] for n in categorical], [second[n] for n in categorical])

    def test_mutating_every_future_bar_changes_labels_but_no_features(self):
        available = self.signal.available_at_index
        cutoff = self.bars[available].end_step
        history = self.history[:cutoff] + [replace(
            s, bid=s.bid * 2, ask=s.ask * 2, post_trade_bid=s.post_trade_bid * 3,
            post_trade_ask=s.post_trade_ask * 3, trade_price=s.trade_price * 3,
            liquidity=99, sentiment=99, signed_order_flow=-100000, buy_volume=0, sell_volume=100000,
        ) for s in self.history[cutoff:]]
        bars = self.bars[:available + 1] + [replace(
            c, open=c.open * 3, high=c.high * 3, low=c.low * 3, close=c.close * 3, volume=999999,
        ) for c in self.bars[available + 1:]]
        changed = self.row(history=history, bars=bars, arrays=synthetic_execution_arrays(history, bars))
        original = self.row()
        self.assertNotEqual(original["net_directional_return"], changed["net_directional_return"])
        self.assert_features_equal(original, changed)

    def test_detector_prefix_reproduces_confirmation_and_features(self):
        available = self.signal.available_at_index
        with threadpool_limits(limits=1):
            prefix = detect_pattern_universe(self.bars[:available + 1])
        signal = next(e for e in prefix.confirmed_breakout_events if e.event_id == self.signal.event_id)
        for name in ("pattern_name", "start_index", "end_index", "available_at_index",
                     "earliest_execution_index", "expected_direction", "event_id"):
            self.assertEqual(getattr(signal, name), getattr(self.signal, name))
        # Batched linear algebra can differ at floating-point roundoff across batch sizes.
        self.assert_features_equal(self.row(signal=signal), self.row(), tolerance=1e-12)
        self.assertEqual(self.row()["feature_end_step"], self.bars[available].end_step)

    def test_hidden_state_cannot_change_observable_features(self):
        history = [replace(s, fair_value=s.fair_value * 17,
                           reference_price=s.reference_price * 17,
                           post_trade_reference_price=s.post_trade_reference_price * 17,
                           maker_inventory=s.maker_inventory + 1000, liquidity=99, sentiment=88)
                   for s in self.history]
        altered_config = replace(self.config, world_weights={"emotional": 1.0})
        changed = self.row(history=history, spec=replace(self.spec, config=altered_config))
        original = self.row()
        self.assert_features_equal(original, changed, "observable")
        self.assertNotEqual(original["oracle_current_liquidity"], changed["oracle_current_liquidity"])
        self.assertNotEqual(original["oracle_weight_momentum"], changed["oracle_weight_momentum"])

    def test_current_spread_uses_closing_public_quote(self):
        row = self.row()
        current = self.history[row["feature_end_step"] - 1]
        expected = (current.post_trade_ask - current.post_trade_bid) / (
            current.post_trade_ask / 2 + current.post_trade_bid / 2) * 10000
        self.assertAlmostEqual(row["observed_spread_bps_current"], expected)

    def test_twenty_bar_labels_long_short_flat_and_costs(self):
        count = len(self.bars)
        for direction in ("bullish", "bearish"):
            signal = PatternDetection("double_bottom", 5, 25, direction, 0.8,
                {"breakout_confirmed_by_availability": True,
                 "breakout_direction_by_availability": direction}, 29, event_id=direction)
            for exit_mid in (95, 100, 105):
                for fee, slip in ((0, 0), (3, 2)):
                    with self.subTest(direction=direction, exit_mid=exit_mid, fee=fee):
                        entry_mids, exit_mids = [100.0] * count, [999.0] * count
                        exit_mids[49] = exit_mid  # entry 30, twenty bars inclusive => exit 49.
                        spread = 0.0 if fee == 0 else 0.002
                        row = self.row(signal=signal, arrays=(entry_mids, exit_mids,
                            [spread] * count, [spread] * count),
                            costs=CostModel(brokerage_bps_per_side=fee, slippage_bps_per_side=slip))
                        half = spread / 2 + slip / 10000
                        side = 1 if direction == "bullish" else -1
                        entry_fill = 100 * (1 + side * half)
                        exit_fill = exit_mid * (1 - side * half)
                        expected = side * (exit_fill - entry_fill) / 100
                        expected -= fee / 10000 * (entry_fill + exit_fill) / 100
                        self.assertAlmostEqual(row["net_directional_return"], expected)
                        self.assertEqual(row["profitable"], int(expected > 0))
                        self.assertEqual((row["entry_index"], row["exit_index"]), (30, 49))

    def test_incomplete_future_is_excluded_instead_of_shortening_horizon(self):
        exit_index = self.signal.earliest_execution_index + 19
        self.assertIsNone(self.row(bars=self.bars[:exit_index]))
        self.assertIsNotNone(self.row(bars=self.bars[:exit_index + 1]))

    def test_unconfirmed_geometry_is_not_a_labeled_signal(self):
        signal = replace(self.signal, metadata={**self.signal.metadata,
                                               "breakout_confirmed_by_availability": False})
        self.assertIsNone(self.row(signal=signal))


class DatasetExportTests(unittest.TestCase):
    def test_splits_ignore_labels_and_keep_repeated_seeds_together(self):
        universe = pd.DataFrame([{"structure_id": f"s{i}", "run_id": f"r{i}-{j}",
                                  "world_family": "structured", "profitable": (i + j) % 2}
                                 for i in range(10) for j in range(2)])
        first = ai.assign_structure_splits(universe, ai.AIExperimentConfig(structures=10))
        universe["profitable"] = 99
        second = ai.assign_structure_splits(universe.iloc[::-1], ai.AIExperimentConfig(structures=10))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["split"].value_counts().to_dict(), {"train": 6, "validation": 2, "test": 2})
        self.assertTrue(first["run_count"].eq(2).all())

    def test_streamed_export_never_reads_pattern_csv_or_trains_and_retains_zero_runs(self):
        experiment = ai.AIExperimentConfig(structures=10, steps_per_run=500, seeds_per_structure=2)
        numeric, categorical = ai.feature_columns("oracle")

        def fake_runs(specs, experiment, **kwargs):
            # Partitions must already exist on disk when the first simulation would begin.
            self.assertTrue((output / "structure_split_manifest.csv").exists())
            for spec in specs:
                row = {**dict.fromkeys(numeric, 0.0), **dict.fromkeys(categorical, "test"),
                       "run_id": run_id(spec.config), "structure_id": spec.structure_id,
                       "world_family": spec.world_family, "event_id": str(spec.simulation_id),
                       "entry_mid": 100.0, "exit_mid": 101.0, "profitable": 1,
                       "net_directional_return": 0.009, "entry_index": 30,
                       "exit_index": 49, "available_at_index": 29}
                rows = [] if spec.simulation_id == 0 else [row]
                yield rows, {"simulation_id": spec.simulation_id, "structure_id": spec.structure_id,
                             "run_id": row["run_id"], "world_family": spec.world_family,
                             "training_rows": len(rows), "skipped_incomplete_horizon": 1,
                             "skipped_unresolved_direction": 0}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with mock.patch.object(generator, "iter_pattern_runs", side_effect=fake_runs), \
                    mock.patch.object(ai, "train_and_evaluate_models", side_effect=AssertionError("training")), \
                    mock.patch.object(pd, "read_csv", side_effect=AssertionError("full CSV read")):
                result = generator.generate_dataset(output, experiment)
            self.assertEqual(result["rows"], 19)
            self.assertEqual(result["generated_runs"], 20)
            self.assertEqual(result["skipped_incomplete_horizon"], 20)
            schema = json.loads((output / "feature_schema.json").read_text())
            observable = sum(schema["model_a_observable"].values(), [])
            oracle = sum(schema["model_b_oracle"].values(), [])
            self.assertTrue(set(observable) < set(oracle))
            self.assertFalse(any(name.startswith("oracle_") for name in observable))
            self.assertNotIn("profitable_after_costs", oracle)
            self.assertNotIn("future_20_bar_return", oracle)
            master = pd.read_csv(output / "synthetic_pattern_dataset.csv")
            parts = pd.concat([pd.read_csv(output / f"{split}.csv") for split in generator.PARTITIONS])
            self.assertEqual(set(master.event_id), set(parts.event_id))
            self.assertFalse(parts.duplicated(["run_id", "event_id"]).any())
            np.testing.assert_allclose(master.future_20_bar_return, 0.01)
            self.assertTrue(master.profitable_after_costs.eq(1).all())
            self.assertTrue(master.groupby("structure_id")["split"].nunique().eq(1).all())
            status = json.loads((output / "dataset_status.json").read_text())
            self.assertEqual(status["status"], "DATASET_COMPLETE")
            ai.load_frozen_research_manifest(output / "ai_research_manifest.json")
            with self.assertRaises(FileExistsError):
                generator.generate_dataset(output, experiment)
            with (output / "synthetic_pattern_dataset.csv").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "changed after generation"):
                ai.load_frozen_research_manifest(output / "ai_research_manifest.json")

    def test_generation_failure_cannot_look_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with mock.patch.object(generator, "iter_pattern_runs", side_effect=RuntimeError("worker failed")):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    generator.generate_dataset(output, ai.AIExperimentConfig(structures=3))
            status = json.loads((output / "dataset_status.json").read_text())
            self.assertEqual(status["status"], "FAILED")
            self.assertFalse((output / "synthetic_pattern_dataset.csv").exists())

    def test_empty_universe_does_not_invent_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(generator, "iter_pattern_runs", return_value=iter(())):
                with self.assertRaisesRegex(RuntimeError, "No eligible"):
                    generator.generate_dataset(Path(directory), ai.AIExperimentConfig(structures=3))


if __name__ == "__main__":
    unittest.main()
