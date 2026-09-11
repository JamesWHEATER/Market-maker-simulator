"""AI research integration and inference regressions; no model training required."""

import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import ai_pattern_research_pipeline as ai
from market_structure_experiment import CostModel, ProcedureConfig, _procedure_payload, procedure_hash
from market_maker_simulator import SyntheticMarketConfig


def trade_rows(returns=(0.1, -0.1), *, same_run=False):
    return pd.DataFrame([
        {"run_id": "r0" if same_run else f"r{i}", "event_id": f"e{i}",
         "entry_index": i * 5, "exit_index": i * 5 + 1,
         "pattern_type": "double_bottom", "trade_direction": "bullish",
         "net_directional_return": value, "profitable": int(value > 0)}
        for i, value in enumerate(returns)
    ])


class AIValidationTests(unittest.TestCase):
    def test_insufficient_validation_support_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_VALIDATION_SUPPORT"):
            ai.select_threshold_on_validation(trade_rows(), np.ones(2), ai.AIExperimentConfig())

    def test_threshold_grid_never_exceeds_frozen_maximum(self):
        config = ai.AIExperimentConfig(min_validation_trades=1, min_validation_runs=1,
                                      validation_threshold_min=0.5, validation_threshold_max=0.9,
                                      validation_threshold_step=0.15)
        visited = []
        selector = ai._select_non_overlapping_trades

        def record(rows, probabilities, threshold):
            visited.append(threshold)
            return selector(rows, probabilities, threshold)

        with mock.patch.object(ai, "_select_non_overlapping_trades", side_effect=record):
            ai.select_threshold_on_validation(trade_rows(), np.ones(2), config)
        self.assertEqual(len(visited), 3)
        self.assertTrue(all(0.5 <= value <= 0.9 for value in visited))

    def test_duplicate_dataframe_indices_do_not_duplicate_fills(self):
        rows = trade_rows(same_run=True)
        rows.index = [7, 7]
        selected = ai._select_non_overlapping_trades(rows, np.ones(2), 0.5)
        self.assertEqual(selected["event_id"].tolist(), ["e0", "e1"])
        candidates, fills = ai._selection_masks(rows, np.ones(2), 0.5)
        np.testing.assert_array_equal(candidates, [1, 1])
        np.testing.assert_array_equal(fills, [1, 1])

    def test_invalid_predictions_cannot_silently_skip_trades(self):
        for values in (np.array([np.nan, 0.5]), np.array([1.1, 0.2]), np.array([[0.5], [0.5]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ai._select_non_overlapping_trades(trade_rows(), values, 0.5)

    def test_missing_run_from_denominator_is_rejected(self):
        for evaluator in (ai.economic_metrics, ai.paired_filter_uplift_metrics):
            with self.subTest(evaluator=evaluator.__name__), self.assertRaisesRegex(ValueError, "omits"):
                evaluator(trade_rows(), np.ones(2), 0.5, run_ids=["r0"])

    def test_additive_return_points_are_explicit_and_include_zero_runs(self):
        rows = trade_rows((0.1, 0.1), same_run=True)
        metrics = ai.economic_metrics(rows, np.ones(2), 0.5, run_ids=["r0", "zero"])
        self.assertAlmostEqual(metrics["mean_run_net_return_points"], 0.1)
        self.assertNotIn("mean_run_total_net_return", metrics)

    def test_repeated_seeds_do_not_inflate_inference_count(self):
        rows = trade_rows((0.4, 0.4, -0.2, -0.2))
        rows["structure_id"] = ["a", "a", "b", "b"]
        values = ai.economic_metrics(rows, np.ones(4), 0.5)
        mean, low, high = ai._mean_ci([0.4, -0.2])
        self.assertEqual(values["inference_structures"], 2.0)
        self.assertAlmostEqual(values["mean_run_net_return_points"], mean)
        self.assertAlmostEqual(values["mean_run_net_return_points_ci_low"], low)
        self.assertAlmostEqual(values["mean_run_net_return_points_ci_high"], high)

    def test_zero_pattern_structure_remains_in_inference(self):
        rows = trade_rows((0.2, 0.2))
        rows["structure_id"] = ["a", "a"]
        result = ai.economic_metrics(
            rows, np.ones(2), 0.5, run_ids=["r0", "r1", "r2"],
            run_structures={"r0": "a", "r1": "a", "r2": "b"},
        )
        self.assertEqual(result["inference_structures"], 2.0)
        self.assertAlmostEqual(result["mean_run_net_return_points"], 0.1)

    def test_split_rejects_misassigned_run_structure(self):
        rows = pd.DataFrame({"run_id": ["r0", "r1", "r2"], "structure_id": ["a", "b", "c"],
                             "world_family": ["structured"] * 3, "profitable": [1, 0, 1]})
        universe = rows[["run_id", "structure_id", "world_family"]].copy()
        universe.loc[0, "structure_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            ai.split_by_structure(rows, ai.AIExperimentConfig(structures=3), universe)

    def test_future_outcomes_are_excluded_from_feature_sets(self):
        for feature_set in ("observable", "oracle"):
            numeric, categorical = ai.feature_columns(feature_set)
            self.assertFalse(set(numeric + categorical) & ai.FUTURE_OR_OUTCOME_COLUMNS)

    def test_rng_seed_collisions_across_structures_are_resampled(self):
        class CollidingRNG:
            def __init__(self):
                self.next_seed = 100

            def integers(self, *args):
                self.next_seed += 1
                return self.next_seed

        def template(structure_index, *args):
            return replace(SyntheticMarketConfig(world_weights={"momentum": 1.0}), base_spread_fraction=0.001 + 0.0001 * structure_index), "structured"

        with mock.patch.object(ai.np.random, "default_rng", side_effect=lambda *args: CollidingRNG()), \
                mock.patch.object(ai, "_sample_structure_template", side_effect=template):
            specs = ai.sample_run_specs(ai.AIExperimentConfig(structures=3, null_world_fraction=0.0))
        self.assertEqual([spec.config.seed for spec in specs], [101, 102, 103])

    def test_failed_rerun_invalidates_previous_confirmatory_status(self):
        with tempfile.TemporaryDirectory() as root:
            args = ai.build_argument_parser().parse_args(["train", "--legacy-in-memory", "--output-dir", root])
            with mock.patch.object(ai, "_run_cli_impl", side_effect=ValueError("INSUFFICIENT_VALIDATION_SUPPORT")):
                with self.assertRaises(ValueError):
                    ai.run_cli(args)
            status = json.loads((Path(root) / "ai_run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "INSUFFICIENT_VALIDATION_SUPPORT")
        self.assertFalse(status["confirmatory_results_valid"])


class AIProcedureTests(unittest.TestCase):
    def _write_reference(self, root, procedure):
        path = Path(root) / "procedure_manifest.json"
        path.write_text(json.dumps({**_procedure_payload(procedure), "procedure_hash": procedure_hash(procedure)}),
                        encoding="utf-8")
        return path

    def test_ai_universe_is_explicitly_separate(self):
        contract = ai.research_procedure_contract(ai.AIExperimentConfig(), CostModel())
        self.assertFalse(contract["headline_comparable_to_statistical_scenario_grid"])
        self.assertEqual(contract["validation_support_policy"], "fail_closed")
        self.assertIn("additive", contract["return_units"])

    def test_matching_reference_records_procedure_hash(self):
        experiment = ai.AIExperimentConfig()
        procedure = ProcedureConfig(steps_per_candle=experiment.candle_steps)
        with tempfile.TemporaryDirectory() as root:
            reference = ai.validate_reference_procedure_manifest(
                self._write_reference(root, procedure), experiment, CostModel())
        self.assertEqual(reference["procedure_hash"], procedure_hash(procedure))
        self.assertFalse(reference["headline_comparable"])

    def test_reference_rejects_every_execution_design_mismatch(self):
        experiment = ai.AIExperimentConfig()
        matching = ProcedureConfig(steps_per_candle=experiment.candle_steps)
        variants = [replace(matching, steps_per_candle=10), replace(matching, horizons=(1,)),
                    replace(matching, overlap_policy="allow"),
                    replace(matching, cost_model=CostModel(brokerage_bps_per_side=3.0))]
        with tempfile.TemporaryDirectory() as root:
            for procedure in variants:
                with self.subTest(procedure=procedure), self.assertRaisesRegex(ValueError, "settings differ"):
                    ai.validate_reference_procedure_manifest(
                        self._write_reference(root, procedure), experiment, CostModel())

    def test_tampered_reference_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_reference(root, ProcedureConfig())
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["procedure_hash"] = "forged"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "procedure_hash"):
                ai.validate_reference_procedure_manifest(path, ai.AIExperimentConfig(), CostModel())


if __name__ == "__main__":
    unittest.main()
