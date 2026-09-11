"""Streaming regressions: economic parity, bounded inference and frozen test choices."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ai_pattern_research_pipeline as ai
from atlas_streaming_evaluation import (
    BoundedExplanationSample,
    ClassificationAccumulator,
    EconomicAccumulator,
    OnlineMoments,
    PairedAccumulator,
    RunTradeSelector,
    StructureMoments,
    evaluate_streaming_models,
)


def fixture_runs():
    columns = ["run_id", "structure_id", "event_id", "entry_index", "exit_index", "pattern_type",
               "trade_direction", "profitable", "net_directional_return", "probability_fixture", "world_family"]
    # The first long position spans prediction batches. Contradictory tied signals
    # at index 11 are skipped by the classical baseline but resolved by this model.
    candidates = [
        [(0, 10, .30, .95, "bullish"), (5, 6, .70, .80, "bullish"),
         (11, 12, -.20, .90, "bullish"), (11, 12, .15, .80, "bearish"),
         (13, 14, .10, .60, "bullish"), (16, 20, -.40, .40, "bullish")],
        [],
        [(1, 2, -.10, .55, "bullish"), (3, 4, .20, .85, "bullish")],
        [(1, 3, .40, .65, "bullish")],
        [], [],
    ]
    result = []
    for i, events in enumerate(candidates):
        summary = {"run_id": f"r{i}", "structure_id": f"s{i // 2}", "structure_index": i // 2,
                   "world_family": "structured" if i < 4 else "null_iid"}
        rows = pd.DataFrame([
            {"run_id": summary["run_id"], "structure_id": summary["structure_id"], "event_id": f"e{i}_{j}",
             "entry_index": entry, "exit_index": exit_, "pattern_type": "double_bottom",
             "trade_direction": direction, "profitable": int(ret > 0), "net_directional_return": ret,
             "probability_fixture": probability, "world_family": summary["world_family"]}
            for j, (entry, exit_, ret, probability, direction) in enumerate(events)
        ], columns=columns)
        result.append((rows, summary))
    return result


class FixturePredictor:
    name = "Fixture model"

    def __init__(self):
        self.maximum_batch_rows = 0

    def predict_frame(self, rows):
        self.maximum_batch_rows = max(self.maximum_batch_rows, len(rows))
        return rows["probability_fixture"].to_numpy(dtype=float)


class FixtureDataset:
    def __init__(self, output_dir=None, runs=None):
        self.runs = runs if runs is not None else fixture_runs()
        self.output_dir = output_dir
        self.visited = []

    def iter_runs(self, split):
        self.visited.append(split)
        if split == "test" and self.output_dir is not None:
            # Test data is not even opened until validation choices are persisted.
            primary = self.output_dir / "primary_model_observable.json"
            assert primary.exists(), "test opened before validation selection was frozen"
            assert json.loads(primary.read_text())["validation_support_satisfied"]
        yield from self.runs


class StreamingEvaluationTests(unittest.TestCase):
    def test_cached_selector_matches_classical_intervals_and_near_ties(self):
        rng = np.random.default_rng(937)
        for case in range(20):
            count = 35
            entry = rng.integers(0, 30, size=count)
            rows = pd.DataFrame({"run_id": ["r"]*count, "entry_index": entry,
                "exit_index": entry+rng.integers(1, 5, size=count),
                "pattern_type": ["double_bottom"]*count,
                "trade_direction": rng.choice(["bullish", "bearish"], size=count),
                "geometry_fit_score": rng.random(count), "event_id": [f"e{i}" for i in range(count)]})
            probabilities = rng.choice([0.5, 0.5-5e-13, 0.5+5e-13, 0.8, 0.2], size=count)
            selector = RunTradeSelector(rows, probabilities)
            for threshold in (0.0, 0.3, 0.5, 0.5+2e-13, 0.8, 1.0):
                with self.subTest(case=case, threshold=threshold):
                    expected = ai._select_non_overlapping_trades(rows, probabilities, threshold)
                    self.assertEqual(selector.select(threshold).event_id.tolist(), expected.event_id.tolist())

    def test_online_moments_match_structure_t_inference(self):
        values = [.1, -.3, .4, 0., .2]
        accumulator = OnlineMoments()
        for value in values:
            accumulator.add(value)
        np.testing.assert_allclose(accumulator.mean_ci(), ai._mean_ci(values))
        self.assertAlmostEqual(accumulator.positive_pvalue(), ai._one_sided_positive_t_pvalue(values))
        singleton = OnlineMoments()
        singleton.add(0)
        self.assertTrue(math.isnan(singleton.mean_ci()[1]))
        self.assertEqual(singleton.positive_pvalue(), 1.0)

    def test_economic_and_paired_semantics_match_in_memory_including_zero_seeds(self):
        runs = fixture_runs()
        frame = pd.concat([rows for rows, _summary in runs], ignore_index=True)
        run_ids = [summary["run_id"] for _rows, summary in runs]
        mapping = {summary["run_id"]: summary["structure_id"] for _rows, summary in runs}
        threshold = .6
        accumulator, baseline, paired = EconomicAccumulator(), EconomicAccumulator(), PairedAccumulator()
        for rows, summary in runs:
            probabilities = rows["probability_fixture"].to_numpy(dtype=float)
            total = accumulator.add_run(rows, probabilities, threshold, summary)
            baseline_total = baseline.add_run(rows, np.ones(len(rows)), .5, summary)
            paired.add_run(summary, total, baseline_total)
        expected = ai.economic_metrics(frame, frame["probability_fixture"].to_numpy(), threshold,
                                       run_ids=run_ids, run_structures=mapping)
        actual = accumulator.metrics()
        for key, value in expected.items():
            if key.startswith("median_"):
                continue
            with self.subTest(key=key):
                self.assertAlmostEqual(actual[key], value)
        self.assertEqual(actual["inference_structures"], 3.0)
        self.assertEqual(actual["evaluation_runs"], 6.0)
        expected_paired = ai.paired_filter_uplift_metrics(frame, frame["probability_fixture"].to_numpy(), threshold,
                                                         run_ids=run_ids, run_structures=mapping)
        for key, value in expected_paired.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(paired.metrics()[key], value)

    def test_classification_batches_match_exact_decomposable_metrics(self):
        labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        probabilities = np.array([0., .2, .3, .3, .9, .8, 1., .4])
        accumulator = ClassificationAccumulator([.5])
        for start in range(0, len(labels), 3):
            accumulator.add(labels[start:start + 3], probabilities[start:start + 3])
        expected = ai.classification_metrics(labels, probabilities, .5)
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(accumulator.metrics()[key], value)
        self.assertEqual(accumulator.metrics()["ranking_metric_method"], "approximate_fixed_probability_histogram")
        self.assertEqual(accumulator.metrics()["ranking_histogram_bins"], 4096)
        self.assertEqual(int(accumulator.calibration_table()["count"].sum()), len(labels))

    def test_histogram_rank_ties_are_explicitly_approximate(self):
        accumulator = ClassificationAccumulator([.5], bins=10)
        accumulator.add([0, 1], [.501, .502])
        self.assertEqual(accumulator.metrics()["roc_auc"], .5)
        self.assertEqual(accumulator.metrics()["average_precision"], .5)
        self.assertEqual(len(accumulator.positive), 10)

    def test_complete_evaluation_freezes_before_test_and_predicts_in_bounded_batches(self):
        config = ai.AIExperimentConfig(min_validation_trades=1, min_validation_runs=1,
                                       validation_threshold_min=.5, validation_threshold_max=.9,
                                       validation_threshold_step=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dataset, model = FixtureDataset(path), FixturePredictor()
            comparison = evaluate_streaming_models(dataset, {"fixture": model}, config, path,
                                                   feature_set="observable", batch_rows=2, explanation_rows=3)
            self.assertEqual(dataset.visited, ["validation", "test"])
            self.assertLessEqual(model.maximum_batch_rows, 2)
            runs = fixture_runs()
            all_rows = pd.concat([rows for rows, _summary in runs], ignore_index=True)
            mapping = {summary["run_id"]: summary["structure_id"] for _rows, summary in runs}
            expected_threshold = ai.select_threshold_on_validation(all_rows, all_rows["probability_fixture"].to_numpy(),
                                                                  config, run_ids=list(mapping), run_structures=mapping)
            self.assertAlmostEqual(comparison.iloc[0]["validation_selected_threshold"], expected_threshold)
            predictions = pd.read_csv(path / "test_predictions_fixture_observable.csv.gz")
            candidates, executed = ai._selection_masks(all_rows, all_rows["probability_fixture"].to_numpy(), expected_threshold)
            np.testing.assert_array_equal(predictions["candidate_selected"], candidates)
            np.testing.assert_array_equal(predictions["selected"], executed)
            self.assertEqual(len(predictions), len(all_rows))
            totals = pd.read_csv(path / "test_run_returns_observable.csv.gz")
            self.assertEqual(len(totals), 6)
            self.assertEqual(totals.loc[totals.run_id == "r1", "fixture_return_points"].iloc[0], 0)
            self.assertEqual(comparison.iloc[0]["paired_inference_structures"], 3)
            explanation = pd.read_csv(path / "explanation_sample_observable.csv.gz")
            self.assertEqual(len(explanation), 3)
            metadata = json.loads((path / "explanation_metadata_observable.json").read_text())
            self.assertEqual(metadata["test_event_population"], len(all_rows))

    def test_validation_without_support_never_opens_test(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dataset = FixtureDataset(path)
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_VALIDATION_SUPPORT"):
                evaluate_streaming_models(dataset, {"fixture": FixturePredictor()}, ai.AIExperimentConfig(),
                                          path, feature_set="observable", batch_rows=2)
            self.assertEqual(dataset.visited, ["validation"])
            self.assertFalse((path / "primary_model_observable.json").exists())
            self.assertFalse((path / "test_predictions_fixture_observable.csv.gz").exists())

    def test_all_zero_event_test_runs_still_contribute_inference_units(self):
        runs = fixture_runs()

        class EmptyTestDataset(FixtureDataset):
            def iter_runs(self, split):
                self.visited.append(split)
                for rows, summary in self.runs:
                    yield (rows.iloc[:0] if split == "test" else rows), summary

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            comparison = evaluate_streaming_models(EmptyTestDataset(), {"fixture": FixturePredictor()},
                                                   ai.AIExperimentConfig(min_validation_trades=1, min_validation_runs=1),
                                                   path, feature_set="observable", explanation_rows=0)
            row = comparison.iloc[0]
            self.assertEqual(row["paired_inference_structures"], 3)
            self.assertEqual(row["paired_evaluation_runs"], 6)
            self.assertEqual(row["paired_ai_equals_baseline_run_fraction"], 1)
            self.assertEqual(row["mean_run_net_return_points"], 0)
            self.assertTrue(pd.read_csv(path / "test_predictions_fixture_observable.csv.gz").empty)

    def test_noncontiguous_structure_indices_are_rejected(self):
        moments = StructureMoments()
        moments.add({"structure_id": "a", "structure_index": 0}, 1.)
        moments.add({"structure_id": "b", "structure_index": 1}, 2.)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            moments.add({"structure_id": "a", "structure_index": 0}, 3.)

    def test_uniform_sample_storage_never_exceeds_limit(self):
        sample = BoundedExplanationSample(7, 42)
        for start in range(0, 1000, 10):
            sample.add(pd.DataFrame({"id": np.arange(start, start + 10)}))
            self.assertLessEqual(len(sample.frame), 7)
            self.assertLessEqual(len(sample.priorities), 7)
        self.assertEqual(sample.population, 1000)
        self.assertEqual(sample.frame.id.nunique(), 7)


if __name__ == "__main__":
    unittest.main()
