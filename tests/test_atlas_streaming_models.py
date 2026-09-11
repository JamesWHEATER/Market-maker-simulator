"""Disk-iterator training, train-only preprocessing, and solver calibration checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

import ai_pattern_research_pipeline as ai
import atlas_streaming_models as models


class DiskFixture:
    """A genuinely reread compressed file per split; tracks all requested reads."""
    def __init__(self, root, frames):
        self.root = Path(root)
        self.manifest = {"rows_by_split": {name: len(frame) for name, frame in frames.items()}}
        self.reads = []
        self.max_rows = 0
        for name, frame in frames.items():
            frame.to_csv(self.root / f"{name}.csv.gz", index=False, compression="gzip")

    def iter_batches(self, split=None, batch_rows=4096, columns=None):
        self.reads.append(split)
        if split == "test":
            raise AssertionError("Training touched the test split")
        for frame in pd.read_csv(self.root / f"{split}.csv.gz", chunksize=batch_rows, usecols=columns):
            self.max_rows = max(self.max_rows, len(frame))
            yield frame


class Monitor:
    def __init__(self):
        self.stages = []

    def check(self, stage):
        self.stages.append(stage)


class StreamingPreprocessorTests(unittest.TestCase):
    def test_train_only_mean_scale_missing_and_unknown(self):
        train = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0], "empty": [np.nan] * 4,
                              "category": ["a", "b", None, "a"], "profitable": [0, 1, 0, 1]})
        validation = pd.DataFrame({"x": [1e9], "empty": [100.0],
                                   "category": ["unseen"], "profitable": [1]})
        with tempfile.TemporaryDirectory() as tmp:
            dataset = DiskFixture(tmp, {"train": train, "validation": validation})
            processor = models.StreamingPreprocessor(["x", "empty"], ["category"])
            processor.fit(dataset, batch_rows=2)
            np.testing.assert_allclose(processor.mean_, [3.0, 0.0])
            np.testing.assert_allclose(processor.scale_, [np.sqrt(2.0), 1.0])
            transformed = processor.transform(train)
            self.assertEqual(transformed.dtype, np.float32)
            self.assertEqual(transformed[1, 0], 0.0)
            self.assertEqual(transformed[1, 2], 1.0)
            self.assertTrue(np.isfinite(transformed).all())
            self.assertEqual(processor.transform(validation)[0, 5], 1.0)  # unknown category
            self.assertEqual(dataset.reads, ["train"])
            self.assertEqual(dataset.max_rows, 2)

    def test_category_cap_and_future_leakage_guard(self):
        frame = pd.DataFrame({"category": [f"c{i}" for i in range(200)], "profitable": [0, 1] * 100})
        with tempfile.TemporaryDirectory() as tmp:
            dataset = DiskFixture(tmp, {"train": frame})
            processor = models.StreamingPreprocessor([], ["category"], max_categories=8).fit(dataset, batch_rows=17)
            self.assertEqual(len(processor.categories_["category"]), 8)
            self.assertEqual(processor.transform(frame).shape, (200, 8))
            self.assertEqual(processor.transform(frame.iloc[-1:])[0, 1], 1.0)
        with self.assertRaisesRegex(ValueError, "leakage"):
            models.StreamingPreprocessor(["profitable"], [])

    def test_invalid_labels_fail_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = DiskFixture(tmp, {"train": pd.DataFrame({"x": [1.0], "profitable": [0.5]})})
            with self.assertRaisesRegex(ValueError, "binary"):
                models.StreamingPreprocessor(["x"], []).fit(dataset)

    def test_memory_guard_failure_propagates(self):
        monitor = mock.Mock()
        monitor.check.side_effect = MemoryError("budget")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = DiskFixture(tmp, {"train": pd.DataFrame({"x": [1.0], "profitable": [0]})})
            with self.assertRaisesRegex(MemoryError, "budget"):
                models.StreamingPreprocessor(["x"], []).fit(dataset, monitor=monitor)


class StreamingTrainingTests(unittest.TestCase):
    def test_incremental_solver_has_bounded_probability_and_calibration_differences(self):
        report = models.validate_incremental_logistic_reference()
        self.assertTrue(report["passed"], report)
        self.assertLess(report["incremental"]["log_loss"], 0.55)
        self.assertLess(report["probability_mean_absolute_difference"], 0.05)
        self.assertIn("ece_10_bins", report["incremental_minus_reference"])

    def test_xgboost_old_version_fails_without_in_memory_fallback(self):
        import xgboost
        with mock.patch.object(xgboost, "__version__", "2.1.4"):
            with self.assertRaisesRegex(RuntimeError, "no in-memory fallback"):
                models.require_external_memory_xgboost()

    def test_all_models_train_from_compressed_disk_batches_and_save(self):
        numeric, categorical = ai.feature_columns("observable")
        rng = np.random.default_rng(248)

        def frame(n):
            data = pd.DataFrame({name: rng.normal(size=n) for name in numeric})
            for name in categorical:
                data[name] = np.where(np.arange(n) % 2, "a", "b")
            p = 1.0 / (1.0 + np.exp(-data[numeric[0]].to_numpy()))
            data["profitable"] = (rng.random(n) < p).astype(int)
            return data

        train, validation = frame(192), frame(64)
        config = ai.AIExperimentConfig(mlp_epochs=2, mlp_batch_size=16, mlp_patience=2)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = DiskFixture(tmp, {"train": train, "validation": validation})
            monitor = Monitor()
            wrappers = models.train_streaming_models(dataset, config, Path(tmp), batch_rows=31,
                                                       logistic_epochs=2, xgb_rounds=3, monitor=monitor)
            self.assertEqual(set(wrappers), {"logistic", "xgboost", "mlp"})
            for wrapper in wrappers.values():
                probabilities = wrapper.predict_frame(validation.iloc[:9])
                self.assertEqual(probabilities.shape, (9,))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
            self.assertLessEqual(dataset.max_rows, 31)
            self.assertNotIn("test", dataset.reads)
            self.assertGreater(dataset.reads.count("train"), 3)
            metadata = json.loads((Path(tmp) / "streaming_training_observable.json").read_text())
            self.assertEqual(metadata["xgboost"]["matrix"], "ExtMemQuantileDMatrix")
            self.assertGreater(metadata["xgboost"]["cache_files_observed"], 0)
            self.assertGreater(metadata["xgboost"]["cache_bytes_observed"], 0)
            self.assertEqual(metadata["mlp"]["device"], "cpu")
            self.assertFalse(metadata["test_split_used_for_training"])
            self.assertTrue(any("round 1 after" in stage for stage in monitor.stages))
            saved = joblib.load(Path(tmp) / "streaming_logistic_observable.joblib")
            transformed = wrappers["logistic"].preprocessor.transform(validation)
            np.testing.assert_allclose(saved.predict_proba(transformed)[:, 1],
                                       wrappers["logistic"].predict_frame(validation))
            # Cache handles must be released, including on Windows, before temp cleanup.
            self.assertEqual(list((Path(tmp) / "xgboost_cache").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
