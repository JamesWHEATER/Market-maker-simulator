"""Capacity screening uses bounded metadata, never pilot observations or labels."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import atlas_capacity
from atlas_resources import GIB


class StreamingCapacityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {
            "generated_runs": 20, "rows": 100,
            "compressed_bytes": 1000, "catalog_bytes": 2000,
            "generation_seconds": 12.0,
            "rows_by_split": {"train": 60, "validation": 20, "test": 20},
            "experiment": {"steps_per_run": 3000},
        }
        self.dataset = SimpleNamespace(
            manifest=self.manifest,
            iter_batches=Mock(side_effect=AssertionError("capacity opened observations")),
            iter_runs=Mock(side_effect=AssertionError("capacity opened run outcomes")),
        )
        self.memory = dict(total=16 * GIB, available=8 * GIB, rss=GIB // 2,
                           private=GIB // 2, os_peak_rss=GIB // 2, processes=1)
        self.report = {
            "peak_process_tree_rss_bytes": GIB,
            "peak_process_tree_private_bytes": GIB,
            "resource_guard_failure": None,
            "stages": {stage: {"samples": 3, "peak_rss_bytes": GIB,
                                "peak_private_bytes": GIB}
                       for stage in ("generation", "preprocess", "logistic", "xgboost", "mlp", "validation", "test", "explanation")},
        }
        self.status = {"status": "EVALUATION_COMPLETE", "command": "all",
                       "confirmatory_results_valid": True}
        for feature_set, size in (("observable", 200), ("oracle", 300)):
            self.write(f"streaming_training_{feature_set}.json", {
                "xgboost": {"cache_bytes_observed": size},
                "preprocessing": {"feature_count": 50},
            })
        # Size-only fixture: deliberately not a parseable observation export.
        (self.root / "test_predictions_fixture.csv.gz").write_bytes(b"do not read test labels")

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def screen(self, *, free=100 * GIB, write_report=True):
        self.write("atlas_status.json", self.status)
        if write_report:
            self.write("resource_report.json", self.report)
        with patch("atlas_streaming_data.StreamingDataset", return_value=self.dataset) as constructor, \
                patch("atlas_resources.memory_snapshot", return_value=self.memory), \
                patch("atlas_capacity.shutil.disk_usage", return_value=SimpleNamespace(free=free)), \
                patch("atlas_capacity.pd.read_csv", side_effect=AssertionError("capacity read CSV")), \
                patch("builtins.print"):
            result = atlas_capacity.streaming_capacity(self.root, 1000, 2, Path("D:/"))
        constructor.assert_called_once_with(self.root / "data")
        self.dataset.iter_batches.assert_not_called()
        self.dataset.iter_runs.assert_not_called()
        return result

    def test_complete_measured_workflow_can_pass_without_loading_observations(self):
        result = self.screen()
        self.assertTrue(result["screen_passed"])
        self.assertFalse(result["full_scale_certified"])
        self.assertEqual(result["projected_pattern_rows"], 10000)
        self.assertEqual(result["projected_compressed_data_and_catalog_bytes"], 300000)
        self.assertEqual(result["projected_live_xgboost_cache_bytes"], 30000)
        self.assertEqual(result["projected_generation_seconds_linear_pilot_rate"], 1200)

    def test_missing_any_required_stage_prevents_a_complete_workflow_screen(self):
        original = dict(self.report["stages"])
        for missing in ("generation", "xgboost", "mlp", "test"):
            with self.subTest(missing_stage=missing):
                self.report["stages"] = {key: value for key, value in original.items() if key != missing}
                self.assertFalse(self.screen()["screen_passed"])

    def test_train_only_status_is_not_an_end_to_end_pilot(self):
        self.status["command"] = "train"
        self.assertFalse(self.screen()["screen_passed"])

    def test_missing_memory_report_and_incomplete_status_fail_screen(self):
        self.assertFalse(self.screen(write_report=False)["screen_passed"])
        for status in ("FAILED", "RUNNING", "DATASET_COMPLETE", "INSUFFICIENT_VALIDATION_SUPPORT"):
            with self.subTest(status=status):
                self.status["status"] = status
                self.assertFalse(self.screen()["screen_passed"])

    def test_both_feature_sets_must_have_measured_training_metadata(self):
        (self.root / "streaming_training_oracle.json").unlink()
        result = self.screen()
        self.assertFalse(result["both_feature_workflow_measured"])
        self.assertFalse(result["screen_passed"])

    def test_latched_memory_or_disk_guard_failure_prevents_passing(self):
        for failure in ("Process-tree memory exceeded budget", "D: free space fell below reserve"):
            with self.subTest(guard_failure=failure):
                self.report["resource_guard_failure"] = failure
                self.assertFalse(self.screen()["screen_passed"])

    def test_projected_resident_rows_and_low_available_memory_fail_screen(self):
        self.manifest["rows_by_split"]["train"] = 1_000_000_000
        result = self.screen()
        self.assertFalse(result["memory_screen_passed"])
        self.assertFalse(result["screen_passed"])
        self.manifest["rows_by_split"]["train"] = 60
        self.memory["available"] = GIB
        result = self.screen()
        self.assertFalse(result["memory_screen_passed"])
        self.assertFalse(result["screen_passed"])

    def test_current_destination_must_cover_disk_allowance(self):
        result = self.screen(free=GIB)
        self.assertGreater(result["planning_disk_bytes_with_50_percent_headroom_and_5_gib_reserve"], GIB)
        self.assertFalse(result["screen_passed"])


if __name__ == "__main__":
    unittest.main()
