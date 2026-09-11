import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_resources import GIB, ResourceMonitor, memory_snapshot, require_d_destination


class ResourceTests(unittest.TestCase):
    def test_native_snapshot_covers_current_process(self):
        snapshot = memory_snapshot()
        self.assertGreater(snapshot["rss"], 0)
        self.assertGreater(snapshot["private"], 0)
        self.assertGreater(snapshot["available"], 0)
        self.assertGreaterEqual(snapshot["total"], snapshot["available"])
        self.assertGreaterEqual(snapshot["processes"], 1)

    def test_memory_breach_is_latched_and_reported_even_after_release(self):
        base = dict(total=16*GIB, available=8*GIB, rss=GIB, private=GIB, os_peak_rss=GIB, processes=1)
        with tempfile.TemporaryDirectory() as root, patch("atlas_resources.memory_snapshot", return_value=base) as snapshot:
            monitor = ResourceMonitor(Path(root), memory_budget_gib=2, min_free_disk_gib=0.001)
            monitor._sample()
            snapshot.return_value = {**base, "rss": 3*GIB, "processes": 2}
            monitor._sample()
            snapshot.return_value = base
            monitor._sample()
            with self.assertRaisesRegex(MemoryError, "exceeded"):
                monitor.check("train")
            self.assertEqual(monitor.report()["peak_process_tree_rss_bytes"], 3*GIB)
            self.assertEqual(monitor.report()["max_processes"], 2)

    def test_budget_is_based_on_available_not_installed_ram(self):
        base = dict(total=16*GIB, available=2*GIB, rss=GIB, private=GIB, os_peak_rss=GIB, processes=1)
        with tempfile.TemporaryDirectory() as root, patch("atlas_resources.memory_snapshot", return_value=base):
            monitor = ResourceMonitor(Path(root))
            self.assertAlmostEqual(monitor.budget_bytes/GIB, 2.3)
            with self.assertRaisesRegex(ValueError, "conservative"):
                ResourceMonitor(Path(root), memory_budget_gib=6)

    def test_rejects_other_disk_before_generation(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "D:"):
                require_d_destination(Path(root))

    def test_stage_cardinality_is_bounded(self):
        self.assertEqual(ResourceMonitor._stage_group("generation_shard_923422"), "generation")

    def test_generation_timer_saves_memory_report_without_console_noise(self):
        base = dict(total=16*GIB, available=8*GIB, rss=GIB, private=GIB, os_peak_rss=GIB, processes=1)
        with tempfile.TemporaryDirectory() as root, patch("atlas_resources.memory_snapshot", return_value=base), \
                patch("atlas_resources.time.monotonic", return_value=0.0) as clock, patch("builtins.print") as output:
            monitor = ResourceMonitor(Path(root), memory_budget_gib=2, min_free_disk_gib=0.001)
            clock.return_value = 31.0
            monitor.check("generation run complete")
            output.assert_not_called()
            self.assertTrue((Path(root) / "resource_report.json").exists())
            self.assertEqual(monitor.report()["peak_process_tree_rss_bytes"], GIB)
            clock.return_value = 62.0
            monitor.check("xgboost round 1 after")
            output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
