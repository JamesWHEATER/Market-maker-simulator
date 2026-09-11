"""Progress cadence and resume timing without running expensive simulations."""

import unittest
from unittest import mock

from atlas_progress import GenerationProgress


class GenerationProgressTests(unittest.TestCase):
    def test_only_new_thousand_run_milestones_print(self):
        clock = mock.Mock(return_value=50.0)
        with mock.patch("builtins.print") as output:
            progress = GenerationProgress(2_000_000, clock=clock)
            progress.update(0)
            clock.return_value = 650.0
            progress.update(999)
            output.assert_not_called()

            progress.update(1000)
            output.assert_called_once()
            self.assertEqual(
                output.call_args.args[0],
                "Atlas generation: 1,000/2,000,000 runs (0.05%) | "
                "elapsed 00:10:00 | ETA 13d 21:10:00",
            )
            progress.update(1000)
            progress.update(1001)
            self.assertEqual(output.call_count, 1)

            clock.return_value = 1250.0
            progress.update(2000)
            self.assertEqual(output.call_count, 2)
            self.assertEqual(
                output.call_args.args[0],
                "Atlas generation: 2,000/2,000,000 runs (0.10%) | "
                "elapsed 00:20:00 | ETA 13d 21:00:00",
            )

    def test_finish_reports_total_once_for_partial_and_exact_milestones(self):
        for total in (18, 2000, 2500):
            with self.subTest(total=total):
                clock = mock.Mock(return_value=10.0)
                with mock.patch("builtins.print") as output:
                    progress = GenerationProgress(total, clock=clock)
                    clock.return_value = 100.0
                    progress.update(total)
                    output.assert_not_called()
                    progress.finish()
                    progress.finish()
                    output.assert_called_once()
                    self.assertEqual(
                        output.call_args.args[0],
                        f"Atlas generation: {total:,}/{total:,} runs (100.00%) | "
                        "elapsed 00:01:30 | ETA 00:00:00",
                    )

    def test_resume_uses_global_milestone_and_only_new_runs_for_eta(self):
        clock = mock.Mock(return_value=100.0)
        with mock.patch("builtins.print") as output:
            progress = GenerationProgress(
                10_000, completed_runs=1500, elapsed_seconds=3600.0, clock=clock,
            )
            progress.update(1500)
            clock.return_value = 400.0
            progress.update(1999)
            output.assert_not_called()
            progress.update(2000)
            output.assert_called_once()
            # Only the 500 runs completed in these 300 seconds determine ETA.
            self.assertEqual(
                output.call_args.args[0],
                "Atlas generation: 2,000/10,000 runs (20.00%) | "
                "elapsed 01:05:00 | ETA 01:20:00",
            )
            progress.update(2000)
            output.assert_called_once()

    def test_completed_resume_still_reports_finish_once(self):
        clock = mock.Mock(return_value=100.0)
        with mock.patch("builtins.print") as output:
            progress = GenerationProgress(
                2000, completed_runs=2000,
                elapsed_seconds=2 * 86400 + 3661, clock=clock,
            )
            progress.update(2000)
            output.assert_not_called()
            progress.finish()
            progress.finish()
            output.assert_called_once()
            self.assertEqual(
                output.call_args.args[0],
                "Atlas generation: 2,000/2,000 runs (100.00%) | "
                "elapsed 2d 01:01:01 | ETA 00:00:00",
            )

    def test_elapsed_and_eta_display_days_without_wrapping(self):
        clock = mock.Mock(return_value=40.0)
        with mock.patch("builtins.print") as output:
            progress = GenerationProgress(3000, clock=clock)
            clock.return_value = 40.0 + 86400 + 3661
            progress.update(1000)
            output.assert_called_once()
            self.assertEqual(
                output.call_args.args[0],
                "Atlas generation: 1,000/3,000 runs (33.33%) | "
                "elapsed 1d 01:01:01 | ETA 2d 02:02:02",
            )


if __name__ == "__main__":
    unittest.main()
