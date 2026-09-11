"""Constant-space generation progress at 1,000-run milestones."""
from __future__ import annotations

import math
import time


def _duration(seconds: float) -> str:
    days, remainder = divmod(int(max(0.0, seconds)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


class GenerationProgress:
    """ETA uses work done in this invocation, not reused checkpoint runs.

    Elapsed time includes saved active generation time when resuming. The final
    line is explicit so 100% is printed only after dataset finalization succeeds.
    """
    def __init__(self, total_runs: int, *, completed_runs: int = 0,
                 elapsed_seconds: float = 0.0, clock=None):
        if isinstance(total_runs, bool) or not isinstance(total_runs, int) or total_runs <= 0:
            raise ValueError("total_runs must be a positive integer")
        if isinstance(completed_runs, bool) or not isinstance(completed_runs, int) or not 0 <= completed_runs <= total_runs:
            raise ValueError("completed_runs must lie between zero and total_runs")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        self.total = total_runs
        self.initial_completed = self.completed = completed_runs
        self.previous_elapsed = elapsed_seconds
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.last_reported = completed_runs
        self.finished = False

    def _report(self) -> None:
        current_elapsed = max(0.0, self.clock() - self.started)
        new_runs = self.completed - self.initial_completed
        remaining = self.total - self.completed
        eta = (_duration(current_elapsed * remaining / new_runs)
               if new_runs and current_elapsed else "estimating")
        if remaining == 0:
            eta = "00:00:00"
        print(f"Atlas generation: {self.completed:,}/{self.total:,} runs "
              f"({100 * self.completed / self.total:.2f}%) | "
              f"elapsed {_duration(self.previous_elapsed + current_elapsed)} | ETA {eta}", flush=True)
        self.last_reported = self.completed

    def update(self, completed_runs: int) -> None:
        if self.finished:
            return
        if isinstance(completed_runs, bool) or not isinstance(completed_runs, int) or not self.completed <= completed_runs <= self.total:
            raise ValueError("completed run count must increase within the planned total")
        self.completed = completed_runs
        if completed_runs < self.total and completed_runs % 1000 == 0 and completed_runs > self.last_reported:
            self._report()

    def finish(self) -> None:
        if not self.finished:
            self.completed = self.total
            self._report()
            self.finished = True
