"""Regression and causal integration checks for the shared detector market-data contract."""

import unittest
from dataclasses import asdict, replace

import numpy as np
from threadpoolctl import threadpool_limits

from chart_renderer import build_candles
from market_maker_simulator import SyntheticMarketConfig, simulate_market
from market_structure_experiment import (
    CostModel,
    _filter_overlaps,
    net_trade_return,
    synthetic_execution_arrays,
)
from pattern_detector import (
    DEFAULT_PATTERN_CONFIG,
    DETECTOR_VERSION,
    DetectionResult,
    PatternDetection,
    _cluster_candidates_causally,
    detect_pattern_universe,
)


def _candidate(name, available_at, *, confirmed=False):
    return PatternDetection(
        pattern_name=name,
        start_index=10,
        end_index=30,
        expected_direction="bearish",
        geometry_fit_score=0.8,
        available_at_index=available_at,
        metadata={
            "pivot_geometry_indices_global": (10, 20, 30),
            "pivot_kinds": ("max", "min", "max"),
            "breakout_confirmed_by_availability": confirmed,
            "breakout_direction_by_availability": "bearish" if confirmed else None,
        },
    )


def _cluster_result(candidates):
    assigned, events, clusters = _cluster_candidates_causally(candidates, DEFAULT_PATTERN_CONFIG)
    return DetectionResult(
        candidates=assigned,
        events=events,
        clusters=clusters,
        detector_version=DETECTOR_VERSION,
        config_hash="test",
        close_series_fingerprint="test",
        full_market_data_fingerprint="test",
        geometry_fingerprint="test",
        config_snapshot=DEFAULT_PATTERN_CONFIG,
    )


class CompetitionContractTests(unittest.TestCase):
    def test_promoted_member_observes_competition_at_its_own_availability(self):
        initial = _candidate("double_top", 31)
        competitor = _candidate("head_and_shoulders", 33)
        promoted = _candidate("double_top", 35, confirmed=True)
        future = _candidate("double_bottom", 40)
        prefix = _cluster_result([initial, competitor, promoted])
        full = _cluster_result([initial, competitor, promoted, future])

        primary = next(e for e in full.events if e.pattern_name == "double_top")
        event = full.confirmed_breakout_events[0]
        self.assertFalse(primary.metadata["competing_pattern"])
        self.assertTrue(event.metadata["competing_pattern"])
        self.assertEqual(event.metadata["known_competing_patterns"], ("head_and_shoulders",))
        self.assertEqual(event.available_at_index, 35)
        self.assertEqual(event.earliest_execution_index, 36)
        self.assertEqual(event.event_id, primary.event_id)
        self.assertEqual(prefix.confirmed_breakout_events, full.confirmed_breakout_events)
        self.assertEqual(primary, prefix.events[0])

    def test_promoted_member_without_competition_has_explicit_false(self):
        result = _cluster_result([
            _candidate("double_top", 31),
            _candidate("double_top", 35, confirmed=True),
        ])
        event = result.confirmed_breakout_events[0]
        self.assertIs(event.metadata["competing_pattern"], False)
        self.assertEqual(event.metadata["known_competing_patterns"], ())
        self.assertEqual(event.metadata["known_competing_event_ids"], ())

    def test_same_time_competition_is_independent_of_candidate_input_order(self):
        candidates = [
            _candidate("double_top", 31),
            _candidate("double_top", 35, confirmed=True),
            _candidate("head_and_shoulders", 35),
        ]
        forward = _cluster_result(candidates)
        reverse = _cluster_result(list(reversed(candidates)))
        self.assertEqual(forward, reverse)
        self.assertTrue(forward.confirmed_breakout_events[0].metadata["competing_pattern"])

    def test_detector_history_and_confirmed_signals_are_prefix_causal(self):
        rng = np.random.default_rng(123)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, 240)))
        with threadpool_limits(limits=1):
            full = detect_pattern_universe(prices)
            prefix = detect_pattern_universe(prices[:170])
        self.assertGreater(len(prefix.candidates), 0)
        self.assertEqual(prefix.candidates, tuple(e for e in full.candidates if e.available_at_index < 170))
        self.assertEqual(prefix.events, tuple(e for e in full.events if e.available_at_index < 170))
        self.assertEqual(
            prefix.confirmed_breakout_events,
            tuple(e for e in full.confirmed_breakout_events if e.available_at_index < 170),
        )
        for event in full.candidates:
            self.assertIsInstance(event.metadata["competing_pattern"], bool)
            self.assertGreater(event.earliest_execution_index, event.available_at_index)
            self.assertLessEqual(event.end_index, event.available_at_index)


class SyntheticCandleExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SyntheticMarketConfig(
            steps=153,
            seed=123,
            world_weights={"momentum": 1.0, "liquidity": 1.0},
        )
        cls.history = simulate_market(cls.config)

    def test_candles_preserve_all_intrastep_prices_volume_and_fixed_duration(self):
        candles = build_candles(self.history, 5)
        self.assertEqual(len(candles), 30)
        for index, candle in enumerate(candles):
            chunk = self.history[index * 5 : (index + 1) * 5]
            self.assertEqual(candle.open, chunk[0].open)
            self.assertEqual(candle.close, chunk[-1].close)
            self.assertEqual(candle.high, max(s.high for s in chunk))
            self.assertEqual(candle.low, min(s.low for s in chunk))
            self.assertEqual(candle.volume, sum(s.buy_volume + s.sell_volume for s in chunk))
            self.assertEqual(candle.end_step - candle.start_step + 1, 5)
            self.assertTrue(candle.is_complete)
        partial = build_candles(self.history, 5, include_partial=True)
        self.assertEqual(partial[:-1], candles)
        self.assertFalse(partial[-1].is_complete)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_candles(self.history[:10] + self.history[11:], 5)

    def test_quote_execution_reconstructs_actual_bid_and_ask_for_both_sides(self):
        candles = build_candles(self.history, 5)
        entry_mids, exit_mids, entry_spreads, exit_spreads = synthetic_execution_arrays(
            self.history, candles
        )
        costs = CostModel(brokerage_bps_per_side=0.0, slippage_bps_per_side=0.0)
        for entry, horizon in ((0, 1), (4, 3), (8, 20)):
            exit_index = entry + horizon - 1
            start = self.history[candles[entry].start_step - 1]
            end = self.history[candles[exit_index].end_step - 1]
            for direction in (-1, 1):
                expected = (
                    (end.post_trade_bid - start.ask) if direction == 1
                    else (start.bid - end.post_trade_ask)
                ) / entry_mids[entry]
                returned = net_trade_return(
                    entry_mids[entry], exit_mids[exit_index], direction,
                    entry_spreads[entry], exit_spreads[exit_index], costs,
                )
                self.assertAlmostEqual(returned[-1], expected, places=14)

    def test_simulator_prefix_and_diagnostics_do_not_change_economic_path(self):
        short = simulate_market(replace(self.config, steps=75), record_step_diagnostics=False)
        for left, right in zip(self.history, short):
            left_data, right_data = asdict(left), asdict(right)
            for metadata in ("scenario_id", "run_id", "world_signals_json", "world_flow_json"):
                left_data.pop(metadata)
                right_data.pop(metadata)
            self.assertEqual(left_data, right_data)
        self.assertEqual(build_candles(self.history, 5)[:15], build_candles(short, 5))


class SimultaneousSignalOrderingTests(unittest.TestCase):
    @staticmethod
    def _ai_rows(events):
        import pandas as pd

        return pd.DataFrame([
            {
                "event_id": event.event_id,
                "pattern_type": event.pattern_name,
                "trade_direction": "short",
                "probability": 0.8,
                "available_at_index": event.available_at_index,
                "pattern_start_index": event.start_index,
                "pattern_end_index": event.end_index,
                "geometry_fit_score": event.geometry_fit_score,
                **event.metadata,
            }
            for event in events
        ])

    def test_provenance_cannot_change_representative_detector_scale(self):
        from ai_pattern_research_pipeline import _resolve_same_entry_group

        short_scale = replace(
            _candidate("double_top", 110, confirmed=True),
            event_id="z-original",
            metadata={
                **_candidate("double_top", 110, confirmed=True).metadata,
                "observation_window_length": 43,
                "core_window_length": 35,
            },
        )
        long_scale = replace(
            short_scale,
            event_id="a-original",
            metadata={**short_scale.metadata, "observation_window_length": 96, "core_window_length": 80},
        )
        original = [long_scale, short_scale]
        relabeled = [replace(short_scale, event_id="a-renamed"), replace(long_scale, event_id="z-renamed")]
        for events in (original, relabeled):
            statistical, _, _ = _filter_overlaps(events, 20, "skip_new", 200)
            ai = _resolve_same_entry_group(self._ai_rows(events))
            self.assertEqual(len(statistical), 1)
            self.assertEqual(statistical[0].metadata["observation_window_length"], 43)
            self.assertEqual(int(ai.iloc[0]["observation_window_length"]), 43)

    def test_equal_probability_ai_baseline_uses_statistical_pattern_priority(self):
        from ai_pattern_research_pipeline import _resolve_same_entry_group

        events = [
            replace(_candidate("broadening_formation", 35, confirmed=True), event_id="a"),
            replace(_candidate("head_and_shoulders", 35, confirmed=True), event_id="z"),
        ]
        statistical, _, _ = _filter_overlaps(events, 20, "skip_new", 100)
        ai = _resolve_same_entry_group(self._ai_rows(events))
        self.assertEqual(statistical[0].pattern_name, "head_and_shoulders")
        self.assertEqual(ai.iloc[0]["pattern_type"], statistical[0].pattern_name)
        conflicting = self._ai_rows(events)
        conflicting.loc[0, "trade_direction"] = "long"
        self.assertTrue(_resolve_same_entry_group(conflicting).empty)


if __name__ == "__main__":
    unittest.main()
