from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from chart_renderer import PriceCandle


@dataclass(frozen=True)
class PatternDetection:
    pattern_name: str
    start_index: int
    end_index: int
    signal: str
    confidence: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LocalExtremum:
    index: int
    value: float
    kind: str
    available_at: int


@dataclass(frozen=True)
class PatternConfig:
    """Configuration consistent with the Lo, Mamaysky and Wang (2000) paper.

    The paper uses rolling windows of length l + d with l = 35 and d = 3, final
    detection delayed by d trading days, and a practical bandwidth of 0.3 * h*.
    """

    window_length: int = 35
    detection_lag: int = 3
    bandwidth_scale: float = 0.30
    head_shoulders_tolerance: float = 0.015
    rectangle_tolerance: float = 0.0075
    double_tolerance: float = 0.015
    double_min_spacing_days: int = 22
    double_min_trough_depth: float = 0.01
    double_require_neckline_confirmation: bool = False
    min_extrema_count: int = 5
    min_pattern_span: int = 3

    @property
    def effective_window_length(self) -> int:
        return self.window_length + self.detection_lag


DEFAULT_PATTERN_CONFIG = PatternConfig()


def _value_from_candle(candle: PriceCandle | float | int) -> float:
    if isinstance(candle, (int, float)):
        return float(candle)
    if hasattr(candle, "close"):
        return float(candle.close)
    raise TypeError(f"Unsupported candle type: {type(candle)!r}")


def _kernel_weight(z: float) -> float:
    return math.exp(-(z * z) / 2.0) / math.sqrt(2.0 * math.pi)


def _cross_validation_bandwidth(values: Sequence[float]) -> float:
    """Leave-one-out cross-validation bandwidth h*; the paper then uses h = 0.3 h*."""
    if len(values) < 5:
        return 1.0

    candidates = [
        0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.25,
        1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00,
    ]
    best_bandwidth = candidates[0]
    best_score = math.inf

    for bandwidth in candidates:
        score = 0.0
        for i in range(len(values)):
            numerator = 0.0
            denominator = 0.0
            for j, price in enumerate(values):
                if i == j:
                    continue
                z = (j - i) / max(bandwidth, 1e-9)
                weight = _kernel_weight(z)
                numerator += weight * price
                denominator += weight
            prediction = numerator / denominator if denominator > 0 else values[i]
            score += (values[i] - prediction) ** 2
        if score < best_score:
            best_score = score
            best_bandwidth = bandwidth
    return best_bandwidth


def _kernel_regression(values: Sequence[float], bandwidth: float) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [float(values[0])]

    bandwidth = max(float(bandwidth), 1e-9)
    smoothed: list[float] = []
    for target in range(len(values)):
        numerator = 0.0
        denominator = 0.0
        for idx, value in enumerate(values):
            z = (idx - target) / bandwidth
            weight = _kernel_weight(z)
            numerator += weight * value
            denominator += weight
        smoothed.append(numerator / denominator if denominator > 0 else float(values[target]))
    return smoothed


def _within_tolerance(lhs: float, rhs: float, tol: float) -> bool:
    return abs(lhs - rhs) <= tol * max(abs(lhs), abs(rhs), 1.0)


def _find_local_extrema(smoothed: Sequence[float], raw_values: Sequence[float]) -> list[LocalExtremum]:
    if len(smoothed) < 3:
        return []

    extrema: list[LocalExtremum] = []
    for i in range(1, len(smoothed) - 1):
        left_slope = smoothed[i] - smoothed[i - 1]
        right_slope = smoothed[i + 1] - smoothed[i]

        if left_slope >= 0 and right_slope < 0:
            availability = min(i + 1, len(raw_values) - 1)
            extrema.append(LocalExtremum(index=i, value=float(raw_values[i]), kind="max", available_at=availability))
        elif left_slope <= 0 and right_slope > 0:
            availability = min(i + 1, len(raw_values) - 1)
            extrema.append(LocalExtremum(index=i, value=float(raw_values[i]), kind="min", available_at=availability))
    return extrema


def _confidence(values: Sequence[float], tol: float) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    if mean_value == 0:
        return 0.0
    avg_abs_deviation = sum(abs(v - mean_value) for v in values) / len(values)
    score = 1.0 - (avg_abs_deviation / max(abs(mean_value), 1e-8)) / max(tol, 1e-8)
    return max(0.0, min(1.0, score))


def _match_head_shoulders(extrema: Sequence[LocalExtremum], config: PatternConfig, inverse: bool) -> list[PatternDetection]:
    detections: list[PatternDetection] = []
    for idx in range(len(extrema) - 4):
        e1, e2, e3, e4, e5 = extrema[idx : idx + 5]
        if inverse:
            if not (e1.kind == "min" and e2.kind == "max" and e3.kind == "min" and e4.kind == "max" and e5.kind == "min"):
                continue
            if not (e3.value < e1.value and e3.value < e5.value):
                continue
            if not _within_tolerance(e1.value, e5.value, config.head_shoulders_tolerance):
                continue
            if not _within_tolerance(e2.value, e4.value, config.head_shoulders_tolerance):
                continue
            signal = "bullish"
            name = "inverse_head_and_shoulders"
        else:
            if not (e1.kind == "max" and e2.kind == "min" and e3.kind == "max" and e4.kind == "min" and e5.kind == "max"):
                continue
            if not (e3.value > e1.value and e3.value > e5.value):
                continue
            if not _within_tolerance(e1.value, e5.value, config.head_shoulders_tolerance):
                continue
            if not _within_tolerance(e2.value, e4.value, config.head_shoulders_tolerance):
                continue
            signal = "bearish"
            name = "head_and_shoulders"

        detections.append(
            PatternDetection(
                pattern_name=name,
                start_index=e1.index,
                end_index=e5.index,
                signal=signal,
                confidence=_confidence([e1.value, e2.value, e3.value, e4.value, e5.value], config.head_shoulders_tolerance),
                metadata={
                    "left_extremum": e1.value,
                    "mid_left_extremum": e2.value,
                    "head": e3.value,
                    "mid_right_extremum": e4.value,
                    "right_extremum": e5.value,
                    "neckline_tolerance": config.head_shoulders_tolerance,
                },
            )
        )
    return detections


def detect_head_and_shoulders(extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    return _match_head_shoulders(extrema, config, inverse=False)


def detect_inverse_head_and_shoulders(extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    return _match_head_shoulders(extrema, config, inverse=True)


def _match_pair_pattern(extrema: Sequence[LocalExtremum], config: PatternConfig, top: bool, kind: str) -> list[PatternDetection]:
    detections: list[PatternDetection] = []
    for idx in range(len(extrema) - 4):
        e1, e2, e3, e4, e5 = extrema[idx : idx + 5]
        peak_ok = e1.kind == "max" and e3.kind == "max" and e5.kind == "max"
        trough_ok = e1.kind == "min" and e3.kind == "min" and e5.kind == "min"
        if top:
            if not peak_ok:
                continue
            if not (e2.kind == "min" and e4.kind == "min"):
                continue
            condition = e1.value < e3.value < e5.value and e2.value > e4.value
            signal = "bearish"
            name = "broadening_top"
        else:
            if not trough_ok:
                continue
            if not (e2.kind == "max" and e4.kind == "max"):
                continue
            condition = e1.value > e3.value > e5.value and e2.value < e4.value
            signal = "bullish"
            name = "broadening_bottom"
        if condition:
            detections.append(
                PatternDetection(
                    pattern_name=name,
                    start_index=e1.index,
                    end_index=e5.index,
                    signal=signal,
                    confidence=1.0,
                    metadata={
                        "main_extrema": [e1.value, e3.value, e5.value],
                        "secondary_extrema": [e2.value, e4.value],
                    },
                )
            )
    return detections


def detect_broadening_top(extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    return _match_pair_pattern(extrema, config, top=True, kind="top")


def detect_broadening_bottom(extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    return _match_pair_pattern(extrema, config, top=False, kind="bottom")


def detect_triangle_top(extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    detections: list[PatternDetection] = []
    for idx in range(len(extrema) - 4):
        e1, e2, e3, e4, e5 = extrema[idx : idx + 5]
        if not (e1.kind == "max" and e2.kind == "min" and e3.kind == "max" and e4.kind == "min" and e5.kind == "max"):
            continue
        if not (e1.value > e3.value > e5.value and e2.value < e4.value):
            continue
        detections.append(
            PatternDetection(
                pattern_name="triangle_top",
                start_index=e1.index,
                end_index=e5.index,
                signal="bearish",
                confidence=1.0,
                metadata={
                    "peak_sequence": [e1.value, e3.value, e5.value],
                    "trough_sequence": [e2.value, e4.value],
                },
            )
        )
    return detections


def detect_triangle_bottom(extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    detections: list[PatternDetection] = []
    for idx in range(len(extrema) - 4):
        e1, e2, e3, e4, e5 = extrema[idx : idx + 5]
        if not (e1.kind == "min" and e2.kind == "max" and e3.kind == "min" and e4.kind == "max" and e5.kind == "min"):
            continue
        if not (e1.value < e3.value < e5.value and e2.value > e4.value):
            continue
        detections.append(
            PatternDetection(
                pattern_name="triangle_bottom",
                start_index=e1.index,
                end_index=e5.index,
                signal="bullish",
                confidence=1.0,
                metadata={
                    "trough_sequence": [e1.value, e3.value, e5.value],
                    "peak_sequence": [e2.value, e4.value],
                },
            )
        )
    return detections


def _find_first_valid_double_candidate(
    extrema: Sequence[LocalExtremum],
    prices: Sequence[float] | None,
    config: PatternConfig,
    is_top: bool,
) -> tuple[int, int, int | None, int, float, float, float, bool] | None:
    """Return the earliest valid double-top/double-bottom candidate.

    The structure is causal: we keep the actual extremum times and the first time the
    pattern became knowable to a trader, and when confirmation is required we search the
    underlying prices for the first true neckline break after the second extremum became
    knowable.
    """
    for i in range(len(extrema) - 1):
        current = extrema[i]
        if (is_top and current.kind != "max") or ((not is_top) and current.kind != "min"):
            continue

        for j in range(i + 1, len(extrema)):
            candidate = extrema[j]
            if (is_top and candidate.kind != "max") or ((not is_top) and candidate.kind != "min"):
                continue
            if candidate.index - current.index < config.double_min_spacing_days:
                continue

            if not _within_tolerance(current.value, candidate.value, config.double_tolerance):
                continue

            between_extrema = extrema[i + 1 : j]
            if not between_extrema:
                continue

            if is_top:
                trough = min(between_extrema, key=lambda e: e.value)
                trough_value = trough.value
                peak_min = min(current.value, candidate.value)
                trough_depth = peak_min - trough_value
                valid_trough = trough_value < current.value and trough_value < candidate.value
            else:
                peak = max(between_extrema, key=lambda e: e.value)
                peak_value = peak.value
                valley_max = max(current.value, candidate.value)
                trough_depth = peak_value - valley_max
                valid_trough = peak_value > current.value and peak_value > candidate.value

            if not valid_trough:
                continue
            if trough_depth < config.double_min_trough_depth:
                continue

            neckline_confirmed = False
            confirmation_index: int | None = None
            earliest_known_index = max(current.available_at, candidate.available_at)
            if config.double_require_neckline_confirmation:
                if prices is None:
                    for later in extrema[j + 1 :]:
                        if is_top and later.value < trough_value:
                            neckline_confirmed = True
                            confirmation_index = later.index
                            break
                        if (not is_top) and later.value > peak_value:
                            neckline_confirmed = True
                            confirmation_index = later.index
                            break
                else:
                    for time_index in range(earliest_known_index, len(prices)):
                        if is_top and prices[time_index] < trough_value:
                            neckline_confirmed = True
                            confirmation_index = time_index
                            break
                        if (not is_top) and prices[time_index] > peak_value:
                            neckline_confirmed = True
                            confirmation_index = time_index
                            break
                if not neckline_confirmed:
                    continue
            else:
                confirmation_index = earliest_known_index

            return (
                current.index,
                candidate.index,
                confirmation_index,
                earliest_known_index,
                current.value,
                candidate.value,
                trough_depth,
                neckline_confirmed,
            )

    return None


def detect_double_top(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    prices: Sequence[float] | None = None,
) -> list[PatternDetection]:
    candidate = _find_first_valid_double_candidate(extrema, prices, config, is_top=True)
    if candidate is None:
        return []

    start_idx, second_idx, confirmation_idx, available_idx, first_top, second_top, trough_depth, neckline_confirmed = candidate
    end_idx = confirmation_idx if confirmation_idx is not None else available_idx
    return [
        PatternDetection(
            pattern_name="double_top",
            start_index=start_idx,
            end_index=end_idx,
            signal="bearish",
            confidence=_confidence([first_top, second_top], config.double_tolerance),
            metadata={
                "first_top": first_top,
                "second_top": second_top,
                "second_top_available_at": available_idx,
                "confirmation_index": confirmation_idx,
                "spacing_days": second_idx - start_idx,
                "tolerance": config.double_tolerance,
                "trough_depth": trough_depth,
                "neckline_confirmed": neckline_confirmed,
            },
        )
    ]


def detect_double_bottom(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    prices: Sequence[float] | None = None,
) -> list[PatternDetection]:
    candidate = _find_first_valid_double_candidate(extrema, prices, config, is_top=False)
    if candidate is None:
        return []

    start_idx, second_idx, confirmation_idx, available_idx, first_bottom, second_bottom, trough_depth, neckline_confirmed = candidate
    end_idx = confirmation_idx if confirmation_idx is not None else available_idx
    return [
        PatternDetection(
            pattern_name="double_bottom",
            start_index=start_idx,
            end_index=end_idx,
            signal="bullish",
            confidence=_confidence([first_bottom, second_bottom], config.double_tolerance),
            metadata={
                "first_bottom": first_bottom,
                "second_bottom": second_bottom,
                "second_bottom_available_at": available_idx,
                "confirmation_index": confirmation_idx,
                "spacing_days": second_idx - start_idx,
                "tolerance": config.double_tolerance,
                "trough_depth": trough_depth,
                "neckline_confirmed": neckline_confirmed,
            },
        )
    ]


def detect_patterns(candles: Sequence[PriceCandle | float | int], config: PatternConfig = DEFAULT_PATTERN_CONFIG) -> list[PatternDetection]:
    """Detect technical patterns on a rolling window using the kernel-regression method of the Lo/Mamaysky/Wang paper.

    The detector is look-ahead-safe: it only uses information within the current rolling
    window; the detection lag is recorded in metadata but never used to create future data.
    """
    if not candles:
        return []

    values = [_value_from_candle(candle) for candle in candles]
    detections: list[PatternDetection] = []
    seen: set[tuple[str, int, int]] = set()

    for start in range(0, max(1, len(values) - config.effective_window_length + 1)):
        window = values[start : start + config.effective_window_length]
        if len(window) < config.min_extrema_count:
            continue

        h_star = _cross_validation_bandwidth(window)
        bandwidth = config.bandwidth_scale * h_star
        smoothed = _kernel_regression(window, bandwidth)
        extrema = _find_local_extrema(smoothed, window)

        if len(extrema) < config.min_extrema_count:
            continue

        for detector in (
            detect_head_and_shoulders,
            detect_inverse_head_and_shoulders,
            detect_broadening_top,
            detect_broadening_bottom,
            detect_triangle_top,
            detect_triangle_bottom,
            detect_double_top,
            detect_double_bottom,
        ):
            if detector in (detect_double_top, detect_double_bottom):
                pattern_candidates = detector(extrema, config, prices=window)
            else:
                pattern_candidates = detector(extrema, config)
            for detection in pattern_candidates:
                abs_start = start + detection.start_index
                abs_end = start + detection.end_index
                if abs_end - abs_start < config.min_pattern_span:
                    continue
                key = (detection.pattern_name, abs_start, abs_end)
                if key in seen:
                    continue
                seen.add(key)
                detections.append(
                    PatternDetection(
                        pattern_name=detection.pattern_name,
                        start_index=abs_start,
                        end_index=abs_end,
                        signal=detection.signal,
                        confidence=detection.confidence,
                        metadata={
                            **detection.metadata,
                            "window_start": start,
                            "window_end": start + config.effective_window_length - 1,
                            "detection_lag": config.detection_lag,
                            "bandwidth": bandwidth,
                            "cv_bandwidth": h_star,
                        },
                    )
                )

    return sorted(detections, key=lambda detection: (detection.start_index, detection.end_index))


if __name__ == "__main__":
    import random

    rng = random.Random(42)
    price_path = [100.0]
    for _ in range(200):
        price_path.append(price_path[-1] * (1.0 + rng.gauss(0.0, 0.012)))

    detections = detect_patterns(price_path, PatternConfig())
    print(f"patterns_detected={len(detections)}")
    for detection in detections[:10]:
        print(detection.pattern_name, detection.start_index, detection.end_index, detection.signal, round(detection.confidence, 4))
