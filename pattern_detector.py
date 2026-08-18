"""Research-grade classical chart-pattern detector for the Market Maker Simulator project.

This module is inspired by Lo, Mamaysky & Wang's smoothing/extrema framework, but it is not
an exact reproduction. It is designed as a frozen, deterministic baseline that can be applied
unchanged to positive-price synthetic paths and real OHLCV candles. Geometry is identified in
relative-log close-price space after Gaussian local-linear smoothing. Raw OHLCV is retained only
for provenance, annotations, and causal context features.

Critical backtest contract:
- ``available_at_index`` is the close of the last bar used to make a detection available.
- ``earliest_execution_index`` is strictly later (next bar by default).
- ``expected_direction`` describes the classical interpretation; the detector never emits a
  long/short trading instruction or uses future profitability to decide whether a pattern exists.
- ``DetectionResult.events`` contains unique causal events; ``candidates`` and ``clusters`` retain
  multi-scale research detail and must not be mistaken for independent strategy signals.

Missing-bar policy, corporate-action adjustment, session construction, and execution/slippage
models belong to the upstream data loader or downstream backtester, not this detector.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import groupby
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


_EPS = 1e-12
_GEOMETRY_FLOOR = 1e-10  # numerical guard only; economic thresholds are dimensionless/noise-scaled
_HASH_GEOMETRY_DECIMALS = 10
DETECTOR_VERSION = "2.1.0"
SMOOTHING_CALIBRATION_ID = "synthetic_geometry_v1_no_returns"
GEOMETRY_CALIBRATION_ID = "synthetic_geometry_v3_no_returns"

__all__ = [
    "DETECTOR_VERSION",
    "SMOOTHING_CALIBRATION_ID",
    "GEOMETRY_CALIBRATION_ID",
    "RunContext",
    "PatternDetection",
    "PatternEventCluster",
    "DetectionResult",
    "PatternConfig",
    "DEFAULT_PATTERN_CONFIG",
    "detect_pattern_universe",
    "detect_pattern_candidates",
    "detect_patterns",
]


@dataclass(frozen=True)
class FrozenDict(Mapping[str, Any]):
    """Small picklable immutable mapping used for emitted research metadata."""

    _items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FrozenDict":
        if mapping is None:
            return cls()
        items = [(str(k), _freeze_value(v)) for k, v in mapping.items()]
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError("metadata keys must remain unique after string normalization")
        return cls(tuple(sorted(items, key=lambda item: item[0])))

    def __getitem__(self, key: str) -> Any:
        for existing, value in self._items:
            if existing == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _freeze_value(value: Any) -> Any:
    """Recursively freeze metadata while retaining multiprocessing/pickle compatibility."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict.from_mapping(value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    if isinstance(value, np.ndarray):
        # Research records must be recursively immutable. Converting arrays to tuples also
        # avoids a subtle mutability hole in frozen dataclasses and remains pickle-friendly.
        return tuple(_freeze_value(v) for v in value.tolist())
    if isinstance(value, np.generic):
        return _freeze_value(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("research metadata cannot contain NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool, bytes)):
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    # Reject unknown custom objects instead of freezing a potentially non-deterministic repr
    # (which may contain a memory address and would undermine experiment reproducibility).
    raise ValueError(f"unsupported research metadata type: {type(value)!r}")


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> FrozenDict:
    return FrozenDict.from_mapping(metadata)


@runtime_checkable
class CandleLike(Protocol):
    """Minimal public market-data interface accepted by the detector."""

    close: float


@dataclass(frozen=True)
class RunContext:
    """Optional experiment identity attached to detections and event IDs.

    Supplying this is strongly recommended for stored Monte Carlo/backtest results.  It keeps
    event identifiers distinct across assets/worlds even when two paths happen to have the same
    numerical structure.
    """

    run_id: str | None = None
    market_world: str | None = None
    seed: int | None = None
    asset: str | None = None
    timeframe: str | None = None
    dataset_id: str | None = None
    git_commit: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "market_world", "asset", "timeframe", "dataset_id", "git_commit"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string when supplied")
                if not value.strip():
                    raise ValueError(f"{name} cannot be blank when supplied")
        if self.seed is not None:
            if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
                raise ValueError("seed must be an integer when supplied")
            object.__setattr__(self, "seed", int(self.seed))

    def stable_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatternDetection:
    """One causal geometric pattern event.

    ``available_at_index`` is the close-of-bar information timestamp: the detector may expose
    the pattern only after that bar has completed. ``earliest_execution_index`` is therefore
    the next bar by default. This distinction makes it impossible for a conventional backtest
    to decide using close[t] and then execute at that same already-known close[t].

    ``expected_direction`` is a classical descriptive label, never an execution instruction.
    The detector deliberately emits no long/short trade signal.
    """

    pattern_name: str
    start_index: int
    end_index: int
    expected_direction: str
    geometry_fit_score: float
    metadata: Mapping[str, Any]
    available_at_index: int
    event_id: str | None = None
    earliest_execution_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_name, str) or not self.pattern_name.strip():
            raise ValueError("pattern_name cannot be empty")
        if self.expected_direction not in {"bullish", "bearish", "neutral"}:
            raise ValueError("expected_direction must be bullish, bearish or neutral")
        if self.earliest_execution_index is None:
            object.__setattr__(self, "earliest_execution_index", int(self.available_at_index) + 1)
        indices = (
            self.start_index,
            self.end_index,
            self.available_at_index,
            self.earliest_execution_index,
        )
        if any(isinstance(v, (bool, np.bool_)) for v in indices):
            raise ValueError("pattern indices cannot be booleans")
        if any(not isinstance(v, (int, np.integer)) for v in indices):
            raise ValueError("pattern indices must be integers")
        object.__setattr__(self, "start_index", int(self.start_index))
        object.__setattr__(self, "end_index", int(self.end_index))
        object.__setattr__(self, "available_at_index", int(self.available_at_index))
        object.__setattr__(self, "earliest_execution_index", int(self.earliest_execution_index))
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("invalid pattern interval")
        if self.available_at_index < self.end_index:
            raise ValueError("available_at_index cannot precede pattern end")
        if self.earliest_execution_index <= self.available_at_index:
            raise ValueError("earliest_execution_index must be strictly after available_at_index")
        if isinstance(self.geometry_fit_score, (bool, np.bool_)) or not isinstance(
            self.geometry_fit_score, (int, float, np.integer, np.floating)
        ):
            raise ValueError("geometry_fit_score must be a real numeric scalar")
        score = float(self.geometry_fit_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("geometry_fit_score must be finite and lie in [0, 1]")
        object.__setattr__(self, "geometry_fit_score", score)
        if self.event_id is not None:
            if not isinstance(self.event_id, str):
                raise ValueError("event_id must be a string when supplied")
            if not self.event_id.strip():
                raise ValueError("event_id cannot be blank")
        frozen = _freeze_metadata(self.metadata)
        if frozen.get("trade_signal", "neutral") != "neutral":
            raise ValueError("detector metadata trade_signal must remain neutral")
        object.__setattr__(self, "metadata", frozen)

    @property
    def pattern_start_index(self) -> int:
        return self.start_index

    @property
    def pattern_end_index(self) -> int:
        return self.end_index

    @property
    def confidence(self) -> float:
        """Compatibility alias. Scores are pattern-specific geometry fits, not probabilities."""
        return self.geometry_fit_score

    @property
    def fit_score(self) -> float:
        return self.geometry_fit_score

    @property
    def detected_at_index(self) -> int:
        return self.available_at_index

    @property
    def information_available_at_index(self) -> int:
        return self.available_at_index

    @property
    def earliest_actionable_index(self) -> int:
        """Compatibility alias for the first bar a conventional backtest may execute on."""
        return self.earliest_execution_index

    @property
    def trade_signal(self) -> str:
        return "neutral"

    @property
    def breakout_confirmed_by_availability(self) -> bool:
        """Whether a breakout had already occurred by this event's information timestamp."""
        return bool(self.metadata.get("breakout_confirmed_by_availability", False))


@dataclass(frozen=True)
class LocalExtremum:
    """One structural pivot.

    Geometry uses ``smooth_value`` at ``smooth_index``. ``raw_index``/``raw_value`` are retained
    only for chart annotation and raw OHLC refinement, preventing noise from being reintroduced
    into the pattern classifier after smoothing.
    """

    smooth_index: int
    raw_index: int
    smooth_value: float
    raw_value: float
    kind: str
    confirmed_at: int
    prominence: float

    def __post_init__(self) -> None:
        if self.kind not in {"max", "min"}:
            raise ValueError("LocalExtremum.kind must be 'max' or 'min'")
        for name in ("smooth_index", "raw_index", "confirmed_at"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.smooth_index < 0 or self.raw_index < 0:
            raise ValueError("extremum indices cannot be negative")
        if self.confirmed_at <= self.smooth_index:
            raise ValueError("confirmed_at must follow the smoothed pivot")
        for name in ("smooth_value", "raw_value", "prominence"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be a real numeric scalar")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, numeric)
        if self.prominence < 0.0:
            raise ValueError("prominence cannot be negative")

    @property
    def value(self) -> float:
        """Geometry value (smoothed), retained as a compatibility alias."""
        return self.smooth_value

    @property
    def geometry_index(self) -> int:
        return self.smooth_index

    @property
    def raw_pivot_index(self) -> int:
        return self.raw_index

    @property
    def pivot_index(self) -> int:
        return self.smooth_index

    @property
    def index(self) -> int:
        return self.smooth_index

    @property
    def slope_confirmation_index(self) -> int:
        return self.confirmed_at


@dataclass(frozen=True)
class _PatternCandidate:
    """Internal geometry-only candidate; never exposed as a strategy-time event."""

    pattern_name: str
    start_index: int
    end_index: int
    expected_direction: str
    geometry_fit_score: float
    metadata: Mapping[str, Any]
    local_shape_confirmation_index: int

    def __post_init__(self) -> None:
        if self.expected_direction not in {"bullish", "bearish", "neutral"}:
            raise ValueError("invalid expected_direction")
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("invalid local candidate interval")
        if self.local_shape_confirmation_index <= self.end_index:
            raise ValueError("local_shape_confirmation_index must follow the pattern end")
        score = float(self.geometry_fit_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("candidate geometry_fit_score must lie in [0, 1]")
        frozen = _freeze_metadata(self.metadata)
        if frozen.get("trade_signal", "neutral") != "neutral":
            raise ValueError("geometry candidate cannot create a trade signal")
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True)
class PatternEventCluster:
    """Post-hoc summary of raw candidates assigned to one immutable causal anchor."""

    event_id: str
    pattern_name: str
    primary_detection: PatternDetection
    candidates: tuple[PatternDetection, ...]
    supporting_core_windows: tuple[int, ...]
    first_available_at_index: int
    last_available_at_index: int

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class DetectionResult:
    """Complete output preserving raw views, unique causal events and provenance."""

    candidates: tuple[PatternDetection, ...]
    events: tuple[PatternDetection, ...]
    clusters: tuple[PatternEventCluster, ...]
    detector_version: str
    config_hash: str
    close_series_fingerprint: str
    full_market_data_fingerprint: str
    geometry_fingerprint: str
    run_context: RunContext | None = None
    config_snapshot: PatternConfig | None = None

    @property
    def full_series_fingerprint(self) -> str:
        """Backward-compatible alias; now truly fingerprints all supplied OHLCV/labels."""
        return self.full_market_data_fingerprint

    @property
    def scale_observation_lengths(self) -> tuple[tuple[int, int], ...]:
        if self.config_snapshot is None:
            return ()
        return tuple(
            (core, self.config_snapshot.observation_length(core))
            for core in self.config_snapshot.resolved_core_window_lengths()
        )

    @property
    def full_multiscale_warmup_bars(self) -> int:
        if self.config_snapshot is None:
            return 0
        return self.config_snapshot.full_multiscale_warmup_bars()


@dataclass(frozen=True)
class PatternConfig:
    """Frozen research configuration for the classical baseline detector.

    The detector is Lo-inspired, not a line-by-line replication. It uses predefined multi-scale
    local-linear Gaussian smoothing so that smoothing choices are fixed before profitability is
    observed. All geometry thresholds are dimensionless in relative-log price space.
    """

    research_profile_name: str = "classical_detector_v2_1_backtest_ready"

    # Multi-scale pattern horizons. These are frozen baseline scales, not profit-optimized values.
    core_window_lengths: tuple[int, ...] = (35, 55, 80)
    leading_context_bars: int = 3
    trailing_context_bars: int = 3
    smoothing_context_bandwidth_multiple: float = 2.0
    # Require the shape to finish in the last 15% of each core horizon. Keeping freshness
    # dimensionless prevents a hidden 5-bar rule from meaning different things at each scale.
    max_pattern_age_fraction: float = 0.15

    # Local-linear Gaussian smoothing. A narrow fixed ensemble is aggregated by pointwise median
    # to make extrema less sensitive to one arbitrary bandwidth constant.
    kernel_bandwidth_fraction: float = 0.045
    smoothing_bandwidth_multipliers: tuple[float, ...] = (0.90, 1.00, 1.10)
    min_kernel_bandwidth: float = 1.5
    max_kernel_bandwidth: float = 12.0

    # Extrema/noise filtering. The acceptance threshold scales with residual observation noise;
    # no fixed percentage floor is used, preserving vertical-volatility invariance.
    prominence_noise_scale: float = 0.75
    prominence_lookaround_bars: int = 3
    pivot_refine_radius: int = 1
    min_pivot_separation_bars: int = 2

    # General pattern-size filter. Pattern amplitude is measured in residual-noise units.
    min_pattern_span_bars: int = 5
    pattern_height_noise_scale: float = 2.00

    # Head & shoulders geometry.
    max_shoulder_error_ratio: float = 0.75
    max_neckline_slope_height_ratio: float = 1.25
    max_time_asymmetry_ratio: float = 3.0
    min_head_prominence_ratio: float = 0.15
    min_shoulder_neckline_clearance_ratio: float = 0.05

    # Double top/bottom geometry. Doubles use consecutive max-min-max / min-max-min pivots only.
    max_double_level_error_ratio: float = 0.75
    min_double_spacing_bars: int = 3
    min_double_spacing_fraction: float = 0.12
    max_double_spacing_fraction: float = 0.80
    require_double_neckline_confirmation: bool = False

    # Rectangle geometry.
    max_rectangle_boundary_error_ratio: float = 0.35

    # Triangle/broadening geometry.
    triangle_min_convergence: float = 0.12
    triangle_max_apex_distance_spans: float = 2.5
    broadening_min_expansion: float = 0.10

    # Causal clustering against an immutable primary anchor.
    cluster_min_interval_overlap: float = 0.40
    cluster_min_pivot_match_fraction: float = 0.60
    cluster_neutral_min_shared_pivot_fraction: float = 0.80
    cluster_pivot_tolerance_bars: int = 2
    cluster_center_tolerance_fraction: float = 0.35
    competition_overlap_threshold: float = 0.40

    # Context and execution semantics.
    context_lookback_bars: int = 10
    execution_delay_bars: int = 1

    def __post_init__(self) -> None:
        def is_plain_int(value: Any) -> bool:
            return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

        def finite(name: str) -> float:
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be a real numeric scalar")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            return value

        def nonnegative(name: str) -> float:
            value = finite(name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            return value

        def positive(name: str) -> float:
            value = finite(name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            return value

        if not isinstance(self.research_profile_name, str) or not self.research_profile_name.strip():
            raise ValueError("research_profile_name cannot be empty")
        if not self.core_window_lengths:
            raise ValueError("core_window_lengths must contain at least one window")
        windows = tuple(sorted(int(v) for v in self.core_window_lengths))
        if len(set(windows)) != len(windows):
            raise ValueError("core_window_lengths cannot contain duplicates")
        if any(not is_plain_int(v) or int(v) < 8 for v in self.core_window_lengths):
            raise ValueError("every core window length must be an integer >= 8")
        object.__setattr__(self, "core_window_lengths", windows)
        if not is_plain_int(self.leading_context_bars) or self.leading_context_bars < 1:
            raise ValueError("leading_context_bars must be an integer >= 1")
        if not is_plain_int(self.trailing_context_bars) or self.trailing_context_bars < 1:
            raise ValueError("trailing_context_bars must be an integer >= 1")
        positive("smoothing_context_bandwidth_multiple")
        age_fraction = finite("max_pattern_age_fraction")
        if not 0.0 <= age_fraction <= 1.0:
            raise ValueError("max_pattern_age_fraction must lie in [0, 1]")

        positive("kernel_bandwidth_fraction")
        positive("min_kernel_bandwidth")
        positive("max_kernel_bandwidth")
        if self.max_kernel_bandwidth < self.min_kernel_bandwidth:
            raise ValueError("max_kernel_bandwidth must be >= min_kernel_bandwidth")
        if not self.smoothing_bandwidth_multipliers:
            raise ValueError("smoothing_bandwidth_multipliers cannot be empty")
        if any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, float, np.integer, np.floating))
            for v in self.smoothing_bandwidth_multipliers
        ):
            raise ValueError("every smoothing bandwidth multiplier must be a real numeric scalar")
        multipliers = tuple(float(v) for v in self.smoothing_bandwidth_multipliers)
        if any(not math.isfinite(v) or v <= 0.0 for v in multipliers):
            raise ValueError("every smoothing bandwidth multiplier must be finite and positive")
        if len(set(round(v, 12) for v in multipliers)) != len(multipliers):
            raise ValueError("smoothing_bandwidth_multipliers cannot contain duplicates")
        if not any(abs(v - 1.0) <= 1e-12 for v in multipliers):
            raise ValueError("smoothing_bandwidth_multipliers must include 1.0")
        object.__setattr__(self, "smoothing_bandwidth_multipliers", tuple(sorted(multipliers)))

        nonnegative("prominence_noise_scale")
        if not is_plain_int(self.prominence_lookaround_bars) or self.prominence_lookaround_bars < 2:
            raise ValueError("prominence_lookaround_bars must be an integer >= 2")
        if not is_plain_int(self.pivot_refine_radius) or self.pivot_refine_radius < 0:
            raise ValueError("pivot_refine_radius must be a non-negative integer")
        if not is_plain_int(self.min_pivot_separation_bars) or self.min_pivot_separation_bars < 1:
            raise ValueError("min_pivot_separation_bars must be an integer >= 1")

        if not is_plain_int(self.min_pattern_span_bars) or self.min_pattern_span_bars < 2:
            raise ValueError("min_pattern_span_bars must be an integer >= 2")
        nonnegative("pattern_height_noise_scale")

        positive("max_shoulder_error_ratio")
        positive("max_neckline_slope_height_ratio")
        if positive("max_time_asymmetry_ratio") < 1.0:
            raise ValueError("max_time_asymmetry_ratio must be >= 1")
        head_prominence = nonnegative("min_head_prominence_ratio")
        if head_prominence > 1.0:
            raise ValueError("min_head_prominence_ratio must lie in [0, 1]")
        shoulder_clearance = nonnegative("min_shoulder_neckline_clearance_ratio")
        if shoulder_clearance >= 1.0:
            raise ValueError("min_shoulder_neckline_clearance_ratio must lie in [0, 1)")

        positive("max_double_level_error_ratio")
        if not is_plain_int(self.min_double_spacing_bars) or self.min_double_spacing_bars < 1:
            raise ValueError("min_double_spacing_bars must be an integer >= 1")
        min_fraction = finite("min_double_spacing_fraction")
        max_fraction = finite("max_double_spacing_fraction")
        if not 0.0 <= min_fraction <= max_fraction <= 1.0:
            raise ValueError("double spacing fractions must satisfy 0 <= min <= max <= 1")
        if not isinstance(self.require_double_neckline_confirmation, bool):
            raise ValueError("require_double_neckline_confirmation must be bool")

        positive("max_rectangle_boundary_error_ratio")
        convergence = finite("triangle_min_convergence")
        if not 0.0 < convergence < 1.0:
            raise ValueError("triangle_min_convergence must lie in (0, 1)")
        positive("triangle_max_apex_distance_spans")
        nonnegative("broadening_min_expansion")

        for name in (
            "cluster_min_interval_overlap",
            "cluster_min_pivot_match_fraction",
            "cluster_neutral_min_shared_pivot_fraction",
            "competition_overlap_threshold",
        ):
            value = finite(name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.cluster_neutral_min_shared_pivot_fraction < self.cluster_min_pivot_match_fraction:
            raise ValueError("neutral shared-pivot threshold cannot be below general pivot threshold")
        center_fraction = positive("cluster_center_tolerance_fraction")
        if center_fraction > 1.0:
            raise ValueError("cluster_center_tolerance_fraction must lie in (0, 1]")
        if not is_plain_int(self.cluster_pivot_tolerance_bars) or self.cluster_pivot_tolerance_bars < 0:
            raise ValueError("cluster_pivot_tolerance_bars must be a non-negative integer")
        if not is_plain_int(self.context_lookback_bars) or self.context_lookback_bars < 0:
            raise ValueError("context_lookback_bars must be a non-negative integer")
        if not is_plain_int(self.execution_delay_bars) or self.execution_delay_bars < 1:
            raise ValueError("execution_delay_bars must be an integer >= 1")

        # Canonicalize numeric scalar types so semantically identical configs hash identically
        # whether callers supplied Python scalars or NumPy scalar types.
        integer_fields = (
            "leading_context_bars", "trailing_context_bars",
            "prominence_lookaround_bars", "pivot_refine_radius", "min_pivot_separation_bars",
            "min_pattern_span_bars", "min_double_spacing_bars", "cluster_pivot_tolerance_bars",
            "context_lookback_bars", "execution_delay_bars",
        )
        float_fields = (
            "smoothing_context_bandwidth_multiple", "max_pattern_age_fraction", "kernel_bandwidth_fraction",
            "min_kernel_bandwidth", "max_kernel_bandwidth",
            "prominence_noise_scale", "pattern_height_noise_scale",
            "max_shoulder_error_ratio", "max_neckline_slope_height_ratio",
            "max_time_asymmetry_ratio", "min_head_prominence_ratio",
            "min_shoulder_neckline_clearance_ratio",
            "max_double_level_error_ratio", "min_double_spacing_fraction",
            "max_double_spacing_fraction",
            "max_rectangle_boundary_error_ratio", "triangle_min_convergence",
            "triangle_max_apex_distance_spans", "broadening_min_expansion",
            "cluster_min_interval_overlap", "cluster_min_pivot_match_fraction",
            "cluster_neutral_min_shared_pivot_fraction", "cluster_center_tolerance_fraction",
            "competition_overlap_threshold",
        )
        for name in integer_fields:
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in float_fields:
            object.__setattr__(self, name, float(getattr(self, name)))

    def resolved_core_window_lengths(self) -> tuple[int, ...]:
        return tuple(sorted(self.core_window_lengths))

    def smoothing_bandwidth(self, core_window_length: int) -> float:
        if not isinstance(core_window_length, (int, np.integer)) or int(core_window_length) < 2:
            raise ValueError("core_window_length must be an integer >= 2")
        value = self.kernel_bandwidth_fraction * float(core_window_length)
        return min(self.max_kernel_bandwidth, max(self.min_kernel_bandwidth, value))

    def smoothing_bandwidths(self, core_window_length: int) -> tuple[float, ...]:
        base = self.smoothing_bandwidth(core_window_length)
        values = {
            round(
                min(self.max_kernel_bandwidth, max(self.min_kernel_bandwidth, base * multiplier)),
                10,
            )
            for multiplier in self.smoothing_bandwidth_multipliers
        }
        return tuple(sorted(values))

    def effective_leading_context(self, core_window_length: int) -> int:
        largest = max(self.smoothing_bandwidths(core_window_length))
        guard = int(math.ceil(self.smoothing_context_bandwidth_multiple * largest))
        return max(self.leading_context_bars, guard)

    def effective_trailing_context(self, core_window_length: int) -> int:
        largest = max(self.smoothing_bandwidths(core_window_length))
        guard = int(math.ceil(self.smoothing_context_bandwidth_multiple * largest))
        return max(self.trailing_context_bars, guard)

    def resolved_max_pattern_age_bars(self, core_window_length: int) -> int:
        """Maximum core-end staleness allowed for a completed shape at this scale."""

        if not isinstance(core_window_length, (int, np.integer)) or int(core_window_length) < 2:
            raise ValueError("core_window_length must be an integer >= 2")
        return int(math.floor(self.max_pattern_age_fraction * int(core_window_length)))

    def observation_length(self, core_window_length: int) -> int:
        """Bars required before a window at this scale can be evaluated causally."""

        core = int(core_window_length)
        if core not in self.resolved_core_window_lengths():
            raise ValueError("core_window_length must be one of the configured core windows")
        return self.effective_leading_context(core) + core + self.effective_trailing_context(core)

    def full_multiscale_warmup_bars(self) -> int:
        """Bars required before every configured scale is simultaneously available."""

        return max(self.observation_length(core) for core in self.resolved_core_window_lengths())

    def detector_required_extrema(self, detector_name: str) -> int:
        """Return the explicit structural-pivot requirement for a registered detector.

        Failing loudly for unknown detector names prevents a future pattern family from being
        silently run with an accidental zero-extrema requirement. Adding a detector therefore
        requires its structural contract to be declared here and regression-tested.
        """

        normalized = detector_name.replace("detect_", "", 1)
        if normalized in {
            "head_and_shoulders",
            "inverse_head_and_shoulders",
            "broadening_top",
            "broadening_bottom",
            "rectangle_top",
            "rectangle_bottom",
            "triangle_top",
            "triangle_bottom",
        }:
            return 5
        if normalized in {"double_top", "double_bottom"}:
            return 3
        raise ValueError(
            f"Unknown detector {detector_name!r}; declare its required extrema explicitly "
            "before adding it to detector_sequence"
        )


DEFAULT_PATTERN_CONFIG = PatternConfig()


@dataclass(frozen=True)
class _MarketSeries:
    close: np.ndarray
    high: np.ndarray | None
    low: np.ndarray | None
    volume: np.ndarray | None
    labels: tuple[Any | None, ...]
    open: np.ndarray | None = None


@dataclass
class _ClusterState:
    event_id: str
    pattern_name: str
    primary: PatternDetection
    members: list[PatternDetection]


def _config_hash(config: PatternConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _run_context_hash(run_context: RunContext | None) -> str:
    if run_context is None:
        return "no-context"
    payload = json.dumps(run_context.stable_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _canonical_geometry_values(values: np.ndarray) -> np.ndarray:
    """Round below economically meaningful precision for scale-stable hashing only."""

    return np.round(np.asarray(values, dtype=float), decimals=_HASH_GEOMETRY_DECIMALS)


def _hash_float_array(digest: "hashlib._Hash", name: str, values: np.ndarray | None) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\x00" if values is None else b"\x01")
    if values is None:
        return
    arr = np.asarray(values, dtype="<f8")
    digest.update(struct.pack("<Q", int(arr.size)))
    digest.update(arr.tobytes())


def _stable_label_bytes(label: Any | None) -> bytes:
    if label is None:
        return b"none:"
    # Datetime-like objects generally expose isoformat(); use it before a repr fallback.
    isoformat = getattr(label, "isoformat", None)
    if callable(isoformat):
        try:
            return f"{type(label).__module__}.{type(label).__qualname__}:iso:{isoformat()}".encode("utf-8")
        except Exception:
            pass
    if isinstance(label, np.generic):
        return _stable_label_bytes(label.item())
    if isinstance(label, (str, int, float, bool)):
        payload = json.dumps(label, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"{type(label).__module__}.{type(label).__qualname__}:{payload}".encode("utf-8")
    raise ValueError(
        "timestamp-like candle labels must have a deterministic scalar/isoformat representation; "
        f"unsupported label type: {type(label)!r}"
    )


def _close_series_fingerprint(close: np.ndarray) -> str:
    """Exact close-only fingerprint. Quote-unit transformations intentionally change it."""

    digest = hashlib.sha256()
    _hash_float_array(digest, "close", close)
    return digest.hexdigest()[:20]


def _raw_series_fingerprint(close: np.ndarray) -> str:
    """Backward-compatible private alias for the exact close-only fingerprint."""

    return _close_series_fingerprint(close)


def _full_market_data_fingerprint(market: _MarketSeries) -> str:
    """Fingerprint every supplied OHLCV field and label used or retained by the detector."""

    digest = hashlib.sha256()
    _hash_float_array(digest, "close", market.close)
    _hash_float_array(digest, "open", market.open)
    _hash_float_array(digest, "high", market.high)
    _hash_float_array(digest, "low", market.low)
    _hash_float_array(digest, "volume", market.volume)
    digest.update(b"labels")
    digest.update(struct.pack("<Q", len(market.labels)))
    for label in market.labels:
        encoded = _stable_label_bytes(label)
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()[:20]


def _causal_market_prefix_fingerprints(market: _MarketSeries) -> tuple[str, ...]:
    """Exact OHLCV/label provenance hash for each causal market-data prefix.

    Unlike the scale-invariant geometry prefix fingerprint, this hash intentionally changes
    with quote units, OHLC refinements, volume, or timestamp labels. It is metadata provenance
    only and is deliberately excluded from geometric event identity.
    """

    digest = hashlib.sha256()
    optional = (("open", market.open), ("high", market.high), ("low", market.low), ("volume", market.volume))
    digest.update(b"market-prefix-v1")
    for name, array in optional:
        digest.update(name.encode("utf-8"))
        digest.update(b"\x01" if array is not None else b"\x00")
    fingerprints: list[str] = []
    for index in range(int(market.close.size)):
        digest.update(struct.pack("<Qd", index, float(market.close[index])))
        for _name, array in optional:
            if array is not None:
                digest.update(struct.pack("<d", float(array[index])))
        label_bytes = _stable_label_bytes(market.labels[index])
        digest.update(struct.pack("<Q", len(label_bytes)))
        digest.update(label_bytes)
        fingerprints.append(digest.hexdigest()[:20])
    return tuple(fingerprints)


def _geometry_fingerprint(normalized_values: np.ndarray) -> str:
    """Scale-invariant fingerprint of normalized close-price geometry."""

    canonical = _canonical_geometry_values(normalized_values)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", int(canonical.size)))
    digest.update(np.asarray(canonical, dtype="<f8").tobytes())
    return digest.hexdigest()[:20]


def _causal_prefix_fingerprints(normalized_values: np.ndarray) -> tuple[str, ...]:
    """Scale-invariant fingerprint for each causal normalized prefix."""

    digest = hashlib.sha256()
    fingerprints: list[str] = []
    for index, value in enumerate(_canonical_geometry_values(normalized_values)):
        digest.update(struct.pack("<Qd", index, float(value)))
        fingerprints.append(digest.hexdigest()[:20])
    return tuple(fingerprints)


def _price_relation_tolerance(*values: float) -> float:
    """Scale-aware tolerance for OHLC consistency checks.

    An absolute epsilon would make input validation depend on quote units (for example, a
    sub-nanodollar asset could tolerate an economically huge OHLC inconsistency). This guard is
    only for ordinary IEEE-754 round-off and scales with the values being compared.
    """

    scale = max((abs(float(value)) for value in values), default=0.0)
    scale = max(scale, float(np.finfo(float).tiny))
    return 16.0 * float(np.finfo(float).eps) * scale


def _finite_number(value: Any, field_name: str, index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} at index {index} is not a finite numeric scalar: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} at index {index} must be finite, got {result!r}")
    return result


def _optional_numeric_attribute(item: Any, name: str, index: int) -> float | None:
    if not hasattr(item, name):
        return None
    value = getattr(item, name)
    if value is None:
        return None
    return _finite_number(value, name, index)


def _extract_label(item: Any) -> Any | None:
    for name in ("timestamp", "datetime", "time", "date", "start_step", "candle_number"):
        if not hasattr(item, name):
            continue
        value = getattr(item, name)
        # Some candle/container classes expose methods named ``time`` or ``date``. A method is
        # not a timestamp label and must not accidentally enter chronology/fingerprint logic.
        if callable(value):
            continue
        return value
    return None


def _validate_labels_are_strictly_increasing(labels: Sequence[Any | None]) -> None:
    """Reject duplicate/out-of-order timestamp-like labels when the objects expose them.

    Numeric price-only paths have no labels and are unaffected. If labels are supplied they
    must be complete, mutually comparable, and strictly increasing; otherwise the detector
    fails loudly instead of guessing a chronology.
    """

    present = [(i, label) for i, label in enumerate(labels) if label is not None]
    if not present:
        return
    for index, label in present:
        if isinstance(label, (bool, np.bool_)):
            raise ValueError(f"boolean candle label at index {index} is not a valid chronology")
        if isinstance(label, (float, np.floating)) and not math.isfinite(float(label)):
            raise ValueError(f"numeric candle label at index {index} must be finite")
    if len(present) != len(labels):
        raise ValueError("timestamp-like labels must be present on every candle or on none of them")
    if len(present) < 2:
        return
    for (prev_index, previous), (index, current) in zip(present, present[1:]):
        try:
            increasing = current > previous
        except TypeError as exc:
            raise ValueError("timestamp-like candle labels must be mutually comparable") from exc
        if not increasing:
            raise ValueError(
                f"candle labels must be strictly increasing; indices {prev_index} and {index} "
                f"contain {previous!r} then {current!r}"
            )


def _extract_market_series(candles: Sequence[CandleLike | float | int]) -> _MarketSeries:
    if len(candles) == 0:
        return _MarketSeries(
            close=np.asarray([], dtype=float),
            open=None,
            high=None,
            low=None,
            volume=None,
            labels=(),
        )

    closes: list[float] = []
    opens: list[float | None] = []
    highs: list[float | None] = []
    lows: list[float | None] = []
    volumes: list[float | None] = []
    labels: list[Any | None] = []

    for index, candle in enumerate(candles):
        if isinstance(candle, (bool, np.bool_)):
            raise TypeError(f"boolean at index {index} is not a valid market price")
        if isinstance(candle, (int, float, np.number)):
            close = _finite_number(candle, "price", index)
            open_value = None
            high = None
            low = None
            volume = None
            label = None
        else:
            if not hasattr(candle, "close"):
                raise TypeError(
                    f"Unsupported candle at index {index}: expected a number or object with .close"
                )
            close = _finite_number(getattr(candle, "close"), "close", index)
            open_value = _optional_numeric_attribute(candle, "open", index)
            high = _optional_numeric_attribute(candle, "high", index)
            low = _optional_numeric_attribute(candle, "low", index)
            volume = _optional_numeric_attribute(candle, "volume", index)
            label = _extract_label(candle)

            for field_name, value in (("open", open_value), ("high", high), ("low", low)):
                if value is not None and value <= 0.0:
                    raise ValueError(f"{field_name} at index {index} must be strictly positive")
            if (high is None) != (low is None):
                raise ValueError(f"high and low must either both be supplied or both be absent at index {index}")
            if high is not None and low is not None:
                if low - high > _price_relation_tolerance(high, low):
                    raise ValueError(f"high at index {index} cannot be below low")
            if high is not None and close - high > _price_relation_tolerance(close, high):
                raise ValueError(f"close at index {index} cannot exceed high")
            if low is not None and low - close > _price_relation_tolerance(close, low):
                raise ValueError(f"close at index {index} cannot be below low")
            if open_value is not None and high is not None and open_value - high > _price_relation_tolerance(open_value, high):
                raise ValueError(f"open at index {index} cannot exceed high")
            if open_value is not None and low is not None and low - open_value > _price_relation_tolerance(open_value, low):
                raise ValueError(f"open at index {index} cannot be below low")
            if volume is not None and volume < 0.0:
                raise ValueError(f"volume at index {index} cannot be negative")

        if close <= 0.0:
            raise ValueError(
                f"price/close at index {index} must be strictly positive; "
                "convert non-positive latent synthetic processes into a positive price series first"
            )

        closes.append(close)
        opens.append(open_value)
        highs.append(high)
        lows.append(low)
        volumes.append(volume)
        labels.append(label)

    _validate_labels_are_strictly_increasing(labels)

    def optional_array(values: list[float | None], field_name: str) -> np.ndarray | None:
        present = [value is not None for value in values]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"{field_name} must be present on every candle or on none of them")
        array = np.asarray(values, dtype=float)
        array.flags.writeable = False
        return array

    close_array = np.asarray(closes, dtype=float)
    close_array.flags.writeable = False
    return _MarketSeries(
        close=close_array,
        open=optional_array(opens, "open"),
        high=optional_array(highs, "high"),
        low=optional_array(lows, "low"),
        volume=optional_array(volumes, "volume"),
        labels=tuple(labels),
    )


def _relative_log_series(close: np.ndarray) -> np.ndarray:
    """Causal, multiplicatively scale-invariant geometry: log(P_t/P_0)."""

    if close.size == 0:
        return np.asarray([], dtype=float)
    baseline = float(close[0])
    # Difference-of-logs avoids overflow/underflow in close / baseline for extreme but finite
    # quote units while preserving exact multiplicative scale invariance in real arithmetic.
    return np.log(close) - math.log(baseline)


@lru_cache(maxsize=2048)
def _local_linear_smoother_matrix(length: int, bandwidth: float) -> np.ndarray:
    """Gaussian local-linear smoother matrix for equally spaced bars.

    Unlike the Nadaraya-Watson local-constant estimator, local-linear regression reproduces
    linear trends even at sample boundaries and substantially reduces first-order boundary bias.
    """

    if length <= 0:
        return np.empty((0, 0), dtype=float)
    if length == 1:
        matrix = np.ones((1, 1), dtype=float)
        matrix.flags.writeable = False
        return matrix

    x = np.arange(length, dtype=float)
    bw = max(float(bandwidth), _EPS)
    matrix = np.empty((length, length), dtype=float)
    for target in range(length):
        dx = x - float(target)
        weights = np.exp(-0.5 * (dx / bw) ** 2)
        s0 = float(np.sum(weights))
        s1 = float(np.sum(weights * dx))
        s2 = float(np.sum(weights * dx * dx))
        denominator = s0 * s2 - s1 * s1
        if denominator <= _EPS:
            row = weights / max(s0, _EPS)
        else:
            # Intercept of the weighted local linear fit evaluated at dx=0.
            row = weights * (s2 - s1 * dx) / denominator
        row_sum = float(np.sum(row))
        if abs(row_sum) > _EPS:
            row = row / row_sum
        matrix[target, :] = row
    matrix.flags.writeable = False
    return matrix


def _local_linear_regression(values: Sequence[float], bandwidth: float) -> list[float]:
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [float(values[0])]
    y = np.asarray(values, dtype=float)
    matrix = _local_linear_smoother_matrix(len(values), max(round(float(bandwidth),8), _EPS))
    return (matrix @ y).tolist()


def _kernel_regression(values: Sequence[float], bandwidth: float) -> list[float]:
    """Compatibility alias for the production local-linear Gaussian smoother."""

    return _local_linear_regression(values, bandwidth)


def _ensemble_local_linear_smoothing(
    values: Sequence[float],
    bandwidths: Sequence[float],
) -> list[float]:
    """Pointwise-median ensemble over a narrow, pre-registered bandwidth neighborhood."""

    if len(values) == 0:
        return []
    unique = tuple(sorted({round(float(bw), 8) for bw in bandwidths if float(bw) > 0.0}))
    if not unique:
        raise ValueError("at least one positive smoothing bandwidth is required")
    curves = np.asarray([_local_linear_regression(values, bw) for bw in unique], dtype=float)
    return np.median(curves, axis=0).tolist()


def _stable_nonnegative_mean(values: np.ndarray) -> float:
    """Finite mean for non-negative finite values without overflow in the accumulator."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    scale = float(np.max(array))
    if scale <= 0.0:
        return 0.0
    mean_scaled = float(np.mean(array / scale))
    result = scale * mean_scaled
    if not math.isfinite(result):
        # This should only be reachable for values outside practical market-data ranges.
        raise ValueError("numeric aggregation overflowed despite finite input values")
    return result


def _local_log_volatility(values: Sequence[float]) -> float:
    """Conventional standard deviation of one-bar log changes, for descriptive metadata."""

    if len(values) < 3:
        return 0.0
    diffs = np.diff(np.asarray(values, dtype=float))
    return float(np.std(diffs, ddof=0)) if diffs.size >= 2 else 0.0


def _robust_log_volatility(values: Sequence[float]) -> float:
    """Robust one-bar log volatility used for detector thresholds.

    Median absolute deviation limits the influence of the very structural swings we are trying
    to detect. A zero MAD is retained as zero rather than replaced by standard deviation, so an
    isolated structural jump cannot inflate the noise threshold used to detect that same jump.
    """

    if len(values) < 3:
        return 0.0
    diffs = np.diff(np.asarray(values, dtype=float))
    if diffs.size < 2:
        return 0.0
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    sigma = 1.4826 * mad
    # If almost all changes are identical and only isolated jumps remain, MAD=0 is the desired
    # robust noise estimate; falling back to standard deviation would let the structural jump
    # define its own detection threshold.
    return max(0.0, sigma)


def _robust_residual_noise(raw_values: Sequence[float], smoothed_values: Sequence[float]) -> float:
    """Robust log-price observation-noise scale after removing the smooth structural curve."""

    if len(raw_values) != len(smoothed_values):
        raise ValueError("raw and smoothed sequences must have the same length")
    if len(raw_values) < 3:
        return 0.0
    residuals = np.asarray(raw_values, dtype=float) - np.asarray(smoothed_values, dtype=float)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return max(0.0, 1.4826 * mad)


def _adaptive_height_threshold(noise_scale: float, config: PatternConfig) -> float:
    """Minimum structural amplitude in relative-log units.

    Thresholding against smoothed residual noise avoids the self-defeating behavior of using
    within-pattern return volatility, where the intended pattern swings inflate the threshold
    used to decide whether that same pattern is large enough.
    """

    if not math.isfinite(float(noise_scale)) or noise_scale < 0.0:
        raise ValueError("noise_scale must be finite and non-negative")
    return max(_GEOMETRY_FLOOR, config.pattern_height_noise_scale * float(noise_scale))


def _smooth_prominence(smoothed: Sequence[float], smooth_index: int, kind: str, lookaround: int) -> float:
    if smooth_index <= 0 or smooth_index >= len(smoothed) - 1:
        return 0.0
    left = list(smoothed[max(0, smooth_index - lookaround) : smooth_index])
    right = list(smoothed[smooth_index + 1 : min(len(smoothed), smooth_index + lookaround + 1)])
    if not left or not right:
        return 0.0
    pivot = float(smoothed[smooth_index])
    if kind == "max":
        higher_saddle = max(min(left), min(right))
        return max(0.0, pivot - higher_saddle)
    if kind == "min":
        lower_saddle = min(max(left), max(right))
        return max(0.0, lower_saddle - pivot)
    return 0.0


def _find_local_extrema(
    smoothed: Sequence[float],
    normalized_values: Sequence[float],
    config: PatternConfig,
) -> list[LocalExtremum]:
    if len(smoothed) < 3:
        return []

    noise_scale = _robust_residual_noise(normalized_values, smoothed)
    prominence_threshold = max(
        _GEOMETRY_FLOOR,
        config.prominence_noise_scale * noise_scale,
    )
    radius = max(0, int(config.pivot_refine_radius))
    extrema: list[LocalExtremum] = []

    for i in range(1, len(smoothed) - 1):
        left_slope = smoothed[i] - smoothed[i - 1]
        right_slope = smoothed[i + 1] - smoothed[i]
        if left_slope >= 0.0 and right_slope < 0.0:
            kind = "max"
        elif left_slope <= 0.0 and right_slope > 0.0:
            kind = "min"
        else:
            continue

        prominence = _smooth_prominence(
            smoothed,
            i,
            kind,
            max(2, int(config.prominence_lookaround_bars)),
        )
        if prominence < prominence_threshold:
            continue

        start = max(0, i - radius)
        end = min(len(normalized_values), i + radius + 1)
        if kind == "max":
            raw_index = max(range(start, end), key=lambda idx: normalized_values[idx])
        else:
            raw_index = min(range(start, end), key=lambda idx: normalized_values[idx])

        candidate = LocalExtremum(
            smooth_index=i,
            raw_index=raw_index,
            smooth_value=float(smoothed[i]),
            raw_value=float(normalized_values[raw_index]),
            kind=kind,
            confirmed_at=i + 1,
            prominence=prominence,
        )

        if extrema and candidate.kind == extrema[-1].kind:
            last = extrema[-1]
            replace_last = candidate.value > last.value if kind == "max" else candidate.value < last.value
            if replace_last:
                extrema[-1] = candidate
            continue

        if extrema and candidate.smooth_index - extrema[-1].smooth_index < config.min_pivot_separation_bars:
            # Close opposite pivots are residual noise. Keep the stronger event but then repair
            # alternation below so this cannot create an invalid same-kind sequence.
            if candidate.prominence > extrema[-1].prominence:
                extrema[-1] = candidate
            continue

        extrema.append(candidate)

    filtered: list[LocalExtremum] = []
    for extremum in extrema:
        if filtered and filtered[-1].kind == extremum.kind:
            last = filtered[-1]
            replace_last = extremum.value > last.value if extremum.kind == "max" else extremum.value < last.value
            if replace_last:
                filtered[-1] = extremum
        else:
            filtered.append(extremum)
    return filtered


def _line_fit(points: Sequence[tuple[int, float]]) -> tuple[float, float, float, float]:
    """Return slope, intercept, bounded R^2-like fit and normalized RMSE.

    The closed-form least-squares calculation avoids platform/BLAS-dependent solver details for
    the tiny two/three-point boundary fits used by classical chart patterns.
    """

    if len(points) < 2:
        return 0.0, 0.0, 0.0, math.inf
    x = np.asarray([p[0] for p in points], dtype=float)
    y = np.asarray([p[1] for p in points], dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    denominator = float(np.sum(centered_x * centered_x))
    if denominator <= _EPS:
        return 0.0, y_mean, 0.0, math.inf
    slope = float(np.sum(centered_x * (y - y_mean)) / denominator)
    intercept = y_mean - slope * x_mean
    fitted = slope * x + intercept
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    centered_y = y - y_mean
    sst = float(np.sum(centered_y**2))
    if len(points) == 2:
        r2 = 1.0
    elif sst <= _EPS:
        r2 = 1.0 if sse <= _EPS else 0.0
    else:
        r2 = 1.0 - sse / sst
    value_range = max(float(np.max(y) - np.min(y)), _EPS)
    nrmse = math.sqrt(sse / max(1, len(points))) / value_range
    return slope, intercept, max(0.0, min(1.0, r2)), float(nrmse)


def _line_value(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def _time_symmetry_score(left_span: int, right_span: int) -> tuple[float, float]:
    if left_span <= 0 or right_span <= 0:
        return 0.0, math.inf
    ratio = max(left_span, right_span) / min(left_span, right_span)
    score = math.exp(-abs(math.log(left_span / right_span)))
    return max(0.0, min(1.0, score)), ratio


def _candidate_metadata(extrema: Sequence[LocalExtremum]) -> dict[str, Any]:
    return {
        "pivot_geometry_indices_local": [e.smooth_index for e in extrema],
        "pivot_raw_indices_local": [e.raw_index for e in extrema],
        "pivot_kinds": [e.kind for e in extrema],
        "pivot_values_smooth_normalized": [e.smooth_value for e in extrema],
        "pivot_values_raw_normalized": [e.raw_value for e in extrema],
        "pivot_prominences": [e.prominence for e in extrema],
        "geometry_space": "relative_log_price_smoothed",
    }


def _new_candidate(
    pattern_name: str,
    block: Sequence[LocalExtremum],
    expected_direction: str,
    fit: float,
    metadata: Mapping[str, Any],
) -> _PatternCandidate:
    return _PatternCandidate(
        pattern_name=pattern_name,
        start_index=block[0].index,
        end_index=block[-1].index,
        expected_direction=expected_direction,
        geometry_fit_score=max(0.0, min(1.0, fit)),
        metadata=metadata,
        local_shape_confirmation_index=block[-1].slope_confirmation_index,
    )


def _match_head_shoulders(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig,
    inverse: bool,
) -> list[_PatternCandidate]:
    detections: list[_PatternCandidate] = []
    expected_kinds = (
        ("min", "max", "min", "max", "min")
        if inverse
        else ("max", "min", "max", "min", "max")
    )

    for idx in range(len(extrema) - 4):
        block = tuple(extrema[idx : idx + 5])
        e1, e2, e3, e4, e5 = block
        if tuple(e.kind for e in block) != expected_kinds:
            continue

        if inverse:
            if not (e3.value < e1.value and e3.value < e5.value):
                continue
            name = "inverse_head_and_shoulders"
            expected_direction = "bullish"
        else:
            if not (e3.value > e1.value and e3.value > e5.value):
                continue
            name = "head_and_shoulders"
            expected_direction = "bearish"

        neckline_slope = (e4.value - e2.value) / max(e4.index - e2.index, 1)
        neckline_intercept = e2.value - neckline_slope * e2.index
        neckline_at_left = _line_value(neckline_slope, neckline_intercept, e1.index)
        neckline_at_head = _line_value(neckline_slope, neckline_intercept, e3.index)
        neckline_at_right = _line_value(neckline_slope, neckline_intercept, e5.index)

        # Detrend the three peaks/troughs by the fitted neckline before judging shoulders and
        # head prominence. This supports genuinely sloping necklines without mistaking the
        # underlying local drift for shoulder asymmetry.
        if inverse:
            left_clearance = neckline_at_left - e1.value
            right_clearance = neckline_at_right - e5.value
            head_clearance = neckline_at_head - e3.value
        else:
            left_clearance = e1.value - neckline_at_left
            right_clearance = e5.value - neckline_at_right
            head_clearance = e3.value - neckline_at_head

        pattern_height = head_clearance
        if pattern_height <= _EPS:
            continue
        minimum_clearance = config.min_shoulder_neckline_clearance_ratio * pattern_height
        if left_clearance < minimum_clearance or right_clearance < minimum_clearance:
            continue
        shoulder_clearance_average = 0.5 * (left_clearance + right_clearance)
        if head_clearance <= max(left_clearance, right_clearance):
            continue

        head_prominence = head_clearance - shoulder_clearance_average
        head_prominence_ratio = head_prominence / pattern_height
        if head_prominence_ratio < config.min_head_prominence_ratio:
            continue

        shoulder_error = abs(left_clearance - right_clearance)
        shoulder_error_ratio = shoulder_error / pattern_height
        neckline_slope_height_ratio = abs(e4.value - e2.value) / pattern_height
        if shoulder_error_ratio > config.max_shoulder_error_ratio:
            continue
        if neckline_slope_height_ratio > config.max_neckline_slope_height_ratio:
            continue

        time_score, asymmetry_ratio = _time_symmetry_score(e3.index - e1.index, e5.index - e3.index)
        if asymmetry_ratio > config.max_time_asymmetry_ratio:
            continue

        shoulder_score = 1.0 - min(1.0, shoulder_error_ratio / max(config.max_shoulder_error_ratio, _EPS))
        neckline_score = 1.0 - min(
            1.0, neckline_slope_height_ratio / max(config.max_neckline_slope_height_ratio, _EPS)
        )
        head_score = min(1.0, head_prominence_ratio)
        clearance_score = min(1.0, min(left_clearance, right_clearance) / max(pattern_height, _EPS))
        fit = (
            0.30 * shoulder_score
            + 0.18 * neckline_score
            + 0.24 * head_score
            + 0.18 * time_score
            + 0.10 * clearance_score
        )

        metadata = {
            **_candidate_metadata(block),
            "pattern_height": pattern_height,
            "shoulder_error_ratio": shoulder_error_ratio,
            "shoulder_clearance_error_ratio": shoulder_error_ratio,
            "neckline_slope_height_ratio": neckline_slope_height_ratio,
            "time_asymmetry_ratio": asymmetry_ratio,
            "head_prominence_ratio": head_prominence_ratio,
            "left_shoulder_neckline_clearance_ratio": left_clearance / pattern_height,
            "right_shoulder_neckline_clearance_ratio": right_clearance / pattern_height,
            "neckline_slope_normalized": neckline_slope,
            "neckline_intercept_normalized": neckline_intercept,
            "expected_direction": expected_direction,
            "trade_signal": "neutral",
            "breakout_confirmed_by_availability": False,
            "breakout_direction_by_availability": None,
            "breakout_rule": "dynamic_neckline",
        }
        detections.append(_new_candidate(name, block, expected_direction, fit, metadata))

    return detections


def detect_head_and_shoulders(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _match_head_shoulders(extrema, config, inverse=False)


def detect_inverse_head_and_shoulders(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _match_head_shoulders(extrema, config, inverse=True)


def _detect_broadening(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig,
    start_with_high: bool,
) -> list[_PatternCandidate]:
    """Detect direction-neutral broadening geometry.

    Starting on a high versus a low is stored as phase information only; it does not imply
    a market top/bottom or a profitable direction.
    """

    detections: list[_PatternCandidate] = []
    expected = (
        ("max", "min", "max", "min", "max")
        if start_with_high
        else ("min", "max", "min", "max", "min")
    )

    for idx in range(len(extrema) - 4):
        block = tuple(extrema[idx : idx + 5])
        if tuple(e.kind for e in block) != expected:
            continue

        highs = [(e.index, e.value) for e in block if e.kind == "max"]
        lows = [(e.index, e.value) for e in block if e.kind == "min"]
        upper_slope, upper_intercept, upper_fit, upper_nrmse = _line_fit(highs)
        lower_slope, lower_intercept, lower_fit, lower_nrmse = _line_fit(lows)
        if upper_slope <= 0.0 or lower_slope >= 0.0:
            continue

        x_start = max(highs[0][0], lows[0][0])
        x_end = min(highs[-1][0], lows[-1][0])
        if x_end <= x_start:
            continue
        initial_width = _line_value(upper_slope, upper_intercept, x_start) - _line_value(
            lower_slope, lower_intercept, x_start
        )
        final_width = _line_value(upper_slope, upper_intercept, x_end) - _line_value(
            lower_slope, lower_intercept, x_end
        )
        if initial_width <= _EPS or final_width <= initial_width:
            continue
        expansion = final_width / initial_width - 1.0
        if expansion < config.broadening_min_expansion:
            continue

        expansion_score = min(1.0, expansion / max(config.broadening_min_expansion * 3.0, _EPS))
        residual_penalty = min(1.0, 0.5 * upper_nrmse + 0.5 * lower_nrmse)
        fit = 0.30 * upper_fit + 0.30 * lower_fit + 0.30 * expansion_score + 0.10 * (1.0 - residual_penalty)
        values = [e.value for e in block]
        metadata = {
            **_candidate_metadata(block),
            "pattern_height": max(values) - min(values),
            "upper_slope_normalized": upper_slope,
            "upper_intercept_normalized": upper_intercept,
            "lower_slope_normalized": lower_slope,
            "lower_intercept_normalized": lower_intercept,
            "boundary_expansion_ratio": expansion,
            "phase_start": "high" if start_with_high else "low",
            "expected_direction": "neutral",
            "trade_signal": "neutral",
            "breakout_confirmed_by_availability": False,
            "breakout_direction_by_availability": None,
            "breakout_rule": "expanding_channel",
        }
        detections.append(_new_candidate("broadening_formation", block, "neutral", fit, metadata))

    return detections


def detect_broadening_top(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _detect_broadening(extrema, config, start_with_high=True)


def detect_broadening_bottom(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _detect_broadening(extrema, config, start_with_high=False)


def _detect_rectangle(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig,
    start_with_high: bool,
) -> list[_PatternCandidate]:
    detections: list[_PatternCandidate] = []
    expected = (
        ("max", "min", "max", "min", "max")
        if start_with_high
        else ("min", "max", "min", "max", "min")
    )

    for idx in range(len(extrema) - 4):
        block = tuple(extrema[idx : idx + 5])
        if tuple(e.kind for e in block) != expected:
            continue
        highs = [e.value for e in block if e.kind == "max"]
        lows = [e.value for e in block if e.kind == "min"]
        resistance = float(np.mean(highs))
        support = float(np.mean(lows))
        height = resistance - support
        if height <= _EPS:
            continue

        high_scatter = max(highs) - min(highs)
        low_scatter = max(lows) - min(lows)
        high_ratio = high_scatter / height
        low_ratio = low_scatter / height
        if high_ratio > config.max_rectangle_boundary_error_ratio:
            continue
        if low_ratio > config.max_rectangle_boundary_error_ratio:
            continue

        high_score = 1.0 - min(1.0, high_ratio / max(config.max_rectangle_boundary_error_ratio, _EPS))
        low_score = 1.0 - min(1.0, low_ratio / max(config.max_rectangle_boundary_error_ratio, _EPS))
        fit = 0.5 * high_score + 0.5 * low_score
        metadata = {
            **_candidate_metadata(block),
            "pattern_height": height,
            "resistance_normalized": resistance,
            "support_normalized": support,
            "high_boundary_error_ratio": high_ratio,
            "low_boundary_error_ratio": low_ratio,
            "phase_start": "high" if start_with_high else "low",
            "expected_direction": "neutral",
            "trade_signal": "neutral",
            "breakout_confirmed_by_availability": False,
            "breakout_direction_by_availability": None,
            "breakout_rule": "horizontal_channel",
        }
        detections.append(_new_candidate("rectangle", block, "neutral", fit, metadata))

    return detections


def detect_rectangle_top(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _detect_rectangle(extrema, config, start_with_high=True)


def detect_rectangle_bottom(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _detect_rectangle(extrema, config, start_with_high=False)


def _detect_triangle(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig,
    start_with_high: bool,
) -> list[_PatternCandidate]:
    detections: list[_PatternCandidate] = []
    expected = (
        ("max", "min", "max", "min", "max")
        if start_with_high
        else ("min", "max", "min", "max", "min")
    )

    for idx in range(len(extrema) - 4):
        block = tuple(extrema[idx : idx + 5])
        if tuple(e.kind for e in block) != expected:
            continue
        highs = [(e.index, e.value) for e in block if e.kind == "max"]
        lows = [(e.index, e.value) for e in block if e.kind == "min"]
        upper_slope, upper_intercept, upper_fit, upper_nrmse = _line_fit(highs)
        lower_slope, lower_intercept, lower_fit, lower_nrmse = _line_fit(lows)
        if upper_slope >= 0.0 or lower_slope <= 0.0:
            continue

        x_start = max(highs[0][0], lows[0][0])
        x_end = min(highs[-1][0], lows[-1][0])
        if x_end <= x_start:
            continue
        initial_width = _line_value(upper_slope, upper_intercept, x_start) - _line_value(
            lower_slope, lower_intercept, x_start
        )
        final_width = _line_value(upper_slope, upper_intercept, x_end) - _line_value(
            lower_slope, lower_intercept, x_end
        )
        if initial_width <= _EPS or final_width <= 0.0 or final_width >= initial_width:
            continue

        convergence = 1.0 - final_width / initial_width
        if convergence < config.triangle_min_convergence:
            continue
        denominator = upper_slope - lower_slope
        if abs(denominator) <= _EPS:
            continue
        apex_x = (lower_intercept - upper_intercept) / denominator
        pattern_span = max(1, block[-1].index - block[0].index)
        if apex_x <= x_end:
            continue
        if apex_x - x_end > config.triangle_max_apex_distance_spans * pattern_span:
            continue

        convergence_score = min(1.0, convergence / max(config.triangle_min_convergence * 3.0, _EPS))
        apex_distance = (apex_x - x_end) / pattern_span
        apex_score = max(0.0, 1.0 - apex_distance / max(config.triangle_max_apex_distance_spans, _EPS))
        line_score = 0.5 * upper_fit + 0.5 * lower_fit
        residual_penalty = min(1.0, 0.5 * upper_nrmse + 0.5 * lower_nrmse)
        fit = 0.35 * line_score + 0.35 * convergence_score + 0.20 * apex_score + 0.10 * (1.0 - residual_penalty)

        values = [e.value for e in block]
        metadata = {
            **_candidate_metadata(block),
            "pattern_height": max(values) - min(values),
            "upper_slope_normalized": upper_slope,
            "upper_intercept_normalized": upper_intercept,
            "lower_slope_normalized": lower_slope,
            "lower_intercept_normalized": lower_intercept,
            "boundary_convergence_ratio": convergence,
            "projected_apex_local_index": float(apex_x),
            "phase_start": "high" if start_with_high else "low",
            "expected_direction": "neutral",
            "trade_signal": "neutral",
            "breakout_confirmed_by_availability": False,
            "breakout_direction_by_availability": None,
            "breakout_rule": "converging_channel",
        }
        detections.append(_new_candidate("symmetrical_triangle", block, "neutral", fit, metadata))

    return detections


def detect_triangle_top(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _detect_triangle(extrema, config, start_with_high=True)


def detect_triangle_bottom(
    extrema: Sequence[LocalExtremum], config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> list[_PatternCandidate]:
    return _detect_triangle(extrema, config, start_with_high=False)


def _find_double_candidates(
    extrema: Sequence[LocalExtremum],
    normalized_prices: Sequence[float],
    config: PatternConfig,
    is_top: bool,
    reference_window_size: int | None = None,
) -> list[_PatternCandidate]:
    """Detect only consecutive structural max-min-max / min-max-min doubles.

    A later same-kind pivot may not skip an intervening same-kind pivot.  More complicated
    retest structures belong to separate pattern families rather than being silently labelled
    as a double top/bottom.
    """

    detections: list[_PatternCandidate] = []
    # Spacing is a property of the configured analysis scale, not of the extra leading/trailing
    # bars used only to stabilize centered smoothing.  Direct helper calls may omit an explicit
    # scale, in which case the supplied price sequence remains the natural fallback.
    window_size = max(
        1,
        int(reference_window_size) if reference_window_size is not None else len(normalized_prices),
    )
    minimum_spacing = max(
        config.min_double_spacing_bars,
        int(math.ceil(config.min_double_spacing_fraction * window_size)),
    )
    maximum_spacing = max(
        minimum_spacing,
        int(math.floor(config.max_double_spacing_fraction * window_size)),
    )

    required_kind = "max" if is_top else "min"
    middle_kind = "min" if is_top else "max"

    for i in range(len(extrema) - 2):
        first, middle, second = extrema[i : i + 3]
        if (first.kind, middle.kind, second.kind) != (required_kind, middle_kind, required_kind):
            continue

        gap = second.index - first.index
        if gap < minimum_spacing or gap > maximum_spacing:
            continue

        if is_top:
            swing_height = min(first.value, second.value) - middle.value
            expected_direction = "bearish"
            name = "double_top"
        else:
            swing_height = middle.value - max(first.value, second.value)
            expected_direction = "bullish"
            name = "double_bottom"
        if swing_height <= _EPS:
            continue

        level_error = abs(first.value - second.value)
        level_error_ratio = level_error / swing_height
        if level_error_ratio > config.max_double_level_error_ratio:
            continue

        level_score = 1.0 - min(
            1.0,
            level_error_ratio / max(config.max_double_level_error_ratio, _EPS),
        )
        spacing_midpoint = 0.5 * (minimum_spacing + maximum_spacing)
        spacing_width = max(1.0, 0.5 * (maximum_spacing - minimum_spacing))
        spacing_score = math.exp(-0.5 * ((gap - spacing_midpoint) / spacing_width) ** 2)
        fit = 0.80 * level_score + 0.20 * spacing_score

        block = (first, middle, second)
        metadata = {
            **_candidate_metadata(block),
            "pattern_height": swing_height,
            "level_error_ratio": level_error_ratio,
            "spacing_bars": gap,
            "spacing_reference_window_bars": window_size,
            "neckline_level_normalized": middle.value,
            "consecutive_structural_pivots": True,
            "expected_direction": expected_direction,
            "trade_signal": "neutral",
            "breakout_confirmed_by_availability": False,
            "breakout_direction_by_availability": None,
            "breakout_rule": "fixed_neckline",
        }
        detections.append(_new_candidate(name, block, expected_direction, fit, metadata))

    return detections


def detect_double_top(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    prices: Sequence[float] | None = None,
    reference_window_size: int | None = None,
) -> list[_PatternCandidate]:
    normalized_prices: Sequence[float] = () if prices is None else prices
    return _find_double_candidates(
        extrema,
        normalized_prices,
        config,
        is_top=True,
        reference_window_size=reference_window_size,
    )


def detect_double_bottom(
    extrema: Sequence[LocalExtremum],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    prices: Sequence[float] | None = None,
    reference_window_size: int | None = None,
) -> list[_PatternCandidate]:
    normalized_prices: Sequence[float] = () if prices is None else prices
    return _find_double_candidates(
        extrema,
        normalized_prices,
        config,
        is_top=False,
        reference_window_size=reference_window_size,
    )


def _evaluate_breakout(candidate: _PatternCandidate, window: Sequence[float]) -> dict[str, Any]:
    """Evaluate breakout information available *by* the candidate's emission window.

    This is descriptive metadata only. The detector still emits ``trade_signal='neutral'`` and
    execution cannot occur before the next bar after ``available_at_index``.
    """

    metadata = candidate.metadata
    search_start = max(candidate.end_index + 1, candidate.local_shape_confirmation_index)
    empty = {
        "breakout_confirmed_by_availability": False,
        "breakout_direction_by_availability": None,
        "breakout_occurred_at_index_local": None,
        "breakout_level_normalized": None,
        "breakout_strength_pattern_height_ratio": None,
        "breakout_strength_volatility_units": None,
        "breakout_price_basis": "close",
    }
    if search_start >= len(window):
        return empty

    rule = metadata.get("breakout_rule")
    pattern_height = max(float(metadata.get("pattern_height", 0.0)), _EPS)
    apex = metadata.get("projected_apex_local_index")
    apex_value = float(apex) if isinstance(apex, (int, float, np.number)) and math.isfinite(float(apex)) else None

    for t in range(search_start, len(window)):
        # Once a symmetrical triangle's two fitted boundaries have met, the original triangle
        # no longer has a meaningful channel to break. A later crossing is a different event.
        if rule == "converging_channel" and apex_value is not None and float(t) + 1e-9 >= apex_value:
            break

        value = float(window[t])
        direction: str | None = None
        level: float | None = None

        if rule == "fixed_neckline":
            level = float(metadata["neckline_level_normalized"])
            if candidate.expected_direction == "bearish" and value < level:
                direction = "bearish"
            elif candidate.expected_direction == "bullish" and value > level:
                direction = "bullish"

        elif rule == "dynamic_neckline":
            slope = float(metadata["neckline_slope_normalized"])
            intercept = float(metadata["neckline_intercept_normalized"])
            level = _line_value(slope, intercept, t)
            if candidate.expected_direction == "bearish" and value < level:
                direction = "bearish"
            elif candidate.expected_direction == "bullish" and value > level:
                direction = "bullish"

        elif rule == "horizontal_channel":
            resistance = float(metadata["resistance_normalized"])
            support = float(metadata["support_normalized"])
            if value > resistance:
                direction = "bullish"
                level = resistance
            elif value < support:
                direction = "bearish"
                level = support

        elif rule in {"converging_channel", "expanding_channel"}:
            upper = _line_value(
                float(metadata["upper_slope_normalized"]),
                float(metadata["upper_intercept_normalized"]),
                t,
            )
            lower = _line_value(
                float(metadata["lower_slope_normalized"]),
                float(metadata["lower_intercept_normalized"]),
                t,
            )
            if upper <= lower:
                if rule == "converging_channel":
                    break
                continue
            if value > upper:
                direction = "bullish"
                level = upper
            elif value < lower:
                direction = "bearish"
                level = lower

        if direction is not None and level is not None:
            distance = value - level if direction == "bullish" else level - value
            local_start = max(0, t - 10)
            sigma = _robust_log_volatility(window[local_start : t + 1])
            return {
                "breakout_confirmed_by_availability": True,
                "breakout_direction_by_availability": direction,
                "breakout_occurred_at_index_local": t,
                "breakout_level_normalized": level,
                "breakout_strength_pattern_height_ratio": max(0.0, distance) / pattern_height,
                "breakout_strength_volatility_units": (
                    max(0.0, distance) / sigma if sigma > _EPS else None
                ),
                "breakout_price_basis": "close",
            }

    return empty


def _interval_overlap_ratio(a: PatternDetection, b: PatternDetection) -> float:
    left = max(a.start_index, b.start_index)
    right = min(a.end_index, b.end_index)
    if right < left:
        return 0.0
    intersection = right - left + 1
    shorter = min(a.end_index - a.start_index + 1, b.end_index - b.start_index + 1)
    return intersection / max(shorter, 1)


def _global_pivots(detection: PatternDetection) -> tuple[int, ...]:
    values = detection.metadata.get("pivot_geometry_indices_global", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(int(v) for v in values)


def _pivot_kinds(detection: PatternDetection) -> tuple[str, ...]:
    values = detection.metadata.get("pivot_kinds", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(str(v) for v in values)


_PHASE_INVARIANT_NEUTRAL_PATTERNS = frozenset(
    {"rectangle", "symmetrical_triangle", "broadening_formation"}
)


def _pivot_match_fraction(a: PatternDetection, b: PatternDetection, tolerance: int) -> float:
    """Position-by-position pivot agreement for phase-sensitive pattern families."""

    a_pivots = _global_pivots(a)
    b_pivots = _global_pivots(b)
    if not a_pivots or not b_pivots or len(a_pivots) != len(b_pivots):
        return 0.0
    a_kinds = _pivot_kinds(a)
    b_kinds = _pivot_kinds(b)
    if a_kinds and b_kinds and a_kinds != b_kinds:
        return 0.0
    matches = sum(abs(x - y) <= tolerance for x, y in zip(a_pivots, b_pivots))
    return matches / len(a_pivots)


def _phase_invariant_pivot_match_fraction(
    a: PatternDetection,
    b: PatternDetection,
    tolerance: int,
) -> float:
    """Order-preserving shared-pivot fraction for neutral channel-like formations.

    A continuing H-L-H-L-H triangle followed one pivot later by L-H-L-H-L is one evolving
    structure if four of five actual extrema are shared. A longest-common-subsequence match
    captures that extension without allowing arbitrary unordered pivot matching.
    """

    a_pivots = _global_pivots(a)
    b_pivots = _global_pivots(b)
    a_kinds = _pivot_kinds(a)
    b_kinds = _pivot_kinds(b)
    if not a_pivots or not b_pivots:
        return 0.0
    if len(a_pivots) != len(a_kinds) or len(b_pivots) != len(b_kinds):
        return 0.0

    rows = len(a_pivots) + 1
    cols = len(b_pivots) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            same_pivot = (
                a_kinds[i - 1] == b_kinds[j - 1]
                and abs(a_pivots[i - 1] - b_pivots[j - 1]) <= tolerance
            )
            if same_pivot:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(1, min(len(a_pivots), len(b_pivots)))


def _structural_similarity(a: PatternDetection, b: PatternDetection, config: PatternConfig) -> float:
    if a.pattern_name != b.pattern_name:
        return 0.0

    overlap = _interval_overlap_ratio(a, b)
    if overlap < config.cluster_min_interval_overlap:
        return 0.0

    if a.pattern_name in _PHASE_INVARIANT_NEUTRAL_PATTERNS:
        pivot_fraction = _phase_invariant_pivot_match_fraction(
            a, b, config.cluster_pivot_tolerance_bars
        )
        required_pivot_fraction = config.cluster_neutral_min_shared_pivot_fraction
    else:
        pivot_fraction = _pivot_match_fraction(a, b, config.cluster_pivot_tolerance_bars)
        required_pivot_fraction = config.cluster_min_pivot_match_fraction
    if pivot_fraction < required_pivot_fraction:
        return 0.0

    a_center = 0.5 * (a.start_index + a.end_index)
    b_center = 0.5 * (b.start_index + b.end_index)
    shorter_span = max(1, min(a.end_index - a.start_index, b.end_index - b.start_index))
    center_score = max(
        0.0,
        1.0
        - abs(a_center - b_center)
        / max(config.cluster_center_tolerance_fraction * shorter_span, 1.0),
    )
    # Every member still has to match the immutable causal anchor directly, preventing chain
    # drift. Center agreement is a tie-breaker rather than a hard condition so neutral pattern
    # extensions that share four of five pivots can remain one event.
    return 0.45 * overlap + 0.45 * pivot_fraction + 0.10 * center_score


def _add_global_context_features(
    metadata: dict[str, Any],
    normalized_global: np.ndarray,
    market: _MarketSeries,
    abs_start: int,
    abs_end: int,
    config: PatternConfig,
) -> dict[str, Any]:
    pre_start = max(0, abs_start - config.context_lookback_bars)
    pre = normalized_global[pre_start:abs_start]
    if pre.size >= 2:
        pre_return = float(pre[-1] - pre[0])
        pre_slope = _line_fit([(i, float(value)) for i, value in enumerate(pre)])[0]
        pre_vol = _local_log_volatility(pre.tolist())
    else:
        pre_return = 0.0
        pre_slope = 0.0
        pre_vol = 0.0

    pattern = normalized_global[abs_start : abs_end + 1]
    updates: dict[str, Any] = {
        **metadata,
        "pre_pattern_log_return": pre_return,
        "pre_pattern_trend_slope": pre_slope,
        "pre_pattern_log_volatility": pre_vol,
        "pre_pattern_robust_log_volatility": _robust_log_volatility(pre.tolist()),
        "pattern_log_volatility": _local_log_volatility(pattern.tolist()),
        "pattern_robust_log_volatility": _robust_log_volatility(pattern.tolist()),
        "pattern_span_bars": abs_end - abs_start + 1,
        "pattern_span_intervals": abs_end - abs_start,
        "context_start_index": pre_start if pre.size else None,
        "context_end_index": abs_start - 1 if pre.size else None,
        "context_bar_count": int(pre.size),
    }

    if market.volume is not None:
        pre_volume = market.volume[pre_start:abs_start]
        pattern_volume = market.volume[abs_start : abs_end + 1]
        pre_valid = pre_volume[np.isfinite(pre_volume)]
        pattern_valid = pattern_volume[np.isfinite(pattern_volume)]
        pre_mean = _stable_nonnegative_mean(pre_valid) if pre_valid.size else None
        pattern_mean = _stable_nonnegative_mean(pattern_valid) if pattern_valid.size else None
        if pre_mean is not None:
            updates["pre_pattern_mean_volume"] = pre_mean
        if pattern_mean is not None:
            updates["pattern_mean_volume"] = pattern_mean
        if pre_mean is not None and pattern_mean is not None and pre_mean > 0.0:
            if pattern_mean == 0.0:
                updates["pattern_to_pre_volume_ratio"] = 0.0
            else:
                log_ratio = math.log(pattern_mean) - math.log(pre_mean)
                if math.isfinite(log_ratio):
                    updates["pattern_to_pre_log_volume_ratio"] = log_ratio
                    if log_ratio <= math.log(float.fromhex("0x1.fffffffffffffp+1023")):
                        ratio = math.exp(log_ratio)
                        if math.isfinite(ratio):
                            updates["pattern_to_pre_volume_ratio"] = ratio

    if market.high is not None and market.low is not None:
        high = market.high[abs_start : abs_end + 1]
        low = market.low[abs_start : abs_end + 1]
        valid = np.isfinite(high) & np.isfinite(low) & (high > 0.0) & (low > 0.0)
        if np.any(valid):
            # Difference-of-logs avoids overflow when high/low is extreme but each quote is finite.
            log_ranges = np.log(high[valid]) - np.log(low[valid])
            updates["pattern_mean_log_high_low_range"] = float(np.mean(log_ranges))

    return updates


def _add_raw_pivot_metadata(
    metadata: Mapping[str, Any],
    market: _MarketSeries,
    window_start: int,
    abs_start: int,
    abs_end: int,
    config: PatternConfig,
) -> dict[str, Any]:
    geometry_local = metadata.get("pivot_geometry_indices_local", ())
    raw_local = metadata.get("pivot_raw_indices_local", ())
    kinds = metadata.get("pivot_kinds", ())
    if not isinstance(geometry_local, Sequence) or isinstance(geometry_local, (str, bytes)):
        return dict(metadata)
    if not isinstance(raw_local, Sequence) or isinstance(raw_local, (str, bytes)):
        raw_local = geometry_local

    geometry_global = [window_start + int(i) for i in geometry_local]
    close_refined_global = [
        min(max(window_start + int(i), abs_start), abs_end)
        for i in raw_local
    ]
    raw_global: list[int] = []
    radius = max(0, int(config.pivot_refine_radius))

    # Refine annotation/reference pivots with OHLC when available, while leaving the smoothed
    # geometry untouched.  Restrict refinement to the declared pattern interval so a context
    # candle cannot become the public first/last pivot of the event.
    for pos, geometry_index in enumerate(geometry_global):
        kind = kinds[pos] if isinstance(kinds, Sequence) and not isinstance(kinds, (str, bytes)) and pos < len(kinds) else None
        left = max(abs_start, geometry_index - radius)
        right = min(abs_end, geometry_index + radius)
        indices = list(range(left, right + 1))
        if not indices:
            raw_global.append(geometry_index)
            continue

        if kind == "max" and market.high is not None:
            valid = [i for i in indices if math.isfinite(float(market.high[i]))]
            if valid:
                raw_global.append(max(valid, key=lambda i: float(market.high[i])))
                continue
        if kind == "min" and market.low is not None:
            valid = [i for i in indices if math.isfinite(float(market.low[i]))]
            if valid:
                raw_global.append(min(valid, key=lambda i: float(market.low[i])))
                continue

        if kind == "max":
            raw_global.append(max(indices, key=lambda i: float(market.close[i])))
        elif kind == "min":
            raw_global.append(min(indices, key=lambda i: float(market.close[i])))
        else:
            raw_global.append(min(max(geometry_index, abs_start), abs_end))

    close_prices = [float(market.close[index]) for index in raw_global]
    close_refined_prices = [float(market.close[index]) for index in close_refined_global]
    refined_prices: list[float] = []
    highs: list[float | None] = []
    lows: list[float | None] = []
    labels: list[Any | None] = []

    for pos, global_index in enumerate(raw_global):
        kind = kinds[pos] if isinstance(kinds, Sequence) and not isinstance(kinds, (str, bytes)) and pos < len(kinds) else None
        high_value = None
        low_value = None
        if market.high is not None and math.isfinite(float(market.high[global_index])):
            high_value = float(market.high[global_index])
        if market.low is not None and math.isfinite(float(market.low[global_index])):
            low_value = float(market.low[global_index])
        highs.append(high_value)
        lows.append(low_value)
        if kind == "max" and high_value is not None:
            refined_prices.append(high_value)
        elif kind == "min" and low_value is not None:
            refined_prices.append(low_value)
        else:
            refined_prices.append(float(market.close[global_index]))
        labels.append(market.labels[global_index] if global_index < len(market.labels) else None)

    return {
        **dict(metadata),
        "pivot_geometry_indices_global": geometry_global,
        # Compatibility alias: these are the OHLC/wick-refined annotation indices.
        "pivot_raw_indices_global": raw_global,
        "pivot_ohlc_refined_indices_global": raw_global,
        "pivot_close_refined_indices_global": close_refined_global,
        "pivot_close_prices_raw": close_prices,
        "pivot_close_prices_at_ohlc_refined_indices": close_prices,
        "pivot_close_refined_prices_raw": close_refined_prices,
        "pivot_prices_raw_refined": refined_prices,
        "pivot_high_prices_raw": highs,
        "pivot_low_prices_raw": lows,
        "pivot_labels": labels,
    }


def _add_global_geometry_coordinates(
    metadata: Mapping[str, Any],
    window_start: int,
) -> dict[str, Any]:
    """Add global-index equivalents for line/apex metadata defined in local window coordinates."""

    updates = dict(metadata)
    if "neckline_slope_normalized" in metadata and "neckline_intercept_normalized" in metadata:
        slope = float(metadata["neckline_slope_normalized"])
        intercept_local = float(metadata["neckline_intercept_normalized"])
        updates["neckline_intercept_global_normalized"] = intercept_local - slope * window_start
        updates["neckline_coordinate_system"] = "global_bar_index"

    for prefix in ("upper", "lower"):
        slope_key = f"{prefix}_slope_normalized"
        intercept_key = f"{prefix}_intercept_normalized"
        if slope_key in metadata and intercept_key in metadata:
            slope = float(metadata[slope_key])
            intercept_local = float(metadata[intercept_key])
            updates[f"{prefix}_intercept_global_normalized"] = intercept_local - slope * window_start
            updates[f"{prefix}_coordinate_system"] = "global_bar_index"

    apex_local = metadata.get("projected_apex_local_index")
    if isinstance(apex_local, (int, float, np.number)) and math.isfinite(float(apex_local)):
        updates["projected_apex_global_index"] = window_start + float(apex_local)

    return updates


def _public_candidate(
    candidate: _PatternCandidate,
    market: _MarketSeries,
    normalized_global: np.ndarray,
    window: Sequence[float],
    window_start: int,
    core_window_length: int,
    observation_length: int,
    leading_context: int,
    trailing_context: int,
    bandwidth: float | Sequence[float],
    config: PatternConfig,
    config_hash: str,
    causal_prefix_fingerprint: str,
    causal_market_data_prefix_fingerprint: str,
    run_context_hash: str,
    run_context: RunContext | None,
    observation_noise_scale: float = 0.0,
) -> PatternDetection | None:
    local_start = candidate.start_index
    local_end = candidate.end_index
    pattern_bar_count = local_end - local_start + 1
    if pattern_bar_count < config.min_pattern_span_bars:
        return None

    core_start_local = leading_context
    core_end_local = leading_context + core_window_length - 1
    # The geometric pattern must lie inside the core region; surrounding bars exist only to
    # stabilize the centered smoother at both boundaries.
    if local_start < core_start_local or local_end > core_end_local:
        return None
    allowed_pattern_age = config.resolved_max_pattern_age_bars(core_window_length)
    if core_end_local - local_end > allowed_pattern_age:
        return None

    abs_start = window_start + local_start
    abs_end = window_start + local_end
    abs_available = window_start + observation_length - 1
    earliest_execution = abs_available + int(config.execution_delay_bars)
    if isinstance(bandwidth, (int, float, np.number)):
        resolved_bandwidths = (float(bandwidth),)
    else:
        resolved_bandwidths = tuple(float(v) for v in bandwidth)
    if not resolved_bandwidths or any(not math.isfinite(v) or v <= 0.0 for v in resolved_bandwidths):
        raise ValueError("bandwidth must contain at least one finite positive value")

    pattern_height = float(candidate.metadata.get("pattern_height", 0.0))
    height_threshold = _adaptive_height_threshold(observation_noise_scale, config)
    if pattern_height < height_threshold:
        return None

    breakout_updates = _evaluate_breakout(candidate, window)
    if (
        config.require_double_neckline_confirmation
        and candidate.pattern_name in {"double_top", "double_bottom"}
        and not bool(breakout_updates.get("breakout_confirmed_by_availability", False))
    ):
        return None

    metadata = {**dict(candidate.metadata), **breakout_updates}
    metadata = _add_raw_pivot_metadata(
        metadata,
        market,
        window_start,
        abs_start,
        abs_end,
        config,
    )
    metadata = _add_global_context_features(
        metadata,
        normalized_global,
        market,
        abs_start,
        abs_end,
        config,
    )
    metadata = _add_global_geometry_coordinates(metadata, window_start)

    breakout_local = metadata.get("breakout_occurred_at_index_local")
    breakout_global = window_start + int(breakout_local) if isinstance(breakout_local, int) else None
    metadata.update(
        {
            "detector_version": DETECTOR_VERSION,
            "config_hash": config_hash,
            "research_profile_name": config.research_profile_name,
            "window_start": window_start,
            "window_end": window_start + observation_length - 1,
            "core_window_length": core_window_length,
            "observation_window_length": observation_length,
            "leading_smoothing_context_bars": leading_context,
            "trailing_smoothing_context_bars": trailing_context,
            "core_region_start_index": window_start + core_start_local,
            "core_region_end_index": window_start + core_end_local,
            "pattern_end_index": abs_end,
            "pattern_bar_count": pattern_bar_count,
            "available_at_index": abs_available,
            "information_available_at_index": abs_available,
            "earliest_execution_index": earliest_execution,
            "earliest_actionable_index": earliest_execution,
            "execution_delay_bars": int(config.execution_delay_bars),
            "execution_semantics": "decision_after_available_bar_close; execute_after_configured_bar_delay",
            "causal_detection_time": abs_available,
            "shape_end_index": abs_end,
            "pattern_age_at_core_end_bars": core_end_local - local_end,
            "pattern_age_at_availability_bars": abs_available - abs_end,
            "max_pattern_age_allowed_bars": allowed_pattern_age,
            "max_pattern_age_fraction": float(config.max_pattern_age_fraction),
            "max_pattern_age_semantics": "scale_relative_core_end_minus_shape_end_before_trailing_smoothing_context",
            "observation_residual_noise_scale": float(observation_noise_scale),
            "pattern_height_threshold": float(height_threshold),
            "threshold_noise_semantics": "robust_MAD_of_log_price_minus_smoothed_structure",
            "bandwidth": float(np.median(np.asarray(resolved_bandwidths, dtype=float))),
            "smoothing_bandwidths": resolved_bandwidths,
            "bandwidth_method": "predefined_scale_fraction_local_linear_ensemble_median",
            "smoothing_method": "gaussian_local_linear_regression",
            "smoothing_calibration_id": SMOOTHING_CALIBRATION_ID,
            "geometry_calibration_id": GEOMETRY_CALIBRATION_ID,
            "breakout_occurred_at_index": breakout_global,
            "breakout_timestamp_is_descriptive_only": True,
            "normalized_start_value": float(window[local_start]),
            "normalized_end_value": float(window[local_end]),
            "raw_start_price": float(market.close[abs_start]),
            "raw_end_price": float(market.close[abs_end]),
            "expected_direction": candidate.expected_direction,
            "trade_signal": "neutral",
            "score_semantics": "pattern_specific_geometry_fit",
            "causal_prefix_fingerprint": causal_prefix_fingerprint,
            "causal_market_data_prefix_fingerprint": causal_market_data_prefix_fingerprint,
            "run_context_hash": run_context_hash,
            "run_context": run_context.stable_payload() if run_context is not None else None,
        }
    )

    return PatternDetection(
        pattern_name=candidate.pattern_name,
        start_index=abs_start,
        end_index=abs_end,
        expected_direction=candidate.expected_direction,
        geometry_fit_score=candidate.geometry_fit_score,
        metadata=metadata,
        available_at_index=abs_available,
        earliest_execution_index=earliest_execution,
        event_id=None,
    )


def _stable_event_id(primary: PatternDetection) -> str:
    """Causal stable id; future bars cannot change it.

    ``run_context_hash`` makes IDs globally distinct across named experiments/assets when a
    RunContext is supplied. The causal prefix fingerprint prevents dependence on future bars.
    """

    payload = {
        "pattern_name": primary.pattern_name,
        "start_index": primary.start_index,
        "end_index": primary.end_index,
        "available_at_index": primary.available_at_index,
        "pivots": list(_global_pivots(primary)),
        "pivot_kinds": list(_pivot_kinds(primary)),
        "detector_version": primary.metadata.get("detector_version", DETECTOR_VERSION),
        "config_hash": primary.metadata.get("config_hash"),
        "causal_prefix_fingerprint": primary.metadata.get("causal_prefix_fingerprint"),
        "run_context_hash": primary.metadata.get("run_context_hash"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "EV-" + hashlib.sha256(encoded).hexdigest()[:18]


def _cluster_candidates_causally(
    candidates: Sequence[PatternDetection],
    config: PatternConfig,
) -> tuple[tuple[PatternDetection, ...], tuple[PatternDetection, ...], tuple[PatternEventCluster, ...]]:
    """Assign raw candidates to events in chronological order without rewriting history.

    Primary event metadata is never augmented with support that arrived later. The separate
    PatternEventCluster object is explicitly post-hoc and is safe for research summaries but
    not for strategy-time features.
    """

    ordered = sorted(
        candidates,
        key=lambda d: (
            d.available_at_index,
            d.pattern_name,
            -d.geometry_fit_score,
            d.start_index,
            d.end_index,
        ),
    )
    clusters: list[_ClusterState] = []
    clusters_by_pattern: dict[str, list[_ClusterState]] = {}
    assigned_candidates: list[PatternDetection] = []
    primary_events: list[PatternDetection] = []
    for available_at, time_group_iter in groupby(ordered, key=lambda d: d.available_at_index):
        time_group = list(time_group_iter)
        new_primary_indices: list[int] = []

        for candidate in time_group:
            best_cluster: _ClusterState | None = None
            best_similarity = 0.0
            # Compare only with the immutable causal primary anchor.  Never compare against
            # arbitrary later members: A~B and B~C must not make C part of A when A!~C.
            for cluster in clusters_by_pattern.get(candidate.pattern_name, []):
                if cluster.primary.end_index < candidate.start_index:
                    continue
                similarity = _structural_similarity(cluster.primary, candidate, config)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if best_cluster is not None and best_similarity > 0.0:
                phase_shifted = (
                    candidate.pattern_name in _PHASE_INVARIANT_NEUTRAL_PATTERNS
                    and _pivot_kinds(candidate) != _pivot_kinds(best_cluster.primary)
                )
                assigned = replace(
                    candidate,
                    event_id=best_cluster.event_id,
                    metadata={
                        **candidate.metadata,
                        "cluster_anchor_similarity": best_similarity,
                        "cluster_relation": "phase_shift_extension" if phase_shifted else "same_structure_view",
                    },
                )
                best_cluster.members.append(assigned)
                assigned_candidates.append(assigned)
                continue

            event_id = _stable_event_id(candidate)
            assigned = replace(candidate, event_id=event_id)
            cluster = _ClusterState(
                event_id=event_id,
                pattern_name=assigned.pattern_name,
                primary=assigned,
                members=[assigned],
            )
            clusters.append(cluster)
            clusters_by_pattern.setdefault(assigned.pattern_name, []).append(cluster)
            assigned_candidates.append(assigned)
            primary_events.append(assigned)
            new_primary_indices.append(len(primary_events) - 1)

        # Competition is causal. Earlier events are never retroactively modified. Events first
        # available at the same timestamp may know about each other because both are observable.
        for index in new_primary_indices:
            event = primary_events[index]
            competitors: list[PatternDetection] = []
            for other in primary_events:
                if other.event_id == event.event_id or other.pattern_name == event.pattern_name:
                    continue
                if other.available_at_index > available_at:
                    continue
                if _interval_overlap_ratio(other, event) >= config.competition_overlap_threshold:
                    competitors.append(other)
            if competitors:
                primary_events[index] = replace(
                    event,
                    metadata={
                        **event.metadata,
                        "competing_pattern": True,
                        "known_competing_event_ids": sorted({p.event_id for p in competitors if p.event_id}),
                        "known_competing_patterns": sorted({p.pattern_name for p in competitors}),
                    },
                )
                # Replace the cluster primary and the corresponding first raw member so the
                # exact same causal primary object is exposed in all public views.
                for cluster in clusters:
                    if cluster.event_id == event.event_id:
                        cluster.primary = primary_events[index]
                        cluster.members[0] = primary_events[index]
                        break
                for cand_index, raw in enumerate(assigned_candidates):
                    if raw.event_id == event.event_id and raw.available_at_index == event.available_at_index and raw.start_index == event.start_index and raw.end_index == event.end_index:
                        assigned_candidates[cand_index] = primary_events[index]
                        break
            else:
                primary_events[index] = replace(
                    event,
                    metadata={
                        **event.metadata,
                        "competing_pattern": False,
                        "known_competing_event_ids": [],
                        "known_competing_patterns": [],
                    },
                )
                for cluster in clusters:
                    if cluster.event_id == event.event_id:
                        cluster.primary = primary_events[index]
                        cluster.members[0] = primary_events[index]
                        break
                for cand_index, raw in enumerate(assigned_candidates):
                    if raw.event_id == event.event_id and raw.available_at_index == event.available_at_index and raw.start_index == event.start_index and raw.end_index == event.end_index:
                        assigned_candidates[cand_index] = primary_events[index]
                        break

    research_clusters: list[PatternEventCluster] = []
    for cluster in clusters:
        members = tuple(
            sorted(
                cluster.members,
                key=lambda d: (d.available_at_index, d.start_index, d.end_index, -d.geometry_fit_score),
            )
        )
        windows = tuple(
            sorted(
                {
                    int(member.metadata["core_window_length"])
                    for member in members
                    if isinstance(member.metadata.get("core_window_length"), int)
                }
            )
        )
        research_clusters.append(
            PatternEventCluster(
                event_id=cluster.event_id,
                pattern_name=cluster.pattern_name,
                primary_detection=cluster.primary,
                candidates=members,
                supporting_core_windows=windows,
                first_available_at_index=min(m.available_at_index for m in members),
                last_available_at_index=max(m.available_at_index for m in members),
            )
        )

    candidate_sort_key = lambda d: (
        d.available_at_index,
        d.pattern_name,
        d.start_index,
        d.end_index,
        -d.geometry_fit_score,
        d.event_id or "",
    )
    cluster_sort_key = lambda c: (
        c.first_available_at_index,
        c.pattern_name,
        c.primary_detection.start_index,
        c.primary_detection.end_index,
        c.event_id,
    )
    return (
        tuple(sorted(assigned_candidates, key=candidate_sort_key)),
        tuple(sorted(primary_events, key=candidate_sort_key)),
        tuple(sorted(research_clusters, key=cluster_sort_key)),
    )


def detect_pattern_universe(
    candles: Sequence[CandleLike | float | int],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    run_context: RunContext | None = None,
) -> DetectionResult:
    """Detect raw multi-scale candidates and unique causal chart-pattern events.

    Research guarantees:
    - finite, strictly positive prices are required; invalid data fails loudly;
    - Python lists, NumPy arrays and independent candle objects are accepted;
    - geometry is multiplicatively price-scale invariant;
    - geometry is evaluated on the smoothed curve, while raw OHLC remains annotation data;
    - no bar after ``available_at_index`` contributes to a candidate;
    - descriptive breakout timestamps never permit back-dated execution;
    - raw multi-scale detections are preserved;
    - clustering is causal and every member must match the immutable primary anchor;
    - doubles use consecutive structural pivots only;
    - pattern geometry is separated from trade execution and profitability;
    - context features are computed from global past history;
    - optional RunContext is incorporated into stable event identity.

    Corporate-action adjustment and missing-bar policy remain data-loader responsibilities.
    """

    market = _extract_market_series(candles)
    config_hash = _config_hash(config)
    context_hash = _run_context_hash(run_context)
    if market.close.size == 0:
        return DetectionResult(
            candidates=(),
            events=(),
            clusters=(),
            detector_version=DETECTOR_VERSION,
            config_hash=config_hash,
            close_series_fingerprint=_close_series_fingerprint(market.close),
            full_market_data_fingerprint=_full_market_data_fingerprint(market),
            geometry_fingerprint=_geometry_fingerprint(np.asarray([], dtype=float)),
            run_context=run_context,
            config_snapshot=config,
        )

    normalized_global = _relative_log_series(market.close)
    prefix_fingerprints = _causal_prefix_fingerprints(normalized_global)
    market_prefix_fingerprints = _causal_market_prefix_fingerprints(market)
    close_fingerprint = _close_series_fingerprint(market.close)
    full_market_fingerprint = _full_market_data_fingerprint(market)
    geometry_fingerprint = _geometry_fingerprint(normalized_global)
    raw_candidates: list[PatternDetection] = []

    detector_sequence = (
        detect_head_and_shoulders,
        detect_inverse_head_and_shoulders,
        detect_broadening_top,
        detect_broadening_bottom,
        detect_rectangle_top,
        detect_rectangle_bottom,
        detect_triangle_top,
        detect_triangle_bottom,
        detect_double_top,
        detect_double_bottom,
    )

    for core_window_length in config.resolved_core_window_lengths():
        actual_bandwidths = config.smoothing_bandwidths(core_window_length)
        leading_context = config.effective_leading_context(core_window_length)
        trailing_context = config.effective_trailing_context(core_window_length)
        observation_length = leading_context + core_window_length + trailing_context
        if observation_length > len(normalized_global):
            continue

        for window_start in range(0, len(normalized_global) - observation_length + 1):
            window = normalized_global[window_start : window_start + observation_length].tolist()
            smoothed = _ensemble_local_linear_smoothing(window, actual_bandwidths)
            observation_noise_scale = _robust_residual_noise(window, smoothed)
            extrema = _find_local_extrema(smoothed, window, config)

            for detector in detector_sequence:
                required = config.detector_required_extrema(detector.__name__)
                if len(extrema) < required:
                    continue
                if detector in (detect_double_top, detect_double_bottom):
                    local_candidates = detector(
                        extrema,
                        config,
                        prices=window,
                        reference_window_size=core_window_length,
                    )
                else:
                    local_candidates = detector(extrema, config)

                for candidate in local_candidates:
                    abs_available = window_start + observation_length - 1
                    public = _public_candidate(
                        candidate=candidate,
                        market=market,
                        normalized_global=normalized_global,
                        window=window,
                        window_start=window_start,
                        core_window_length=core_window_length,
                        observation_length=observation_length,
                        leading_context=leading_context,
                        trailing_context=trailing_context,
                        bandwidth=actual_bandwidths,
                        observation_noise_scale=observation_noise_scale,
                        config=config,
                        config_hash=config_hash,
                        causal_prefix_fingerprint=prefix_fingerprints[abs_available],
                        causal_market_data_prefix_fingerprint=market_prefix_fingerprints[abs_available],
                        run_context_hash=context_hash,
                        run_context=run_context,
                    )
                    if public is not None:
                        raw_candidates.append(public)

    assigned, events, clusters = _cluster_candidates_causally(raw_candidates, config)
    return DetectionResult(
        candidates=assigned,
        events=events,
        clusters=clusters,
        detector_version=DETECTOR_VERSION,
        config_hash=config_hash,
        close_series_fingerprint=close_fingerprint,
        full_market_data_fingerprint=full_market_fingerprint,
        geometry_fingerprint=geometry_fingerprint,
        run_context=run_context,
        config_snapshot=config,
    )


def detect_pattern_candidates(
    candles: Sequence[CandleLike | float | int],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    run_context: RunContext | None = None,
) -> list[PatternDetection]:
    """Return every raw multi-scale candidate, including multiple views of one event."""

    return list(detect_pattern_universe(candles, config, run_context).candidates)


def detect_patterns(
    candles: Sequence[CandleLike | float | int],
    config: PatternConfig = DEFAULT_PATTERN_CONFIG,
    run_context: RunContext | None = None,
) -> list[PatternDetection]:
    """Return unique causal pattern events for backtesting/strategy research."""

    return list(detect_pattern_universe(candles, config, run_context).events)


if __name__ == "__main__":
    import random

    rng = random.Random(42)
    price_path = [100.0]
    for _ in range(400):
        price_path.append(price_path[-1] * math.exp(rng.gauss(0.0, 0.006)))

    result = detect_pattern_universe(price_path)
    print(
        f"detector={result.detector_version} config={result.config_hash} "
        f"raw_candidates={len(result.candidates)} unique_events={len(result.events)}"
    )
    for event in result.events[:15]:
        print(
            event.event_id,
            event.pattern_name,
            event.start_index,
            event.end_index,
            "available=",
            event.available_at_index,
            "expected=",
            event.expected_direction,
            "fit=",
            round(event.geometry_fit_score, 4),
            "breakout=",
            event.metadata.get("breakout_direction_by_availability"),
            "competing=",
            event.metadata.get("competing_pattern", False),
        )
