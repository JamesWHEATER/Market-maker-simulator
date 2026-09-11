"""Machine-learning research layer for conditional chart-pattern predictability.

This module shares the frozen classical research implementation but runs a separate,
sampled mechanism-discovery experiment. Its headline rates must not be compared directly
with the statistical runner's exhaustive, equal-weight scenario grid::

    market_maker_simulator.py
        -> chart_renderer.py
        -> pattern_detector.py
        -> ai_pattern_research_pipeline.py

It does not contain a second synthetic market or a second pattern detector.  Every AI observation
comes from the same ``SyntheticMarketConfig`` / ``simulate_market`` engine, the same candle builder,
and the same unique causal ``DetectionResult.confirmed_breakout_events`` used by the statistical
experiment runner.

Research question
-----------------
Estimate, out of sample::

    P(pattern is profitable after costs |
      pattern geometry, observable market state, synthetic market structure)

Two feature sets are trained separately:

``observable``
    Pattern geometry and causal market measurements available no later than the detector's
    information timestamp.  These are the closest bridge to later real-market work.

``oracle``
    The observable features plus the true synthetic mechanism weights, simulator parameters,
    and latent synthetic state.  This model is a mechanism-discovery instrument: it asks which
    controlled market conditions are associated with pattern edge.

Critical safeguards
-------------------
* Only ``DetectionResult.confirmed_breakout_events`` are training observations; raw multi-scale
  candidates are not treated as independent patterns.
* ``available_at_index`` and ``earliest_execution_index`` come from the detector contract.
* Future prices are used only for outcome/return fields, never as model inputs.
* Execution prices and transaction-cost semantics are shared with ``market_structure_experiment``.
* Entire synthetic *structures* (all of their repeated seeds) are assigned to exactly one of
  train/validation/test.  This is stronger than randomly splitting neighboring pattern rows.
* The validation partition selects the AI trading threshold; the test partition is untouched
  until final evaluation.
* Economic evaluation enforces one open position per run and reports an all-pattern baseline.
* The oracle XGBoost model exports SHAP summaries plus feature-response tables; those are
  predictive/associational explanations, not automatic proof of causality.

The implementation is designed for hundreds of thousands of pattern occurrences.  Simulation and
pattern detection can run in independent processes, expensive per-step JSON diagnostics are disabled
for research sweeps, and detector smoothing is vectorized/batched in ``pattern_detector.py``.
Actual wall-clock capacity still depends on path length, detector density, CPU count, RAM, and model
settings, so no finite program can guarantee a particular sample count on arbitrary hardware.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import warnings
import sqlite3
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from scipy import stats

from chart_renderer import CANDLE_BUILDER_VERSION, PriceCandle, build_candles
from market_maker_simulator import (
    RNG_STREAM_VERSION,
    SIMULATOR_VERSION,
    WORLD_NAMES,
    SyntheticMarketConfig,
    canonical_behavior_config,
    candle_dataset_id,
    run_id as simulator_run_id,
    scenario_id as simulator_scenario_id,
    simulate_market,
)
from market_structure_experiment import (
    EXPERIMENT_RUNNER_VERSION,
    CostModel,
    _procedure_from_manifest,
    net_trade_return,
    procedure_hash,
    synthetic_execution_arrays,
    trade_direction,
)
from pattern_detector import (
    DEFAULT_PATTERN_CONFIG,
    DETECTOR_VERSION,
    PatternDetection,
    RunContext,
    causal_signal_priority,
    detect_pattern_universe,
)


AI_PIPELINE_VERSION = "2.5.0"
_EPS = 1e-12
STRUCTURED_WORLDS = tuple(name for name in WORLD_NAMES if name != "random")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorldSamplingRanges:
    """Pre-registered ranges for synthetic mechanism discovery.

    These are deliberately broad *experimental* ranges, not empirical estimates.  Each sampled
    structure is held fixed across its repeated seeds, allowing the AI to learn cross-structure
    relationships while the test split can hold out complete structures.
    """

    fundamental_vol_fraction: tuple[float, float] = (0.00015, 0.00120)
    base_spread_fraction: tuple[float, float] = (0.00010, 0.00200)
    base_arrival_rate: tuple[float, float] = (0.75, 4.00)
    order_impact_fraction: tuple[float, float] = (0.00008, 0.00080)
    decision_signal_strength: tuple[float, float] = (1.20, 3.20)
    decision_noise: tuple[float, float] = (0.25, 1.25)
    fundamental_anchor_strength: tuple[float, float] = (0.005, 0.080)
    microstructure_noise_fraction: tuple[float, float] = (0.0, 0.00010)

    rule_based_activity_boost: tuple[float, float] = (0.10, 1.20)
    emotional_fomo_sensitivity: tuple[float, float] = (0.05, 0.45)
    emotional_fear_sensitivity: tuple[float, float] = (0.05, 0.50)
    information_informed_fraction: tuple[float, float] = (0.03, 0.35)
    mean_reversion_deviation_scale_fraction: tuple[float, float] = (0.002, 0.015)
    momentum_return_scale: tuple[float, float] = (0.002, 0.015)
    regime_stay_probability: tuple[float, float] = (0.970, 0.999)
    liquidity_log_vol: tuple[float, float] = (0.02, 0.20)
    adaptive_learning_rate: tuple[float, float] = (0.04, 0.35)

    def __post_init__(self) -> None:
        for name, bounds in asdict(self).items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"{name} must be a (low, high) tuple")
            low, high = (float(bounds[0]), float(bounds[1]))
            if not math.isfinite(low) or not math.isfinite(high) or high < low:
                raise ValueError(f"invalid sampling range for {name}: {bounds!r}")


@dataclass(frozen=True, slots=True)
class AIExperimentConfig:
    """Dataset, split, label, and model settings for one frozen AI experiment."""

    structures: int = 1_000
    seeds_per_structure: int = 1
    steps_per_run: int = 3_000
    candle_steps: int = 5
    label_horizon_candles: int = 20
    market_feature_lookback_candles: int = 20
    base_seed: int = 42

    null_world_fraction: float = 0.10
    efficient_null_share: float = 0.50
    min_active_worlds: int = 1
    max_active_worlds: int = 4
    random_background_probability: float = 0.45
    world_weight_concentration: float = 1.0
    minimum_world_weight: float = 0.04

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20

    validation_threshold_min: float = 0.50
    validation_threshold_max: float = 0.90
    validation_threshold_step: float = 0.025
    min_validation_trades: int = 25
    min_validation_runs: int = 5

    mlp_epochs: int = 120
    mlp_batch_size: int = 512
    mlp_learning_rate: float = 1e-3
    mlp_patience: int = 15

    def __post_init__(self) -> None:
        positive_int_fields = (
            "structures",
            "seeds_per_structure",
            "steps_per_run",
            "candle_steps",
            "label_horizon_candles",
            "market_feature_lookback_candles",
            "min_active_worlds",
            "max_active_worlds",
            "min_validation_trades",
            "min_validation_runs",
            "mlp_epochs",
            "mlp_batch_size",
            "mlp_patience",
        )
        for name in positive_int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be an integer > 0")
            object.__setattr__(self, name, int(value))
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, (int, np.integer)):
            raise ValueError("base_seed must be an integer")
        object.__setattr__(self, "base_seed", int(self.base_seed))

        if self.structures < 3:
            raise ValueError("at least three structures are required")
        if self.market_feature_lookback_candles < 2:
            raise ValueError("market_feature_lookback_candles must be >= 2")
        if self.min_active_worlds < 1 or self.max_active_worlds < self.min_active_worlds:
            raise ValueError("invalid min_active_worlds/max_active_worlds")
        if self.max_active_worlds > len(WORLD_NAMES):
            raise ValueError("max_active_worlds exceeds available simulator worlds")

        probability_fields = (
            "null_world_fraction",
            "efficient_null_share",
            "random_background_probability",
        )
        for name in probability_fields:
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be a finite real number")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)

        for name in (
            "world_weight_concentration",
            "minimum_world_weight",
            "train_fraction",
            "validation_fraction",
            "test_fraction",
            "validation_threshold_min",
            "validation_threshold_max",
            "validation_threshold_step",
            "mlp_learning_rate",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be a finite real number")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        if self.world_weight_concentration <= 0.0:
            raise ValueError("world_weight_concentration must be > 0")
        if not 0.0 <= self.minimum_world_weight < 1.0:
            raise ValueError("minimum_world_weight must lie in [0, 1)")
        if self.max_active_worlds * self.minimum_world_weight >= 1.0:
            raise ValueError(
                "minimum_world_weight is too large for max_active_worlds; their product must be < 1"
            )

        split_total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(split_total, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("train/validation/test fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0.0:
            raise ValueError("all split fractions must be positive")
        if not 0.0 <= self.validation_threshold_min <= self.validation_threshold_max <= 1.0:
            raise ValueError("invalid validation threshold range")
        if self.validation_threshold_step <= 0.0:
            raise ValueError("validation_threshold_step must be > 0")
        if self.mlp_learning_rate <= 0.0:
            raise ValueError("mlp_learning_rate must be > 0")

    @property
    def total_runs(self) -> int:
        return self.structures * self.seeds_per_structure


@dataclass(frozen=True, slots=True)
class RunSpec:
    simulation_id: int
    structure_index: int
    structure_id: str
    world_family: str
    config: SyntheticMarketConfig


# Pattern metadata that has a consistent dimensionless/causal interpretation across pattern types.
PATTERN_METADATA_FEATURES = [
    "pattern_height",
    "pattern_height_threshold",
    "pattern_span_bars",
    "pattern_age_at_availability_bars",
    "core_window_length",
    "observation_window_length",
    "observation_residual_noise_scale",
    "bandwidth",
    "level_error_ratio",
    "spacing_bars",
    "head_prominence_ratio",
    "shoulder_error_ratio",
    "shoulder_clearance_error_ratio",
    "neckline_slope_height_ratio",
    "high_boundary_error_ratio",
    "low_boundary_error_ratio",
    "boundary_convergence_ratio",
    "boundary_expansion_ratio",
    "pre_pattern_log_return",
    "pre_pattern_log_volatility",
    "pre_pattern_robust_log_volatility",
    "pre_pattern_trend_slope",
    "pattern_log_volatility",
    "pattern_robust_log_volatility",
    "pattern_mean_log_high_low_range",
    "pattern_to_pre_volume_ratio",
    "pattern_to_pre_log_volume_ratio",
    "breakout_strength_pattern_height_ratio",
    "breakout_strength_volatility_units",
]

OBSERVABLE_NUMERIC_FEATURES = [
    "geometry_fit_score",
    "pattern_length_bars",
    "pattern_height_to_threshold",
    "breakout_confirmed_by_availability",
    "competing_pattern",
    *PATTERN_METADATA_FEATURES,
    "recent_return_5",
    "recent_return_10",
    "recent_return_20",
    "recent_volatility_10",
    "recent_volatility_20",
    "recent_volatility_50",
    "recent_range_pct_20",
    "recent_volume_mean_20",
    "recent_volume_ratio",
    "recent_trend_strength_20",
    "observed_order_flow_imbalance",
    "observed_order_flow_persistence",
    "observed_buy_volume_fraction",
    "observed_spread_bps_current",
    "observed_spread_bps_mean",
    "observed_spread_bps_std",
    "observed_volume_mean",
    "observed_volume_cv",
    "observed_trade_count_mean",
    "observed_price_impact_proxy",
]

ORACLE_NUMERIC_FEATURES = [
    *(f"oracle_weight_{name}" for name in WORLD_NAMES),
    "oracle_strict_random_null",
    "oracle_efficient_random_null",
    "oracle_fundamental_vol_fraction",
    "oracle_base_spread_bps",
    "oracle_base_arrival_rate",
    "oracle_order_impact_fraction",
    "oracle_decision_signal_strength",
    "oracle_decision_noise",
    "oracle_fundamental_anchor_strength",
    "oracle_microstructure_noise_fraction",
    "oracle_inventory_skew_fraction",
    "oracle_rule_based_activity_boost",
    "oracle_emotional_fomo_sensitivity",
    "oracle_emotional_fear_sensitivity",
    "oracle_information_informed_fraction",
    "oracle_mean_reversion_deviation_scale_fraction",
    "oracle_momentum_return_scale",
    "oracle_regime_stay_probability",
    "oracle_liquidity_log_vol",
    "oracle_adaptive_learning_rate",
    "oracle_effective_rule_based_activity",
    "oracle_effective_emotional_fomo",
    "oracle_effective_emotional_fear",
    "oracle_effective_information_informed_share",
    "oracle_effective_mean_reversion_gain",
    "oracle_effective_momentum_gain",
    "oracle_effective_regime_persistence",
    "oracle_effective_liquidity_vol",
    "oracle_effective_adaptive_learning",
    "oracle_current_liquidity",
    "oracle_current_sentiment",
    "oracle_current_fomo",
    "oracle_current_fear",
    "oracle_current_greed",
    "oracle_current_information_gap_fraction",
    "oracle_current_information_event",
    "oracle_current_aggregate_signal",
    "oracle_current_maker_inventory_fraction",
    "oracle_current_rejected_volume_fraction",
]

OBSERVABLE_CATEGORICAL_FEATURES = ["pattern_type", "trade_direction"]
ORACLE_CATEGORICAL_FEATURES = ["oracle_regime"]

FUTURE_OR_OUTCOME_COLUMNS = {
    "profitable_after_costs",
    "future_20_bar_return",
    "label_horizon_bars",
    "profitable",
    "directional_correct",
    "gross_directional_return",
    "spread_slippage_cost",
    "brokerage_cost",
    "total_transaction_cost",
    "net_directional_return",
    "maximum_favorable_transaction_excursion",
    "maximum_adverse_transaction_excursion",
    "entry_index",
    "exit_index",
    "entry_mid",
    "exit_mid",
}


# ---------------------------------------------------------------------------
# Stable provenance / sampling
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_path(obj: Any) -> Path:
    return Path(inspect.getfile(obj)).resolve()


def _structure_id(config: SyntheticMarketConfig) -> str:
    frozen = replace(config, steps=1, seed=0)
    return "AI-ST-" + simulator_scenario_id(frozen)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = map(float, bounds)
    return low if high == low else float(rng.uniform(low, high))


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = map(float, bounds)
    if low < 0.0:
        raise ValueError("log-uniform bounds cannot be negative")
    if low == high:
        return low
    if low == 0.0:
        # Preserve the possibility of exact zero while otherwise sampling scale logarithmically.
        if rng.random() < 0.10:
            return 0.0
        low = max(high * 1e-4, 1e-12)
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _sample_world_weights(
    rng: np.random.Generator,
    selected: Sequence[str],
    *,
    concentration: float,
    minimum_weight: float,
) -> dict[str, float]:
    names = tuple(selected)
    if not names:
        raise ValueError("selected world set cannot be empty")
    if len(names) * minimum_weight >= 1.0:
        raise ValueError("minimum_world_weight is too large for the selected world count")
    raw = rng.gamma(shape=concentration, scale=1.0, size=len(names))
    if not np.all(np.isfinite(raw)) or float(np.sum(raw)) <= 0.0:
        raw = np.ones(len(names), dtype=float)
    base = raw / float(np.sum(raw))
    remaining = 1.0 - minimum_weight * len(names)
    weights = minimum_weight + remaining * base
    return {name: float(weight) for name, weight in zip(names, weights)}


def _sample_structure_template(
    structure_index: int,
    is_null: bool,
    experiment: AIExperimentConfig,
    ranges: WorldSamplingRanges,
    rng: np.random.Generator,
) -> tuple[SyntheticMarketConfig, str]:
    base = SyntheticMarketConfig(steps=experiment.steps_per_run, seed=0)

    common = dict(
        steps=experiment.steps_per_run,
        seed=0,
        fair_value_step_vol_fraction=_log_uniform(rng, ranges.fundamental_vol_fraction),
        base_spread_fraction=_log_uniform(rng, ranges.base_spread_fraction),
        base_arrival_rate=_uniform(rng, ranges.base_arrival_rate),
        order_impact_fraction=_log_uniform(rng, ranges.order_impact_fraction),
        decision_signal_strength=_uniform(rng, ranges.decision_signal_strength),
        decision_noise=_uniform(rng, ranges.decision_noise),
        fundamental_anchor_strength=_uniform(rng, ranges.fundamental_anchor_strength),
        microstructure_noise_fraction=_log_uniform(rng, ranges.microstructure_noise_fraction),
    )

    if is_null:
        mode = "efficient" if rng.random() < experiment.efficient_null_share else "microstructure"
        config = replace(
            base,
            **common,
            strict_random_null=True,
            random_null_mode=mode,
            world_weights={"random": 1.0},
        )
        return config, f"null_{mode}"

    count = int(rng.integers(experiment.min_active_worlds, experiment.max_active_worlds + 1))
    # A request for all nine worlds necessarily includes the random world; for smaller mixtures it
    # remains an optional background population according to the pre-registered probability.
    include_random = (
        count > len(STRUCTURED_WORLDS)
        or (count >= 2 and rng.random() < experiment.random_background_probability)
    )
    structured_count = count - int(include_random)
    selected = list(rng.choice(STRUCTURED_WORLDS, size=structured_count, replace=False))
    if include_random:
        selected.append("random")
    weights = _sample_world_weights(
        rng,
        selected,
        concentration=experiment.world_weight_concentration,
        minimum_weight=experiment.minimum_world_weight,
    )

    config = replace(
        base,
        **common,
        strict_random_null=False,
        world_weights=weights,
        rule_based=replace(
            base.rule_based,
            activity_boost=_uniform(rng, ranges.rule_based_activity_boost),
        ),
        emotional=replace(
            base.emotional,
            fomo_sensitivity=_uniform(rng, ranges.emotional_fomo_sensitivity),
            fear_sensitivity=_uniform(rng, ranges.emotional_fear_sensitivity),
        ),
        information=replace(
            base.information,
            informed_fraction=_uniform(rng, ranges.information_informed_fraction),
        ),
        mean_reversion=replace(
            base.mean_reversion,
            deviation_scale_fraction=_log_uniform(
                rng, ranges.mean_reversion_deviation_scale_fraction
            ),
        ),
        momentum=replace(
            base.momentum,
            return_scale=_log_uniform(rng, ranges.momentum_return_scale),
        ),
        regime=replace(
            base.regime,
            stay_probability=_uniform(rng, ranges.regime_stay_probability),
        ),
        liquidity=replace(
            base.liquidity,
            log_liquidity_vol=_uniform(rng, ranges.liquidity_log_vol),
        ),
        adaptive=replace(
            base.adaptive,
            learning_rate=_uniform(rng, ranges.adaptive_learning_rate),
        ),
    )
    return config, "structured"


def sample_run_specs(
    experiment: AIExperimentConfig,
    ranges: WorldSamplingRanges = WorldSamplingRanges(),
) -> list[RunSpec]:
    """Create deterministic sampled structures and independent repeated seeds."""

    master = np.random.default_rng(experiment.base_seed)
    null_count = int(round(experiment.structures * experiment.null_world_fraction))
    null_flags = np.zeros(experiment.structures, dtype=bool)
    if null_count:
        null_flags[:null_count] = True
        master.shuffle(null_flags)

    specs: list[RunSpec] = []
    simulation_id = 0
    seen_structures: set[str] = set()
    used_seeds: set[int] = set()
    for structure_index in range(experiment.structures):
        # Duplicate continuous structures are extremely unlikely, but resample deterministically
        # if canonical hashing ever finds one.
        for _attempt in range(100):
            template, family = _sample_structure_template(
                structure_index,
                bool(null_flags[structure_index]),
                experiment,
                ranges,
                master,
            )
            template = canonical_behavior_config(template)
            sid = _structure_id(template)
            if sid not in seen_structures:
                seen_structures.add(sid)
                break
        else:  # pragma: no cover - effectively impossible with continuous ranges
            raise RuntimeError("could not generate a unique synthetic structure")

        seed_rng = np.random.default_rng(experiment.base_seed + 1_000_003 * (structure_index + 1))
        for _replicate in range(experiment.seeds_per_structure):
            # Repeated seeds are intended to be independent replications.  Integer collisions are
            # extremely unlikely but should still be ruled out deterministically rather than
            # silently reducing the effective sample size. Uniqueness spans structures as well
            # as replicates so train/test partitions cannot share a simulator RNG stream.
            while True:
                seed = int(seed_rng.integers(0, 2**31 - 1))
                if seed not in used_seeds:
                    used_seeds.add(seed)
                    break
            config = replace(template, seed=seed)
            specs.append(
                RunSpec(
                    simulation_id=simulation_id,
                    structure_index=structure_index,
                    structure_id=sid,
                    world_family=family,
                    config=config,
                )
            )
            simulation_id += 1
    return specs


# ---------------------------------------------------------------------------
# Causal features and labels
# ---------------------------------------------------------------------------


def _safe_log_returns(prices: Sequence[float]) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.size < 2:
        return np.empty(0, dtype=float)
    values = np.maximum(values, _EPS)
    return np.diff(np.log(values))


def _simple_return(prices: Sequence[float], lookback: int) -> float:
    if len(prices) < 2:
        return 0.0
    start_index = max(0, len(prices) - 1 - lookback)
    start = float(prices[start_index])
    end = float(prices[-1])
    return end / start - 1.0 if start > 0.0 else 0.0


def _volatility(prices: Sequence[float], lookback: int) -> float:
    returns = _safe_log_returns(prices[-(lookback + 1) :])
    return float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    return float("nan")


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    return float("nan")


def _resolved_trade_direction(event: PatternDetection) -> tuple[int, str] | None:
    direction = trade_direction(event)
    if direction is None:
        return None
    return direction, "bullish" if direction > 0 else "bearish"


def _pattern_features(event: PatternDetection) -> dict[str, Any]:
    metadata = event.metadata
    height = _metadata_float(metadata, "pattern_height")
    threshold = _metadata_float(metadata, "pattern_height_threshold")
    if math.isfinite(height) and math.isfinite(threshold) and threshold > 0.0:
        height_to_threshold = height / threshold
    else:
        height_to_threshold = float("nan")

    output: dict[str, Any] = {
        "geometry_fit_score": float(event.geometry_fit_score),
        "pattern_length_bars": float(event.end_index - event.start_index + 1),
        "pattern_height_to_threshold": height_to_threshold,
        "breakout_confirmed_by_availability": _metadata_bool(
            metadata, "breakout_confirmed_by_availability"
        ),
        "competing_pattern": _metadata_bool(metadata, "competing_pattern"),
    }
    for key in PATTERN_METADATA_FEATURES:
        output[key] = _metadata_float(metadata, key)
    return output


def _observable_candle_features(
    candles: Sequence[PriceCandle],
    available_index: int,
) -> dict[str, float]:
    # The longest feature needs 51 closes, regardless of the full run length.
    past = candles[max(0, available_index - 50) : available_index + 1]
    closes = [float(c.close) for c in past]
    current = candles[available_index]
    recent = past[-20:]
    high = max((float(c.high) for c in recent), default=float(current.high))
    low = min((float(c.low) for c in recent), default=float(current.low))
    mean_volume = float(np.mean([float(c.volume) for c in recent])) if recent else 0.0
    trend_returns = _safe_log_returns(closes[-21:])
    trend_path = float(np.sum(np.abs(trend_returns)))
    return {
        "recent_return_5": _simple_return(closes, 5),
        "recent_return_10": _simple_return(closes, 10),
        "recent_return_20": _simple_return(closes, 20),
        "recent_volatility_10": _volatility(closes, 10),
        "recent_volatility_20": _volatility(closes, 20),
        "recent_volatility_50": _volatility(closes, 50),
        "recent_range_pct_20": (high - low) / max(float(current.close), _EPS),
        "recent_volume_mean_20": mean_volume,
        "recent_volume_ratio": float(current.volume) / max(mean_volume, _EPS),
        "recent_trend_strength_20": (
            float(np.sum(trend_returns)) / trend_path if trend_path > 0.0 else 0.0
        ),
    }


def _lag1_correlation(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= _EPS or float(np.std(right)) <= _EPS:
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def _observable_microstructure_features(
    history: Sequence[Any],
    detection_end_step: int,
    lookback_steps: int,
) -> dict[str, float]:
    # Synthetic steps are contiguous and 1-based. Slice endpoint ``detection_end_step`` therefore
    # includes the complete step whose label equals that value.
    end = min(len(history), max(0, int(detection_end_step)))
    start = max(0, end - int(lookback_steps))
    window = history[start:end]
    if not window:
        return {name: 0.0 for name in (
            "observed_order_flow_imbalance",
            "observed_order_flow_persistence",
            "observed_buy_volume_fraction",
            "observed_spread_bps_current",
            "observed_spread_bps_mean",
            "observed_spread_bps_std",
            "observed_volume_mean",
            "observed_volume_cv",
            "observed_trade_count_mean",
            "observed_price_impact_proxy",
        )}

    volumes = np.asarray([float(item.buy_volume + item.sell_volume) for item in window])
    signed = np.asarray([float(item.signed_order_flow) for item in window])
    flow_ratio = np.divide(signed, np.maximum(volumes, 1.0))
    buy_total = float(sum(item.buy_volume for item in window))
    sell_total = float(sum(item.sell_volume for item in window))
    total_volume = buy_total + sell_total
    # Reference prices are latent simulator state and can differ from quoted mids because
    # of inventory skew. Observable features must be reconstructible from public quotes.
    def quote(item: Any, *, closing: bool = False) -> tuple[float, float]:
        bid = float(item.post_trade_bid if closing else item.bid)
        ask = float(item.post_trade_ask if closing else item.ask)
        if not math.isfinite(bid) or not math.isfinite(ask) or not 0.0 < bid <= ask:
            raise ValueError("observable features require valid opening and closing bid/ask quotes")
        mid = bid / 2.0 + ask / 2.0
        return mid, (ask - bid) / mid * 1e4

    spread_bps = np.asarray([quote(item)[1] for item in window], dtype=float)
    trade_counts = np.asarray([float(item.trade_count) for item in window], dtype=float)
    volume_mean = float(np.mean(volumes)) if volumes.size else 0.0
    volume_sd = float(np.std(volumes, ddof=1)) if volumes.size > 1 else 0.0

    closing_quotes = [quote(item, closing=True) for item in window]
    price_returns = _safe_log_returns([mid for mid, _spread in closing_quotes])
    impact_proxy = (
        float(np.mean(np.abs(price_returns))) / max(volume_mean, 1.0)
        if price_returns.size
        else 0.0
    )
    return {
        "observed_order_flow_imbalance": (
            float(np.sum(signed) / total_volume) if total_volume > 0.0 else 0.0
        ),
        "observed_order_flow_persistence": _lag1_correlation(flow_ratio),
        "observed_buy_volume_fraction": buy_total / total_volume if total_volume > 0.0 else 0.5,
        "observed_spread_bps_current": closing_quotes[-1][1],
        "observed_spread_bps_mean": float(np.mean(spread_bps)),
        "observed_spread_bps_std": float(np.std(spread_bps, ddof=1)) if len(spread_bps) > 1 else 0.0,
        "observed_volume_mean": volume_mean,
        "observed_volume_cv": volume_sd / max(volume_mean, _EPS),
        "observed_trade_count_mean": float(np.mean(trade_counts)),
        "observed_price_impact_proxy": impact_proxy,
    }


def _oracle_features(
    config: SyntheticMarketConfig,
    snapshot: Any,
) -> dict[str, Any]:
    weights = config.normalized_world_weights
    total_attempted = float(snapshot.traded_volume + snapshot.rejected_volume)
    output: dict[str, Any] = {
        **{f"oracle_weight_{name}": float(weights.get(name, 0.0)) for name in WORLD_NAMES},
        "oracle_strict_random_null": float(config.is_strict_random_null),
        "oracle_efficient_random_null": float(config.is_efficient_random_null),
        "oracle_fundamental_vol_fraction": float(config.effective_fundamental_vol_fraction),
        "oracle_base_spread_bps": float(config.effective_base_spread_fraction * 1e4),
        "oracle_base_arrival_rate": float(config.base_arrival_rate),
        "oracle_order_impact_fraction": float(config.order_impact_fraction),
        "oracle_decision_signal_strength": float(config.decision_signal_strength),
        "oracle_decision_noise": float(config.decision_noise),
        "oracle_fundamental_anchor_strength": float(config.fundamental_anchor_strength),
        "oracle_microstructure_noise_fraction": float(config.microstructure_noise_fraction),
        "oracle_inventory_skew_fraction": float(config.effective_inventory_skew_fraction),
        "oracle_rule_based_activity_boost": float(config.rule_based.activity_boost),
        "oracle_emotional_fomo_sensitivity": float(config.emotional.fomo_sensitivity),
        "oracle_emotional_fear_sensitivity": float(config.emotional.fear_sensitivity),
        "oracle_information_informed_fraction": float(config.information.informed_fraction),
        "oracle_mean_reversion_deviation_scale_fraction": float(
            config.mean_reversion.deviation_scale_fraction
        ),
        "oracle_momentum_return_scale": float(config.momentum.return_scale),
        "oracle_regime_stay_probability": float(config.regime.stay_probability),
        "oracle_liquidity_log_vol": float(config.liquidity.log_liquidity_vol),
        "oracle_adaptive_learning_rate": float(config.adaptive.learning_rate),
        "oracle_effective_rule_based_activity": float(weights.get("rule_based", 0.0))
        * float(config.rule_based.activity_boost),
        "oracle_effective_emotional_fomo": float(weights.get("emotional", 0.0))
        * float(config.emotional.fomo_sensitivity),
        "oracle_effective_emotional_fear": float(weights.get("emotional", 0.0))
        * float(config.emotional.fear_sensitivity),
        "oracle_effective_information_informed_share": float(weights.get("information", 0.0))
        * float(config.information.informed_fraction),
        "oracle_effective_mean_reversion_gain": float(weights.get("mean_reversion", 0.0))
        / max(float(config.mean_reversion.deviation_scale_fraction), _EPS),
        "oracle_effective_momentum_gain": float(weights.get("momentum", 0.0))
        / max(float(config.momentum.return_scale), _EPS),
        "oracle_effective_regime_persistence": float(weights.get("regime", 0.0))
        * float(config.regime.stay_probability),
        "oracle_effective_liquidity_vol": float(weights.get("liquidity", 0.0))
        * float(config.liquidity.log_liquidity_vol),
        "oracle_effective_adaptive_learning": float(weights.get("adaptive", 0.0))
        * float(config.adaptive.learning_rate),
        "oracle_current_liquidity": float(snapshot.liquidity),
        "oracle_current_sentiment": float(snapshot.sentiment),
        "oracle_current_fomo": float(snapshot.fomo),
        "oracle_current_fear": float(snapshot.fear),
        "oracle_current_greed": float(snapshot.greed),
        "oracle_current_information_gap_fraction": float(snapshot.information_gap)
        / max(abs(float(snapshot.public_fundamental)), _EPS),
        "oracle_current_information_event": float(snapshot.information_event),
        "oracle_current_aggregate_signal": float(snapshot.aggregate_signal),
        "oracle_current_maker_inventory_fraction": float(snapshot.maker_inventory)
        / max(float(config.max_abs_inventory), 1.0),
        "oracle_current_rejected_volume_fraction": (
            float(snapshot.rejected_volume) / total_attempted if total_attempted > 0.0 else 0.0
        ),
        "oracle_regime": str(snapshot.regime),
    }
    return output


def _future_excursions(
    candles: Sequence[PriceCandle],
    entry_index: int,
    exit_index: int,
    entry_mid: float,
    direction: int,
) -> tuple[float, float]:
    """Transaction-candle excursions from entry quote mid, not executable quote-path MAE/MFE."""
    future = candles[entry_index : exit_index + 1]
    if not future:
        return 0.0, 0.0
    if direction > 0:
        favorable = max(float(c.high) for c in future) / entry_mid - 1.0
        adverse = min(float(c.low) for c in future) / entry_mid - 1.0
    else:
        favorable = 1.0 - min(float(c.low) for c in future) / entry_mid
        adverse = 1.0 - max(float(c.high) for c in future) / entry_mid
    return float(max(0.0, favorable)), float(min(0.0, adverse))


def create_pattern_row(
    spec: RunSpec,
    history: Sequence[Any],
    candles: Sequence[PriceCandle],
    event: PatternDetection,
    entry_mids: Sequence[float],
    exit_mids: Sequence[float],
    entry_spreads: Sequence[float],
    exit_spreads: Sequence[float],
    experiment: AIExperimentConfig,
    costs: CostModel,
) -> dict[str, Any] | None:
    """Build one causal feature row and one future-only supervised target."""

    resolved = _resolved_trade_direction(event)
    if resolved is None:
        return None
    direction, trade_direction = resolved
    available = int(event.available_at_index)
    entry = int(event.earliest_execution_index)
    exit_ = entry + experiment.label_horizon_candles - 1
    if available < 0 or available >= len(candles) or entry <= available:
        raise AssertionError("detector availability/execution contract was violated")
    if entry >= len(candles) or exit_ >= len(candles):
        return None

    gross, spread_slippage, brokerage, total_cost, net = net_trade_return(
        float(entry_mids[entry]),
        float(exit_mids[exit_]),
        direction,
        float(entry_spreads[entry]),
        float(exit_spreads[exit_]),
        costs,
    )
    favorable, adverse = _future_excursions(
        candles,
        entry,
        exit_,
        float(entry_mids[entry]),
        direction,
    )

    current_candle = candles[available]
    current_snapshot = history[current_candle.end_step - 1]
    if int(current_snapshot.step) != int(current_candle.end_step):
        raise AssertionError("candle-to-snapshot chronology mismatch")

    row: dict[str, Any] = {
        "simulation_id": spec.simulation_id,
        "structure_index": spec.structure_index,
        "structure_id": spec.structure_id,
        "world_family": spec.world_family,
        "scenario_id": simulator_scenario_id(spec.config),
        "run_id": simulator_run_id(spec.config),
        "seed": spec.config.seed,
        "event_id": event.event_id or "",
        "detector_version": DETECTOR_VERSION,
        "detector_config_hash": str(event.metadata.get("config_hash", "")),
        "pattern_type": str(event.pattern_name),
        "classical_expected_direction": str(event.expected_direction),
        "trade_direction": trade_direction,
        "pattern_start_index": int(event.start_index),
        "pattern_end_index": int(event.end_index),
        "available_at_index": available,
        "feature_end_step": int(current_candle.end_step),
        **_pattern_features(event),
        **_observable_candle_features(candles, available),
        **_observable_microstructure_features(
            history,
            current_candle.end_step,
            experiment.market_feature_lookback_candles * experiment.candle_steps,
        ),
        **_oracle_features(spec.config, current_snapshot),
        # Future/outcome fields below are NEVER used as model inputs.
        "entry_index": entry,
        "exit_index": exit_,
        "entry_mid": float(entry_mids[entry]),
        "exit_mid": float(exit_mids[exit_]),
        "gross_directional_return": float(gross),
        "spread_slippage_cost": float(spread_slippage),
        "brokerage_cost": float(brokerage),
        "total_transaction_cost": float(total_cost),
        "net_directional_return": float(net),
        "directional_correct": int(gross > 0.0),
        "maximum_favorable_transaction_excursion": favorable,
        "maximum_adverse_transaction_excursion": adverse,
        "profitable": int(net > 0.0),
    }
    return row


def _process_run_spec(
    spec: RunSpec,
    experiment: AIExperimentConfig,
    costs: CostModel,
    git_commit: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Worker-safe single-run simulation -> detector -> AI rows."""

    # Avoid BLAS oversubscription when many Python worker processes are active.
    with threadpool_limits(limits=1):
        history = simulate_market(spec.config, record_step_diagnostics=False)
        if any(snapshot.arrival_cap_hit for snapshot in history):
            raise RuntimeError(
                "synthetic arrival cap was hit; the sampled run is truncated and cannot enter "
                "the AI research dataset"
            )
        candles = build_candles(
            history,
            experiment.candle_steps,
            include_partial=False,
            require_contiguous_steps=True,
        )
        dataset_id = candle_dataset_id(
            spec.config,
            experiment.candle_steps,
            include_partial=False,
        )
        context = RunContext(
            run_id=simulator_run_id(spec.config),
            market_world=spec.world_family,
            seed=spec.config.seed,
            asset=f"synthetic:{spec.structure_id}",
            timeframe=f"{experiment.candle_steps}_sim_steps",
            dataset_id=dataset_id,
            git_commit=git_commit,
        )
        result = detect_pattern_universe(candles, DEFAULT_PATTERN_CONFIG, context)
        entry_mids, exit_mids, entry_spreads, exit_spreads = synthetic_execution_arrays(
            history, candles
        )

        rows: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        skipped_unresolved_direction = 0
        skipped_incomplete_horizon = 0
        for event in result.confirmed_breakout_events:
            if event.event_id:
                if event.event_id in seen_event_ids:
                    raise AssertionError("duplicate causal event_id emitted within one detector run")
                seen_event_ids.add(event.event_id)
            row = create_pattern_row(
                spec,
                history,
                candles,
                event,
                entry_mids,
                exit_mids,
                entry_spreads,
                exit_spreads,
                experiment,
                costs,
            )
            if row is not None:
                rows.append(row)
            elif _resolved_trade_direction(event) is None:
                skipped_unresolved_direction += 1
            else:
                skipped_incomplete_horizon += 1

    summary = {
        "simulation_id": spec.simulation_id,
        "structure_index": spec.structure_index,
        "structure_id": spec.structure_id,
        "world_family": spec.world_family,
        "seed": spec.config.seed,
        "scenario_id": simulator_scenario_id(spec.config),
        "run_id": simulator_run_id(spec.config),
        "active_worlds": "+".join(spec.config.active_worlds),
        "world_weights_json": json.dumps(dict(spec.config.normalized_world_weights), sort_keys=True),
        "steps": spec.config.steps,
        "candles": len(candles),
        "raw_candidates": len(result.candidates),
        "causal_events": len(result.events),
        "confirmed_breakout_events": len(result.confirmed_breakout_events),
        "training_rows": len(rows),
        "skipped_unresolved_direction": skipped_unresolved_direction,
        "skipped_incomplete_horizon": skipped_incomplete_horizon,
    }
    return rows, summary


def iter_pattern_runs(
    specs: Sequence[RunSpec],
    experiment: AIExperimentConfig,
    *,
    costs: CostModel = CostModel(),
    jobs: int = 1,
    git_commit: str | None = None,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Yield shared detector/feature/label results in design order with bounded buffering."""
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs == 0 or jobs < -1:
        raise ValueError("jobs must be -1 or a positive integer")
    if jobs == 1:
        for spec in specs:
            yield _process_run_spec(spec, experiment, costs, git_commit)
        return
    workers = min(len(specs), (os.cpu_count() or 1) if jobs == -1 else jobs)
    if not workers:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        remaining = iter(specs)
        pending = deque()

        def submit_one() -> None:
            spec = next(remaining, None)
            if spec is not None:
                pending.append((spec.simulation_id, pool.submit(
                    _process_run_spec, spec, experiment, costs, git_commit,
                )))

        for _ in range(workers * 2):
            submit_one()
        while pending:
            simulation_id, future = pending.popleft()
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"AI dataset simulation {simulation_id} failed") from exc
            yield result
            submit_one()


def checkpointed_pattern_runs(specs, experiment, *, output_dir, costs, ranges, jobs, git_commit):
    """Commit each complete simulation before yielding it; replay on restart.

    SQLite transactions discard incomplete writes. A single writer is required per
    output directory. CSV assembly and training restart, but committed simulations do not.
    """
    files = [_module_path(obj) for obj in (
        checkpointed_pattern_runs, SyntheticMarketConfig, build_candles,
        detect_pattern_universe, net_trade_return,
    )]
    contract = json.dumps({
        "experiment": asdict(experiment), "costs": asdict(costs),
        "ranges": asdict(ranges), "git_commit": git_commit,
        "code": {str(path): _file_sha256(path) for path in files},
    }, sort_keys=True)
    connection = sqlite3.connect(output_dir / "generation_checkpoints.sqlite3")
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS contract (value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS completed (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
        saved = connection.execute("SELECT value FROM contract").fetchone()
        if saved is None:
            connection.execute("INSERT INTO contract VALUES (?)", (contract,))
            connection.commit()
        elif saved[0] != contract:
            raise ValueError("Checkpoint settings/code differ; restore the original procedure or use a new output directory")
        # Commits are in design order, so only a contiguous prefix may be reused.
        count, last = connection.execute("SELECT COUNT(*), MAX(id) FROM completed").fetchone()
        if count and (last != count - 1 or count > len(specs)):
            raise ValueError("Invalid checkpoint simulation sequence")
        if count:
            print(f"Resuming: reusing {count} committed simulations", flush=True)
        for simulation_id, payload in connection.execute("SELECT id, payload FROM completed ORDER BY id"):
            rows, summary = json.loads(zlib.decompress(payload))
            if summary["simulation_id"] != simulation_id:
                raise ValueError("Checkpoint identity mismatch")
            yield rows, summary
        for rows, summary in iter_pattern_runs(
            specs[count:], experiment, costs=costs, jobs=jobs, git_commit=git_commit,
        ):
            if summary["simulation_id"] != count:
                raise ValueError("Nonsequential simulation checkpoint")
            # Internal Python JSON preserves NaN feature values for the existing imputer.
            payload = zlib.compress(json.dumps([rows, summary], allow_nan=True).encode("utf-8"))
            connection.execute("INSERT INTO completed VALUES (?, ?)", (count, payload))
            connection.commit()
            count += 1
            yield rows, summary
    finally:
        connection.close()


def build_training_dataset(
    experiment: AIExperimentConfig,
    output_dir: Path,
    *,
    jobs: int = 1,
    costs: CostModel = CostModel(),
    ranges: WorldSamplingRanges = WorldSamplingRanges(),
    git_commit: str | None = None,
    checkpoint: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sampled synthetic structures and write one row per eligible causal event.

    Pattern rows are streamed to disk in deterministic ``simulation_id`` order instead of being
    retained as a Python list of dictionaries.  This materially reduces peak memory when research
    runs produce hundreds of thousands of occurrences.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    specs = sample_run_specs(experiment, ranges)
    dataset_path = output_dir / "pattern_training_dataset.csv"
    temporary_path = output_dir / ".pattern_training_dataset.building.csv"
    if temporary_path.exists():
        temporary_path.unlink()

    run_summaries: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    writer: csv.DictWriter | None = None
    labeled_count = 0

    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        def write_result(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
            nonlocal fieldnames, writer, labeled_count
            run_summaries.append(summary)
            for row in rows:
                if fieldnames is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                elif set(row) != set(fieldnames):
                    missing = sorted(set(fieldnames) - set(row))
                    extra = sorted(set(row) - set(fieldnames))
                    raise AssertionError(
                        f"AI dataset row schema changed within one run: missing={missing}, extra={extra}"
                    )
                assert writer is not None
                writer.writerow(row)
                labeled_count += 1

        for completed, (rows, summary) in enumerate(
            (checkpointed_pattern_runs(
                specs, experiment, output_dir=output_dir, costs=costs, ranges=ranges,
                jobs=jobs, git_commit=git_commit,
            ) if checkpoint else iter_pattern_runs(
                specs, experiment, costs=costs, jobs=jobs, git_commit=git_commit,
            )),
            start=1,
        ):
            write_result(rows, summary)
            if completed == 1 or completed % max(1, min(100, len(specs) // 20)) == 0:
                print(
                    f"dataset progress: {completed}/{len(specs)} runs, "
                    f"{labeled_count} labeled causal patterns",
                    flush=True,
                )

    if fieldnames is None or labeled_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "No eligible causal pattern rows were generated. Increase path length/structure count "
            "or inspect detector diagnostics before changing frozen detector thresholds."
        )
    os.replace(temporary_path, dataset_path)

    dataset = pd.read_csv(dataset_path)
    runs = pd.DataFrame(run_summaries).sort_values("simulation_id").reset_index(drop=True)
    duplicate_keys = dataset.duplicated(["run_id", "event_id"], keep=False) & dataset["event_id"].ne("")
    if bool(duplicate_keys.any()):
        raise AssertionError("duplicate run_id/event_id pairs found in AI dataset")
    if bool((dataset["entry_index"] <= dataset["available_at_index"]).any()):
        raise AssertionError("AI dataset contains execution before/at information availability")

    runs.to_csv(output_dir / "synthetic_runs.csv", index=False)
    print(
        f"dataset complete: {len(dataset)} rows across {dataset['structure_id'].nunique()} "
        f"structures and {dataset['run_id'].nunique()} runs; "
        f"success_rate={dataset['profitable'].mean():.4f}",
        flush=True,
    )
    return dataset, runs


# ---------------------------------------------------------------------------
# Leakage-safe structure-level split
# ---------------------------------------------------------------------------


def _partition_counts(total: int, experiment: AIExperimentConfig) -> dict[str, int]:
    """Deterministic integer allocation close to the requested split fractions."""

    if total <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    names = ("train", "validation", "test")
    fractions = np.asarray(
        [experiment.train_fraction, experiment.validation_fraction, experiment.test_fraction],
        dtype=float,
    )
    raw = fractions * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(np.sum(counts))
    fractional = raw - counts
    order = sorted(range(3), key=lambda i: (-fractional[i], i))
    for index in order[:remainder]:
        counts[index] += 1

    # When a stratum has enough structures, represent it in every partition.  This is based only
    # on the pre-generated market family, never on profitability labels.
    if total >= 3:
        for missing in [i for i, count in enumerate(counts) if count == 0]:
            donors = [i for i, count in enumerate(counts) if count > 1]
            if not donors:
                break
            donor = max(donors, key=lambda i: (counts[i], fractions[i], -i))
            counts[donor] -= 1
            counts[missing] += 1
    return dict(zip(names, map(int, counts)))


def assign_structure_splits(
    run_universe: pd.DataFrame,
    experiment: AIExperimentConfig,
) -> pd.DataFrame:
    """Freeze partitions from the complete run design, without reading pattern rows or labels."""
    required = ["run_id", "structure_id", "world_family"]
    if run_universe[required].isna().any().any():
        raise ValueError("run universe identities cannot be missing")
    universe = run_universe[required].astype(str).copy()
    if universe["run_id"].duplicated().any():
        raise ValueError("run universe contains duplicate run_id values")
    family_counts = universe.groupby("structure_id")["world_family"].nunique()
    if bool((family_counts > 1).any()):
        raise ValueError("one synthetic structure maps to multiple world families")

    structure_table = (
        universe.groupby("structure_id", sort=True)
        .agg(world_family=("world_family", "first"), run_count=("run_id", "nunique"))
        .reset_index()
    )
    if len(structure_table) < 3:
        raise ValueError("at least three generated structures are required")
    structure_table["split_stratum"] = np.where(
        structure_table["world_family"].str.startswith("null_"),
        "null",
        "structured",
    )

    rng = np.random.default_rng(experiment.base_seed + 10_007)
    split_lookup: dict[str, str] = {}
    # Stratify on null-versus-structured design only.  Never use target outcomes to shape splits.
    for _stratum, group in structure_table.groupby("split_stratum", sort=True):
        ids = np.asarray(sorted(group["structure_id"].tolist()), dtype=object)
        rng.shuffle(ids)
        counts = _partition_counts(len(ids), experiment)
        cursor = 0
        for split in ("train", "validation", "test"):
            stop = cursor + counts[split]
            for sid in ids[cursor:stop]:
                split_lookup[str(sid)] = split
            cursor = stop
        if cursor != len(ids):
            raise AssertionError("structure split allocation did not consume the full stratum")

    # Tiny smoke tests can have too few null structures for stratification to populate all global
    # partitions. Fall back to a deterministic overall allocation rather than silently losing a set.
    assigned_counts = {name: sum(value == name for value in split_lookup.values()) for name in ("train", "validation", "test")}
    if any(count == 0 for count in assigned_counts.values()):
        ids = np.asarray(sorted(structure_table["structure_id"].tolist()), dtype=object)
        rng = np.random.default_rng(experiment.base_seed + 10_007)
        rng.shuffle(ids)
        counts = _partition_counts(len(ids), experiment)
        split_lookup.clear()
        cursor = 0
        for split in ("train", "validation", "test"):
            stop = cursor + counts[split]
            for sid in ids[cursor:stop]:
                split_lookup[str(sid)] = split
            cursor = stop

    if set(split_lookup) != set(structure_table["structure_id"]):
        raise AssertionError("structure split manifest is incomplete")

    structure_table["split"] = structure_table["structure_id"].map(split_lookup)
    return structure_table.sort_values("structure_id").reset_index(drop=True)


def split_by_structure(
    dataset: pd.DataFrame,
    experiment: AIExperimentConfig,
    run_universe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign complete synthetic structures to partitions before conditioning on pattern presence.

    ``run_universe`` should be ``synthetic_runs.csv`` (or the in-memory equivalent) and therefore
    includes structures/runs that produced zero eligible patterns.  Keeping them in the split
    manifest prevents the economic backtest from conditioning on "a pattern happened somewhere in
    this run" and gives zero-pattern test runs their correct zero-return contribution.
    """

    required_dataset = {"structure_id", "world_family", "run_id", "profitable"}
    missing_dataset = required_dataset.difference(dataset.columns)
    if missing_dataset:
        raise KeyError(f"dataset missing structure-split columns: {sorted(missing_dataset)}")

    if run_universe is None:
        universe = (
            dataset[["structure_id", "world_family", "run_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        required_runs = {"structure_id", "world_family", "run_id"}
        missing_runs = required_runs.difference(run_universe.columns)
        if missing_runs:
            raise KeyError(f"run universe missing columns: {sorted(missing_runs)}")
        universe = run_universe[["structure_id", "world_family", "run_id"]].copy()
        if universe["run_id"].duplicated().any():
            raise ValueError("run universe contains duplicate run_id values")

    universe["structure_id"] = universe["structure_id"].astype(str)
    universe["world_family"] = universe["world_family"].astype(str)
    universe["run_id"] = universe["run_id"].astype(str)
    if universe["run_id"].duplicated().any():
        raise ValueError("one run_id maps to multiple universe records")
    row_identity = dataset[["run_id", "structure_id", "world_family"]].astype(str).drop_duplicates()
    identity_check = row_identity.merge(
        universe, on="run_id", how="left", suffixes=("_row", "_universe"), validate="many_to_one"
    )
    if bool((identity_check["structure_id_row"] != identity_check["structure_id_universe"]).any()
            or (identity_check["world_family_row"] != identity_check["world_family_universe"]).any()):
        raise ValueError("pattern run identity does not match the complete generated run universe")
    structure_table = assign_structure_splits(universe, experiment)
    split_lookup = dict(zip(structure_table["structure_id"], structure_table.pop("split")))

    result = dataset.copy()
    result["structure_id"] = result["structure_id"].astype(str)
    result["split"] = result["structure_id"].map(split_lookup)
    if result["split"].isna().any():
        raise AssertionError("pattern row belongs to an unassigned structure")

    # Build a manifest over the entire generated universe, including zero-pattern structures.
    row_stats = (
        result.groupby("structure_id", sort=True)
        .agg(
            runs_with_pattern_rows=("run_id", "nunique"),
            pattern_rows=("profitable", "size"),
            success_rate=("profitable", "mean"),
        )
        .reset_index()
    )
    manifest = structure_table.merge(row_stats, on="structure_id", how="left")
    manifest["runs_with_pattern_rows"] = manifest["runs_with_pattern_rows"].fillna(0).astype(int)
    manifest["pattern_rows"] = manifest["pattern_rows"].fillna(0).astype(int)
    manifest["zero_pattern_runs"] = manifest["run_count"] - manifest["runs_with_pattern_rows"]
    manifest["split"] = manifest["structure_id"].map(split_lookup)
    manifest = manifest[
        [
            "structure_id",
            "split",
            "split_stratum",
            "world_family",
            "run_count",
            "runs_with_pattern_rows",
            "zero_pattern_runs",
            "pattern_rows",
            "success_rate",
        ]
    ].sort_values("structure_id").reset_index(drop=True)

    groups = {
        name: set(manifest.loc[manifest["split"] == name, "structure_id"].astype(str))
        for name in ("train", "validation", "test")
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise AssertionError("structure leakage across train/validation/test")

    for split in ("train", "validation", "test"):
        subset = result[result["split"] == split]
        if subset.empty:
            raise ValueError(
                f"{split} split contains no pattern rows; increase structures/path length without "
                "retuning the frozen detector"
            )
        if subset["profitable"].nunique() < 2:
            raise ValueError(
                f"{split} split has only one target class; increase structures/path length "
                "rather than weakening the detector after seeing outcomes"
            )
    return result, manifest


# ---------------------------------------------------------------------------
# Model preprocessing / models
# ---------------------------------------------------------------------------


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    feature_set = feature_set.lower()
    if feature_set not in {"observable", "oracle"}:
        raise ValueError("feature_set must be 'observable' or 'oracle'")
    numeric = list(OBSERVABLE_NUMERIC_FEATURES)
    categorical = list(OBSERVABLE_CATEGORICAL_FEATURES)
    if feature_set == "oracle":
        numeric.extend(ORACLE_NUMERIC_FEATURES)
        categorical.extend(ORACLE_CATEGORICAL_FEATURES)
    overlap = FUTURE_OR_OUTCOME_COLUMNS.intersection(numeric + categorical)
    if overlap:
        raise AssertionError(f"future/outcome columns leaked into feature schema: {sorted(overlap)}")
    return numeric, categorical


def build_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


class TorchMLPClassifier:
    """Small tabular PyTorch binary classifier with validation-based early stopping."""

    def __init__(
        self,
        input_dim: int,
        *,
        seed: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
    ) -> None:
        # CUDA's deterministic GEMM path requires this setting before the CUDA context is created.
        # It is harmless on CPU-only installations and makes saved research runs more reproducible
        # across repeated executions on the same software/hardware stack.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyTorch is required for the MLP model") from exc
        self.torch = torch
        self.nn = nn
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        # Seed before constructing layers; otherwise initial weights depend on ambient process state.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        # All operations in this small MLP have deterministic implementations.  Failing loudly if
        # a future architectural change introduces a nondeterministic operation is preferable to
        # silently producing irreproducible research artifacts.
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._make_model().to(self.device)
        self.epochs_trained = 0
        self.best_validation_loss = math.inf

    def _make_model(self):
        nn = self.nn
        return nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> "TorchMLPClassifier":
        torch = self.torch
        nn = self.nn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)

        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        x_validation = np.asarray(x_validation, dtype=np.float32)
        y_validation = np.asarray(y_validation, dtype=np.float32)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train), torch.from_numpy(y_train)
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=True,
            generator=generator,
        )

        # The research target is an actual success probability, so preserve the empirical class
        # prior rather than reweighting classes (which would distort probability calibration).
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        x_val = torch.from_numpy(x_validation).to(self.device)
        y_val = torch.from_numpy(y_validation).to(self.device)
        best_state: dict[str, Any] | None = None
        no_improvement = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x).squeeze(1)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_fn(self.model(x_val).squeeze(1), y_val).item()
                )
            self.epochs_trained = epoch
            if validation_loss < self.best_validation_loss - 1e-6:
                self.best_validation_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        values = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(values).to(self.device)).squeeze(1)
            return torch.sigmoid(logits).cpu().numpy().astype(float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "seed": self.seed,
                "epochs_trained": self.epochs_trained,
                "best_validation_loss": self.best_validation_loss,
                "pipeline_version": AI_PIPELINE_VERSION,
            },
            path,
        )


def _require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install xgboost to train the boosted-tree model") from exc
    return XGBClassifier


def _prepare_arrays(
    dataset: pd.DataFrame,
    preprocessor: ColumnTransformer,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> dict[str, Any]:
    features = list(numeric) + list(categorical)
    missing = [name for name in features if name not in dataset.columns]
    if missing:
        raise KeyError(f"AI dataset is missing feature columns: {missing}")
    overlap = FUTURE_OR_OUTCOME_COLUMNS.intersection(features)
    if overlap:
        raise AssertionError(f"future/outcome leakage in feature matrix: {sorted(overlap)}")

    train = dataset[dataset["split"] == "train"].copy()
    validation = dataset[dataset["split"] == "validation"].copy()
    test = dataset[dataset["split"] == "test"].copy()
    x_train = preprocessor.fit_transform(train[features])
    x_validation = preprocessor.transform(validation[features])
    x_test = preprocessor.transform(test[features])
    names = list(preprocessor.get_feature_names_out())
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "x_train": np.asarray(x_train, dtype=np.float32),
        "x_validation": np.asarray(x_validation, dtype=np.float32),
        "x_test": np.asarray(x_test, dtype=np.float32),
        "y_train": train["profitable"].to_numpy(dtype=int),
        "y_validation": validation["profitable"].to_numpy(dtype=int),
        "y_test": test["profitable"].to_numpy(dtype=int),
        "feature_names": names,
    }


# ---------------------------------------------------------------------------
# Prediction / economic evaluation
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lower) & (p <= upper if index == bins - 1 else p < upper)
        count = int(np.sum(mask))
        if not count:
            continue
        error += count / max(len(y), 1) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(error)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) >= 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": expected_calibration_error(y, p),
        "accuracy_at_threshold": float(accuracy_score(y, predicted)),
        "precision_at_threshold": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_threshold": float(recall_score(y, predicted, zero_division=0)),
    }


def _prefixed_validation_metrics(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    run_ids: Sequence[str] | None = None,
    run_structures: Mapping[str, str] | None = None,
) -> dict[str, float]:
    classification = classification_metrics(y_true, probabilities, threshold)
    economics = economic_metrics(
        rows, probabilities, threshold, run_ids=run_ids, run_structures=run_structures
    )
    return {
        "validation_roc_auc": classification["roc_auc"],
        "validation_average_precision": classification["average_precision"],
        "validation_brier_score": classification["brier_score"],
        "validation_expected_calibration_error": classification["expected_calibration_error"],
        "validation_mean_run_net_return_points": economics["mean_run_net_return_points"],
        "validation_selected_trades": economics["selected_trades"],
    }


def _mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Student-t confidence interval across independent inference-unit observations."""

    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    center = float(np.mean(clean))
    if clean.size < 2:
        return center, float("nan"), float("nan")
    sd = float(np.std(clean, ddof=1))
    if sd == 0.0:
        return center, center, center
    sem = sd / math.sqrt(clean.size)
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=int(clean.size - 1)))
    return center, center - critical * sem, center + critical * sem


def _one_sided_positive_t_pvalue(values: Sequence[float]) -> float:
    """One-sided test of a positive mean across independent inference-unit observations."""

    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size < 2:
        return 1.0
    sd = float(np.std(clean, ddof=1))
    center = float(np.mean(clean))
    if sd == 0.0:
        return 0.0 if center > 0.0 else 1.0
    result = stats.ttest_1samp(clean, popmean=0.0, alternative="greater")
    p_value = float(result.pvalue)
    return p_value if math.isfinite(p_value) else 1.0


def _resolve_same_entry_group(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) <= 1:
        return group
    top_probability = float(group["probability"].max())
    tied = group[
        group["probability"].map(
            lambda value: math.isclose(
                float(value),
                top_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    ]
    if tied["trade_direction"].astype(str).nunique() > 1:
        # Equal-scored contradictory information must not be resolved by pattern-name ordering.
        # This matters especially for the all-pattern baseline where every candidate has score 1.
        return group.iloc[0:0]
    # Use the classical backtest's causal geometry/scale order for equally scored signals.
    # A public event hash contains provenance and must not determine which signal is filled.
    ordered_positions = sorted(
        range(len(tied)),
        key=lambda position: causal_signal_priority(
            tied.iloc[position]["pattern_type"],
            tied.iloc[position].get("available_at_index", 0),
            tied.iloc[position].get("pattern_start_index", 0),
            tied.iloc[position].get("pattern_end_index", 0),
            tied.iloc[position].get("observation_window_length", 0),
            tied.iloc[position].get("core_window_length", 0),
            tied.iloc[position].get("geometry_fit_score", 0),
        ),
    )
    return tied.iloc[[ordered_positions[0]]]


def _select_non_overlapping_trades(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(rows):
        raise ValueError("probabilities must be a one-dimensional array matching the row count")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value in [0, 1]")
    # Selection is positional. Duplicate caller indices must not duplicate fills through .loc.
    frame = rows.reset_index(drop=True).copy()
    frame["run_id"] = frame["run_id"].astype(str)
    for column in ("entry_index", "exit_index"):
        indices = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(indices)) or np.any(indices != np.floor(indices)):
            raise ValueError(f"{column} must contain finite integers")
    if bool(((frame["entry_index"] < 0) | (frame["exit_index"] < frame["entry_index"])).any()):
        raise ValueError("trade intervals must satisfy 0 <= entry_index <= exit_index")
    frame["probability"] = values
    candidates = frame[frame["probability"] >= threshold].copy()
    if candidates.empty:
        return candidates

    simultaneous: list[pd.DataFrame] = []
    for (_run_id, _entry), group in candidates.groupby(["run_id", "entry_index"], sort=False):
        resolved = _resolve_same_entry_group(group)
        if not resolved.empty:
            simultaneous.append(resolved)
    if not simultaneous:
        return candidates.iloc[0:0]
    candidates = pd.concat(simultaneous, ignore_index=False).sort_values(
        ["run_id", "entry_index", "probability"],
        ascending=[True, True, False],
    )

    keep: list[int] = []
    for _run_id, group in candidates.groupby("run_id", sort=False):
        open_until = -1
        for index, row in group.iterrows():
            entry = int(row["entry_index"])
            exit_ = int(row["exit_index"])
            if entry <= open_until:
                continue
            keep.append(index)
            open_until = exit_
    return candidates.loc[keep].sort_values(["run_id", "entry_index"])


def _selection_masks(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return threshold-candidate and actually executed masks in original row order."""

    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(rows):
        raise ValueError("probability count must match row count")
    marker = "__selection_row_position__"
    if marker in rows.columns:
        raise ValueError(f"reserved selection column already exists: {marker}")
    positioned = rows.copy()
    positioned[marker] = np.arange(len(positioned), dtype=int)
    selected = _select_non_overlapping_trades(positioned, values, threshold)
    executed = np.zeros(len(positioned), dtype=int)
    if not selected.empty:
        executed[selected[marker].to_numpy(dtype=int)] = 1
    return (values >= threshold).astype(int), executed


def _complete_evaluation_runs(
    rows: pd.DataFrame, run_ids: Sequence[str] | None
) -> list[str]:
    """Validate the denominator before adding zero-trade runs to economic inference."""

    if rows["run_id"].isna().any():
        raise ValueError("economic evaluation contains a missing run_id")
    if "net_directional_return" in rows:
        returns = rows["net_directional_return"].to_numpy(dtype=float)
        if not np.all(np.isfinite(returns)):
            raise ValueError("economic evaluation requires finite trade returns")
    observed = set(rows["run_id"].astype(str))
    if run_ids is None:
        return list(dict.fromkeys(rows["run_id"].astype(str)))
    complete = [str(value) for value in run_ids]
    if len(complete) != len(set(complete)):
        raise ValueError("evaluation run universe contains duplicate run_id values")
    if not observed.issubset(complete):
        raise ValueError("evaluation run universe omits runs present in the pattern rows")
    return complete


def _structure_inference_values(
    rows: pd.DataFrame,
    run_ids: Sequence[str],
    values: Sequence[float],
    run_structures: Mapping[str, str] | None,
) -> list[float]:
    """Average repeated seeds within each sampled structure before computing uncertainty."""

    if run_structures is None and "structure_id" not in rows:
        return list(values)
    observed = (
        rows[["run_id", "structure_id"]].astype(str).drop_duplicates()
        if "structure_id" in rows else pd.DataFrame(columns=["run_id", "structure_id"])
    )
    if observed["run_id"].duplicated().any():
        raise ValueError("one evaluation run maps to multiple structures")
    mapping = dict(zip(observed["run_id"], observed["structure_id"]))
    if run_structures is not None:
        mapping = {str(k): str(v) for k, v in run_structures.items()}
        if any(mapping.get(rid) != sid for rid, sid in observed.itertuples(index=False, name=None)):
            raise ValueError("evaluation run/structure mapping differs from pattern rows")
    if any(rid not in mapping for rid in run_ids):
        raise ValueError("complete run/structure mapping is required for zero-pattern runs")
    frame = pd.DataFrame({"structure": [mapping[rid] for rid in run_ids], "value": values})
    return frame.groupby("structure", sort=True)["value"].mean().tolist()


def economic_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    run_ids: Sequence[str] | None = None,
    run_structures: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Fixed-notional, additive return points; these are not compounded equity returns."""

    complete_run_ids = _complete_evaluation_runs(rows, run_ids)
    if not complete_run_ids:
        raise ValueError("economic evaluation requires at least one run_id")

    evaluation_rows = rows.copy()
    evaluation_rows["run_id"] = evaluation_rows["run_id"].astype(str)
    selected = _select_non_overlapping_trades(evaluation_rows, probabilities, threshold)
    raw_selected = int(np.sum(np.asarray(probabilities, dtype=float) >= threshold))
    if selected.empty:
        units = _structure_inference_values(
            evaluation_rows, complete_run_ids, [0.0] * len(complete_run_ids), run_structures
        )
        mean_total, ci_low, ci_high = _mean_ci(units)
        return {
            "candidate_selected_trades": float(raw_selected),
            "selected_trades": 0.0,
            "selected_runs": 0.0,
            "inference_structures": float(len(units)),
            "selection_rate": 0.0,
            "selected_success_rate": float("nan"),
            "mean_selected_net_return": float("nan"),
            "median_selected_net_return": float("nan"),
            "mean_run_net_return_points": mean_total,
            "mean_run_net_return_points_ci_low": ci_low,
            "mean_run_net_return_points_ci_high": ci_high,
            "median_run_net_return_points": 0.0,
            "mean_run_max_drawdown_return_points": 0.0,
        }

    selected_groups = {run_id: group for run_id, group in selected.groupby("run_id", sort=False)}
    run_totals: list[float] = []
    run_drawdowns: list[float] = []
    # Runs with eligible pattern observations but no AI-selected trade contribute zero.  Excluding
    # them would condition performance on the model deciding to trade and overstate strategy P&L.
    for run_id in complete_run_ids:
        group = selected_groups.get(run_id)
        if group is None or group.empty:
            run_totals.append(0.0)
            run_drawdowns.append(0.0)
            continue
        returns = group.sort_values("entry_index")["net_directional_return"].to_numpy(dtype=float)
        cumulative = np.cumsum(returns)
        running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
        path = np.concatenate(([0.0], cumulative))
        drawdown = running_peak - path
        run_totals.append(float(np.sum(returns)))
        run_drawdowns.append(float(np.max(drawdown)))
    units = _structure_inference_values(evaluation_rows, complete_run_ids, run_totals, run_structures)
    mean_total, ci_low, ci_high = _mean_ci(units)
    return {
        "candidate_selected_trades": float(raw_selected),
        "selected_trades": float(len(selected)),
        "selected_runs": float(selected["run_id"].nunique()),
        "inference_structures": float(len(units)),
        "selection_rate": float(len(selected) / max(len(rows), 1)),
        "selected_success_rate": float(selected["profitable"].mean()),
        "mean_selected_net_return": float(selected["net_directional_return"].mean()),
        "median_selected_net_return": float(selected["net_directional_return"].median()),
        "mean_run_net_return_points": mean_total,
        "mean_run_net_return_points_ci_low": ci_low,
        "mean_run_net_return_points_ci_high": ci_high,
        "median_run_net_return_points": float(np.median(run_totals)),
        "mean_run_max_drawdown_return_points": float(np.mean(run_drawdowns)),
    }


def baseline_metrics(
    rows: pd.DataFrame,
    *,
    run_ids: Sequence[str] | None = None,
    run_structures: Mapping[str, str] | None = None,
) -> dict[str, float]:
    values = economic_metrics(
        rows,
        np.ones(len(rows), dtype=float),
        threshold=0.5,
        run_ids=run_ids,
        run_structures=run_structures,
    )
    return {f"baseline_{key}": value for key, value in values.items()}


def paired_filter_uplift_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    run_ids: Sequence[str] | None = None,
    run_structures: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Paired held-out economic comparison of the AI filter versus the classical baseline.

    The independent unit is a sampled structure, averaging its repeated seeds. A run with no detected pattern, no
    AI-selected trade, or no baseline trade contributes zero where appropriate.  Because the
    model and threshold were selected using validation data, this paired test is meaningful only
    on the untouched test partition; the primary model selected on validation is the pre-specified
    model whose p-value should be interpreted inferentially.
    """

    frame = rows.copy()
    frame["run_id"] = frame["run_id"].astype(str)
    complete_run_ids = _complete_evaluation_runs(frame, run_ids)
    if not complete_run_ids:
        raise ValueError("paired economic comparison requires at least one run_id")

    ai_selected = _select_non_overlapping_trades(frame, probabilities, threshold)
    baseline_selected = _select_non_overlapping_trades(
        frame, np.ones(len(frame), dtype=float), 0.5
    )
    ai_by_run = (
        ai_selected.groupby("run_id")["net_directional_return"].sum().to_dict()
        if not ai_selected.empty
        else {}
    )
    baseline_by_run = (
        baseline_selected.groupby("run_id")["net_directional_return"].sum().to_dict()
        if not baseline_selected.empty
        else {}
    )
    differences = [
        float(ai_by_run.get(run_id, 0.0) - baseline_by_run.get(run_id, 0.0))
        for run_id in complete_run_ids
    ]
    units = _structure_inference_values(frame, complete_run_ids, differences, run_structures)
    mean_uplift, ci_low, ci_high = _mean_ci(units)
    return {
        "paired_ai_minus_baseline_mean_run_return_points": mean_uplift,
        "paired_ai_minus_baseline_return_points_ci_low": ci_low,
        "paired_ai_minus_baseline_return_points_ci_high": ci_high,
        "paired_ai_minus_baseline_one_sided_p_value": _one_sided_positive_t_pvalue(units),
        "paired_ai_outperforms_baseline_run_fraction": float(
            np.mean(np.asarray(differences, dtype=float) > 0.0)
        ),
        "paired_ai_equals_baseline_run_fraction": float(
            np.mean(np.asarray(differences, dtype=float) == 0.0)
        ),
        "paired_evaluation_runs": float(len(differences)),
        "paired_inference_structures": float(len(units)),
    }


def market_family_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    run_universe: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Report structured/null test performance separately to expose trivial null discrimination."""

    frame = rows.copy()
    frame["_probability"] = np.asarray(probabilities, dtype=float)
    output: dict[str, float] = {}
    masks = {
        "structured": frame["world_family"].eq("structured"),
        "null": frame["world_family"].astype(str).str.startswith("null_"),
    }
    for prefix, mask in masks.items():
        subset = frame[mask].copy()
        if run_universe is not None:
            family_mask = (
                run_universe["world_family"].eq("structured")
                if prefix == "structured"
                else run_universe["world_family"].astype(str).str.startswith("null_")
            )
            family_run_ids = run_universe.loc[family_mask, "run_id"].astype(str).tolist()
        else:
            family_run_ids = subset["run_id"].astype(str).drop_duplicates().tolist()
        if subset.empty:
            output[f"{prefix}_test_observations"] = 0.0
            output[f"{prefix}_test_runs"] = float(len(family_run_ids))
            output[f"{prefix}_roc_auc"] = float("nan")
            output[f"{prefix}_mean_run_net_return_points"] = 0.0 if family_run_ids else float("nan")
            output[f"{prefix}_selected_success_rate"] = float("nan")
            continue
        p = subset.pop("_probability").to_numpy(dtype=float)
        y = subset["profitable"].to_numpy(dtype=int)
        output[f"{prefix}_test_observations"] = float(len(subset))
        output[f"{prefix}_test_runs"] = float(len(family_run_ids))
        output[f"{prefix}_roc_auc"] = (
            float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan")
        )
        economics = economic_metrics(
            subset, p, threshold, run_ids=family_run_ids,
            run_structures=(dict(zip(run_universe["run_id"].astype(str),
                                    run_universe["structure_id"].astype(str)))
                            if run_universe is not None else None),
        )
        output[f"{prefix}_mean_run_net_return_points"] = economics["mean_run_net_return_points"]
        output[f"{prefix}_selected_success_rate"] = economics["selected_success_rate"]
    return output


def select_threshold_on_validation(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    experiment: AIExperimentConfig,
    *,
    run_ids: Sequence[str] | None = None,
    run_structures: Mapping[str, str] | None = None,
) -> float:
    all_runs = _complete_evaluation_runs(rows, run_ids)
    # Integer step counts prevent floating-point arange endpoints from exceeding the frozen max.
    count = int(math.floor(
        (experiment.validation_threshold_max - experiment.validation_threshold_min)
        / experiment.validation_threshold_step + 1e-12
    )) + 1
    thresholds = [min(experiment.validation_threshold_max,
                      experiment.validation_threshold_min + i * experiment.validation_threshold_step)
                  for i in range(count)]
    best_threshold = float(experiment.validation_threshold_min)
    best_key = (-math.inf, -1, -math.inf)
    for threshold in thresholds:
        selected = _select_non_overlapping_trades(rows, probabilities, float(threshold))
        if len(selected) < experiment.min_validation_trades:
            continue
        if selected["run_id"].nunique() < experiment.min_validation_runs:
            continue
        selected_totals = selected.groupby("run_id")["net_directional_return"].sum().to_dict()
        run_totals = [float(selected_totals.get(run_id, 0.0)) for run_id in all_runs]
        units = _structure_inference_values(rows, all_runs, run_totals, run_structures)
        score = float(np.mean(units)) if units else -math.inf
        # Pre-registered tie break: prefer more independent runs, then the lower threshold.
        current_key = (score, int(selected["run_id"].nunique()), -float(threshold))
        if current_key > best_key:
            best_key = current_key
            best_threshold = float(threshold)
    if not math.isfinite(best_key[0]):
        raise ValueError(
            "INSUFFICIENT_VALIDATION_SUPPORT: no threshold met the pre-registered minimum "
            "trade/run counts. Confirmatory test evaluation is disabled."
        )
    return best_threshold


def save_calibration_table(y_true: np.ndarray, p: np.ndarray, path: Path) -> None:
    observed, predicted = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_success_rate": observed,
        }
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def save_logistic_importance(model: LogisticRegression, names: Sequence[str], path: Path) -> None:
    coefficients = np.asarray(model.coef_).reshape(-1)
    pd.DataFrame(
        {
            "feature": list(names),
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False).to_csv(path, index=False)


def save_xgboost_importance(model: Any, names: Sequence[str], path: Path) -> None:
    values = np.asarray(model.feature_importances_, dtype=float)
    pd.DataFrame(
        {"feature": list(names), "feature_importance": values}
    ).sort_values("feature_importance", ascending=False).to_csv(path, index=False)


def save_xgboost_shap(
    model: Any,
    x_test: np.ndarray,
    names: Sequence[str],
    path: Path,
    *,
    seed: int,
    max_rows: int = 5_000,
) -> None:
    try:
        import shap
    except ImportError:  # pragma: no cover
        warnings.warn("SHAP is unavailable; skipping XGBoost SHAP output", RuntimeWarning)
        return
    x = np.asarray(x_test)
    if len(x) > max_rows:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), size=max_rows, replace=False)]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.shape[1] != len(names):
        raise RuntimeError("SHAP feature count does not match transformed feature names")
    pd.DataFrame(
        {
            "feature": list(names),
            "mean_abs_shap": np.mean(np.abs(array), axis=0),
            "mean_signed_shap": np.mean(array, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).to_csv(path, index=False)


def save_oracle_condition_response(
    test_rows: pd.DataFrame,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    """Descriptive held-out response curves for true synthetic mechanism features.

    The table contains both an all-pattern view and pattern-specific views when the held-out sample
    is large enough.  This is intentionally descriptive: correlated synthetic mechanisms can share
    predictive information, so these response curves and SHAP values are not causal estimands.
    """

    frame = test_rows.copy()
    frame["probability"] = np.asarray(probabilities, dtype=float)
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("__all_patterns__", frame)]
    scopes.extend((str(name), group.copy()) for name, group in frame.groupby("pattern_type", sort=True))

    for pattern_scope, scope in scopes:
        minimum_rows = 20 if pattern_scope == "__all_patterns__" else 30
        if len(scope) < minimum_rows:
            continue
        for feature in ORACLE_NUMERIC_FEATURES:
            values = pd.to_numeric(scope[feature], errors="coerce")
            valid = scope[values.notna()].copy()
            valid[feature] = values[values.notna()]
            if len(valid) < minimum_rows or valid[feature].nunique() < 3:
                continue
            try:
                valid["condition_bin"] = pd.qcut(valid[feature], q=5, duplicates="drop")
            except ValueError:
                continue
            for condition_bin, group in valid.groupby("condition_bin", observed=True):
                # Collapse repeated seeds within each sampled structure before estimating
                # uncertainty; event-weighted columns remain descriptive only.
                run_summary = group.groupby(["structure_id", "run_id"], sort=False)[
                    ["profitable", "net_directional_return", "probability"]
                ].mean().groupby("structure_id", sort=False).mean()
                run_success = run_summary["profitable"].to_numpy(dtype=float)
                run_net = run_summary["net_directional_return"].to_numpy(dtype=float)
                run_probability = run_summary["probability"].to_numpy(dtype=float)
                mean_run_success, success_lo, success_hi = _mean_ci(run_success)
                mean_run_net, net_lo, net_hi = _mean_ci(run_net)
                output.append(
                    {
                        "pattern_scope": pattern_scope,
                        "feature": feature,
                        "condition_bin": str(condition_bin),
                        "observations": int(len(group)),
                        "runs": int(group["run_id"].nunique()),
                        "structures": int(group["structure_id"].nunique()),
                        "mean_feature_value": float(group[feature].mean()),
                        "actual_success_rate_event_weighted": float(group["profitable"].mean()),
                        "mean_net_directional_return_event_weighted": float(group["net_directional_return"].mean()),
                        "mean_predicted_probability_event_weighted": float(group["probability"].mean()),
                        "mean_run_success_rate": mean_run_success,
                        "mean_run_success_rate_ci_low": success_lo,
                        "mean_run_success_rate_ci_high": success_hi,
                        "mean_run_net_directional_return": mean_run_net,
                        "mean_run_net_directional_return_ci_low": net_lo,
                        "mean_run_net_directional_return_ci_high": net_hi,
                        "mean_run_predicted_probability": float(np.mean(run_probability)),
                    }
                )
    pd.DataFrame(output).to_csv(path, index=False)


def save_per_pattern_test_metrics(
    test_rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    path: Path,
    *,
    run_ids: Sequence[str] | None = None,
    run_structures: Mapping[str, str] | None = None,
) -> None:
    frame = test_rows.copy()
    frame["probability"] = np.asarray(probabilities, dtype=float)
    rows: list[dict[str, Any]] = []
    for pattern, group in frame.groupby("pattern_type", sort=True):
        p = group["probability"].to_numpy(dtype=float)
        y = group["profitable"].to_numpy(dtype=int)
        rows.append(
            {
                "pattern_type": pattern,
                "observations": int(len(group)),
                "base_success_rate": float(group["profitable"].mean()),
                **classification_metrics(y, p, threshold),
                **economic_metrics(group, p, threshold, run_ids=run_ids, run_structures=run_structures),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def train_and_evaluate_models(
    dataset: pd.DataFrame,
    experiment: AIExperimentConfig,
    output_dir: Path,
    *,
    feature_set: str,
    run_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric, categorical = feature_columns(feature_set)
    preprocessor = build_preprocessor(numeric, categorical)
    arrays = _prepare_arrays(dataset, preprocessor, numeric, categorical)
    joblib.dump(preprocessor, output_dir / f"preprocessor_{feature_set}.joblib")

    x_train = arrays["x_train"]
    x_validation = arrays["x_validation"]
    x_test = arrays["x_test"]
    y_train = arrays["y_train"]
    y_validation = arrays["y_validation"]
    y_test = arrays["y_test"]
    feature_names = arrays["feature_names"]
    if len(np.unique(y_train)) < 2:
        raise ValueError("training target has one class; increase structures/path length")

    if run_universe is not None:
        required = {"run_id", "structure_id", "world_family", "split"}
        missing = required.difference(run_universe.columns)
        if missing:
            raise KeyError(f"run universe missing evaluation columns: {sorted(missing)}")
        validation_runs = run_universe[run_universe["split"] == "validation"].copy()
        test_runs = run_universe[run_universe["split"] == "test"].copy()
        validation_run_ids = validation_runs["run_id"].astype(str).tolist()
        test_run_ids = test_runs["run_id"].astype(str).tolist()
    else:
        validation_runs = None
        test_runs = None
        validation_run_ids = arrays["validation"]["run_id"].astype(str).drop_duplicates().tolist()
        test_run_ids = arrays["test"]["run_id"].astype(str).drop_duplicates().tolist()

    validation_run_structures = dict(zip(
        (validation_runs if validation_runs is not None else arrays["validation"])["run_id"].astype(str),
        (validation_runs if validation_runs is not None else arrays["validation"])["structure_id"].astype(str),
    ))
    test_run_structures = dict(zip(
        (test_runs if test_runs is not None else arrays["test"])["run_id"].astype(str),
        (test_runs if test_runs is not None else arrays["test"])["structure_id"].astype(str),
    ))

    prediction_columns = [
        "simulation_id",
        "structure_id",
        "world_family",
        "run_id",
        "event_id",
        "pattern_type",
        "trade_direction",
        "available_at_index",
        "entry_index",
        "exit_index",
        "profitable",
        "gross_directional_return",
        "total_transaction_cost",
        "net_directional_return",
    ]
    predictions = arrays["test"][prediction_columns].copy()
    results: list[dict[str, Any]] = []

    # Logistic regression: interpretable linear baseline.
    logistic = LogisticRegression(
        max_iter=2_000,
        class_weight=None,
        random_state=experiment.base_seed,
        solver="lbfgs",
    )
    logistic.fit(x_train, y_train)
    joblib.dump(logistic, output_dir / f"logistic_regression_{feature_set}.joblib")
    save_logistic_importance(
        logistic,
        feature_names,
        output_dir / f"logistic_feature_importance_{feature_set}.csv",
    )
    log_val = logistic.predict_proba(x_validation)[:, 1]
    log_test = logistic.predict_proba(x_test)[:, 1]
    log_threshold = select_threshold_on_validation(
        arrays["validation"],
        log_val,
        experiment,
        run_ids=validation_run_ids, run_structures=validation_run_structures,
    )
    predictions["logistic_probability"] = log_test
    logistic_candidates, logistic_executed = _selection_masks(
        arrays["test"], log_test, log_threshold
    )
    predictions["logistic_candidate_selected"] = logistic_candidates
    predictions["logistic_selected"] = logistic_executed
    save_calibration_table(y_test, log_test, output_dir / f"logistic_calibration_{feature_set}.csv")
    results.append(
        {
            "model": "Logistic Regression",
            "feature_set": feature_set,
            "validation_selected_threshold": log_threshold,
            **_prefixed_validation_metrics(
                arrays["validation"],
                y_validation,
                log_val,
                log_threshold,
                run_ids=validation_run_ids, run_structures=validation_run_structures,
            ),
            **classification_metrics(y_test, log_test, log_threshold),
            **economic_metrics(
                arrays["test"], log_test, log_threshold, run_ids=test_run_ids, run_structures=test_run_structures
            ),
            **baseline_metrics(arrays["test"], run_ids=test_run_ids, run_structures=test_run_structures),
            **paired_filter_uplift_metrics(
                arrays["test"], log_test, log_threshold, run_ids=test_run_ids, run_structures=test_run_structures
            ),
            **market_family_metrics(
                arrays["test"],
                log_test,
                log_threshold,
                run_universe=test_runs,
            ),
        }
    )
    save_per_pattern_test_metrics(
        arrays["test"],
        log_test,
        log_threshold,
        output_dir / f"logistic_per_pattern_test_{feature_set}.csv",
        run_ids=test_run_ids, run_structures=test_run_structures,
    )

    # XGBoost: nonlinear interaction model, usually best suited to this tabular experiment.
    XGBClassifier = _require_xgboost()
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=40,
        random_state=experiment.base_seed,
        n_jobs=max(1, min(os.cpu_count() or 1, 8)),
    )
    xgb.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    xgb.save_model(output_dir / f"xgboost_{feature_set}.json")
    save_xgboost_importance(
        xgb, feature_names, output_dir / f"xgboost_feature_importance_{feature_set}.csv"
    )
    xgb_val = xgb.predict_proba(x_validation)[:, 1]
    xgb_test = xgb.predict_proba(x_test)[:, 1]
    xgb_threshold = select_threshold_on_validation(
        arrays["validation"],
        xgb_val,
        experiment,
        run_ids=validation_run_ids, run_structures=validation_run_structures,
    )
    predictions["xgboost_probability"] = xgb_test
    xgboost_candidates, xgboost_executed = _selection_masks(
        arrays["test"], xgb_test, xgb_threshold
    )
    predictions["xgboost_candidate_selected"] = xgboost_candidates
    predictions["xgboost_selected"] = xgboost_executed
    save_calibration_table(y_test, xgb_test, output_dir / f"xgboost_calibration_{feature_set}.csv")
    save_xgboost_shap(
        xgb,
        x_test,
        feature_names,
        output_dir / f"xgboost_shap_{feature_set}.csv",
        seed=experiment.base_seed,
    )
    results.append(
        {
            "model": "XGBoost",
            "feature_set": feature_set,
            "validation_selected_threshold": xgb_threshold,
            **_prefixed_validation_metrics(
                arrays["validation"],
                y_validation,
                xgb_val,
                xgb_threshold,
                run_ids=validation_run_ids, run_structures=validation_run_structures,
            ),
            **classification_metrics(y_test, xgb_test, xgb_threshold),
            **economic_metrics(
                arrays["test"], xgb_test, xgb_threshold, run_ids=test_run_ids, run_structures=test_run_structures
            ),
            **baseline_metrics(arrays["test"], run_ids=test_run_ids, run_structures=test_run_structures),
            **paired_filter_uplift_metrics(
                arrays["test"], xgb_test, xgb_threshold, run_ids=test_run_ids, run_structures=test_run_structures
            ),
            **market_family_metrics(
                arrays["test"],
                xgb_test,
                xgb_threshold,
                run_universe=test_runs,
            ),
        }
    )
    save_per_pattern_test_metrics(
        arrays["test"],
        xgb_test,
        xgb_threshold,
        output_dir / f"xgboost_per_pattern_test_{feature_set}.csv",
        run_ids=test_run_ids, run_structures=test_run_structures,
    )
    if feature_set == "oracle":
        save_oracle_condition_response(
            arrays["test"],
            xgb_test,
            output_dir / "xgboost_oracle_condition_response.csv",
        )

    # Small MLP: neural-network comparator, not assumed to be superior.
    mlp = TorchMLPClassifier(
        x_train.shape[1],
        seed=experiment.base_seed,
        epochs=experiment.mlp_epochs,
        batch_size=experiment.mlp_batch_size,
        learning_rate=experiment.mlp_learning_rate,
        patience=experiment.mlp_patience,
    )
    mlp.fit(x_train, y_train, x_validation, y_validation)
    mlp.save(output_dir / f"pytorch_mlp_{feature_set}.pt")
    mlp_val = mlp.predict_proba(x_validation)
    mlp_test = mlp.predict_proba(x_test)
    mlp_threshold = select_threshold_on_validation(
        arrays["validation"],
        mlp_val,
        experiment,
        run_ids=validation_run_ids, run_structures=validation_run_structures,
    )
    predictions["mlp_probability"] = mlp_test
    mlp_candidates, mlp_executed = _selection_masks(
        arrays["test"], mlp_test, mlp_threshold
    )
    predictions["mlp_candidate_selected"] = mlp_candidates
    predictions["mlp_selected"] = mlp_executed
    save_calibration_table(y_test, mlp_test, output_dir / f"mlp_calibration_{feature_set}.csv")
    results.append(
        {
            "model": "PyTorch MLP",
            "feature_set": feature_set,
            "validation_selected_threshold": mlp_threshold,
            **_prefixed_validation_metrics(
                arrays["validation"],
                y_validation,
                mlp_val,
                mlp_threshold,
                run_ids=validation_run_ids, run_structures=validation_run_structures,
            ),
            "mlp_epochs_trained": float(mlp.epochs_trained),
            "mlp_best_validation_loss": float(mlp.best_validation_loss),
            **classification_metrics(y_test, mlp_test, mlp_threshold),
            **economic_metrics(
                arrays["test"], mlp_test, mlp_threshold, run_ids=test_run_ids, run_structures=test_run_structures
            ),
            **baseline_metrics(arrays["test"], run_ids=test_run_ids, run_structures=test_run_structures),
            **paired_filter_uplift_metrics(
                arrays["test"], mlp_test, mlp_threshold, run_ids=test_run_ids, run_structures=test_run_structures
            ),
            **market_family_metrics(
                arrays["test"],
                mlp_test,
                mlp_threshold,
                run_universe=test_runs,
            ),
        }
    )
    save_per_pattern_test_metrics(
        arrays["test"],
        mlp_test,
        mlp_threshold,
        output_dir / f"mlp_per_pattern_test_{feature_set}.csv",
        run_ids=test_run_ids, run_structures=test_run_structures,
    )

    predictions.to_csv(output_dir / f"test_predictions_{feature_set}.csv", index=False)
    comparison = pd.DataFrame(results)
    # The primary model is selected strictly from validation results before interpreting test P&L.
    # All three test results remain visible because the model comparison itself was pre-registered.
    ranking = comparison.copy()
    ranking["_auc_rank"] = ranking["validation_roc_auc"].fillna(-math.inf)
    ranking["_brier_rank"] = -ranking["validation_brier_score"].fillna(math.inf)
    ranking["_econ_rank"] = ranking["validation_mean_run_net_return_points"].fillna(-math.inf)
    ranking = ranking.sort_values(
        ["_auc_rank", "_brier_rank", "_econ_rank", "model"],
        ascending=[False, False, False, True],
    )
    primary_model = str(ranking.iloc[0]["model"])
    comparison["primary_model_selected_on_validation"] = comparison["model"].eq(primary_model)
    comparison["test_inference_role"] = np.where(
        comparison["primary_model_selected_on_validation"], "primary_within_feature_set", "exploratory_comparator"
    )
    # Observable and oracle are two pre-specified primary-model families. Reserve both slots even
    # when the CLI runs only one, so choosing which feature set to show cannot evade correction.
    comparison["confirmatory_p_value_bonferroni"] = np.where(
        comparison["primary_model_selected_on_validation"],
        np.minimum(1.0, 2.0 * comparison["paired_ai_minus_baseline_one_sided_p_value"]),
        np.nan,
    )
    comparison.to_csv(output_dir / f"model_comparison_{feature_set}.csv", index=False)
    (output_dir / f"primary_model_{feature_set}.json").write_text(
        json.dumps(
            {
                "feature_set": feature_set,
                "primary_model": primary_model,
                "selection_rule": (
                    "highest validation ROC-AUC; tie -> lowest validation Brier score; "
                    "tie -> highest validation mean run additive net return points; tie -> model name"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return comparison


# ---------------------------------------------------------------------------
# Diagnostics / manifest
# ---------------------------------------------------------------------------


def research_procedure_contract(
    experiment: AIExperimentConfig, costs: CostModel
) -> dict[str, Any]:
    """An explicit AI procedure, separate from the statistical matched-null estimand."""

    return {
        "study_role": "independent_sampled_mechanism_discovery",
        "universe": "sampled_world_subsets_weights_and_parameters",
        "headline_comparable_to_statistical_scenario_grid": False,
        "candle_steps": experiment.candle_steps,
        "label_horizon_candles": experiment.label_horizon_candles,
        "execution": "synthetic_quote_mid_and_spread_at_open_and_close",
        "overlap_policy": "skip_new",
        "position_scope": "all_pattern_types_within_each_run",
        "simultaneous_signal_rule": "highest_probability; abstain_on_equal_score_opposite_directions",
        "matched_null_rules": "not_used; paired_AI_vs_all_pattern_portfolio_baseline",
        "cost_model": asdict(costs),
        "return_units": "additive_fixed_notional_return_points_not_compounded_portfolio_return",
        "validation_support_policy": "fail_closed",
    }


def validate_reference_procedure_manifest(
    path: Path, experiment: AIExperimentConfig, costs: CostModel
) -> dict[str, Any]:
    """Verify shared execution settings without claiming identical experimental universes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("reference procedure manifest must be a JSON object")
    procedure = _procedure_from_manifest(payload)
    frozen_hash = payload.get("procedure_hash")
    if frozen_hash != procedure_hash(procedure):
        raise ValueError("reference procedure_hash does not match the current frozen research stack")
    mismatches = {}
    if experiment.candle_steps != procedure.steps_per_candle:
        mismatches["candle_steps"] = (experiment.candle_steps, procedure.steps_per_candle)
    if experiment.label_horizon_candles not in procedure.horizons:
        mismatches["horizon"] = (experiment.label_horizon_candles, procedure.horizons)
    if procedure.overlap_policy != "skip_new":
        mismatches["overlap_policy"] = ("skip_new", procedure.overlap_policy)
    if costs != procedure.cost_model:
        mismatches["cost_model"] = (asdict(costs), asdict(procedure.cost_model))
    if mismatches:
        raise ValueError(f"AI/reference procedure settings differ: {mismatches}")
    return {
        "procedure_hash": frozen_hash,
        "manifest_sha256": _file_sha256(path),
        "procedure": asdict(procedure),
        "comparison_scope": "shared_execution_settings_only",
        "headline_comparable": False,
        "remaining_design_differences": [
            "sampled AI structures versus statistical scenario manifest",
            "global all-pattern AI portfolio versus statistical per-pattern portfolios",
            "paired AI-filter baseline versus statistical matched-null inference",
        ],
    }


def save_dataset_diagnostics(
    dataset: pd.DataFrame,
    output_dir: Path,
    runs: pd.DataFrame | None = None,
) -> None:
    by_pattern = (
        dataset.groupby("pattern_type")
        .agg(
            observations=("profitable", "size"),
            structures=("structure_id", "nunique"),
            runs=("run_id", "nunique"),
            success_rate=("profitable", "mean"),
            mean_net_return=("net_directional_return", "mean"),
        )
        .reset_index()
        .sort_values("observations", ascending=False)
    )
    by_world = (
        dataset.groupby("world_family")
        .agg(
            observations=("profitable", "size"),
            structures=("structure_id", "nunique"),
            runs=("run_id", "nunique"),
            success_rate=("profitable", "mean"),
            mean_net_return=("net_directional_return", "mean"),
        )
        .reset_index()
    )
    by_pattern.to_csv(output_dir / "dataset_by_pattern.csv", index=False)
    by_world.to_csv(output_dir / "dataset_by_world_family.csv", index=False)
    payload = {
        "rows": int(len(dataset)),
        "structures": int(dataset["structure_id"].nunique()),
        "runs": int(dataset["run_id"].nunique()),
        "overall_success_rate": float(dataset["profitable"].mean()),
        "overall_mean_net_return": float(dataset["net_directional_return"].mean()),
    }
    if runs is not None and not runs.empty:
        pattern_run_ids = set(dataset["run_id"].astype(str))
        pattern_structure_ids = set(dataset["structure_id"].astype(str))
        payload.update(
            {
                "generated_runs": int(runs["run_id"].nunique()),
                "generated_structures": int(runs["structure_id"].nunique()),
                "runs_with_eligible_patterns": int(
                    runs["run_id"].astype(str).isin(pattern_run_ids).sum()
                ),
                "zero_pattern_runs": int(
                    (~runs["run_id"].astype(str).isin(pattern_run_ids)).sum()
                ),
                "structures_with_eligible_patterns": int(
                    runs["structure_id"].astype(str).isin(pattern_structure_ids).groupby(
                        runs["structure_id"].astype(str)
                    ).any().sum()
                ),
            }
        )
        coverage = runs[["run_id", "structure_id", "world_family"]].copy()
        coverage["has_eligible_pattern"] = coverage["run_id"].astype(str).isin(pattern_run_ids)
        coverage_summary = (
            coverage.groupby("world_family", sort=True)
            .agg(
                generated_runs=("run_id", "nunique"),
                structures=("structure_id", "nunique"),
                runs_with_eligible_patterns=("has_eligible_pattern", "sum"),
                pattern_run_fraction=("has_eligible_pattern", "mean"),
            )
            .reset_index()
        )
        coverage_summary.to_csv(
            output_dir / "run_pattern_coverage_by_world_family.csv",
            index=False,
        )
    (output_dir / "dataset_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_research_manifest(
    output_dir: Path,
    experiment: AIExperimentConfig,
    ranges: WorldSamplingRanges,
    costs: CostModel,
    *,
    git_commit: str | None,
    reference_procedure_manifest: Path | None = None,
) -> None:
    empty_detection = detect_pattern_universe([], DEFAULT_PATTERN_CONFIG)
    files = {
        "ai_pipeline": _module_path(write_research_manifest),
        "simulator": _module_path(SyntheticMarketConfig),
        "candle_builder": _module_path(build_candles),
        "pattern_detector": _module_path(detect_pattern_universe),
        "experiment_runner": _module_path(net_trade_return),
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "ai_pipeline_version": AI_PIPELINE_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "candle_builder_version": CANDLE_BUILDER_VERSION,
        "detector_version": DETECTOR_VERSION,
        "detector_config_hash": empty_detection.config_hash,
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
        "experiment": asdict(experiment),
        "world_sampling_ranges": asdict(ranges),
        "cost_model": asdict(costs),
        "ai_procedure_contract": research_procedure_contract(experiment, costs),
        "reference_statistical_procedure": (
            validate_reference_procedure_manifest(reference_procedure_manifest, experiment, costs)
            if reference_procedure_manifest is not None else None
        ),
        "observable_numeric_features": OBSERVABLE_NUMERIC_FEATURES,
        "oracle_numeric_features": ORACLE_NUMERIC_FEATURES,
        "observable_categorical_features": OBSERVABLE_CATEGORICAL_FEATURES,
        "oracle_categorical_features": ORACLE_CATEGORICAL_FEATURES,
        "module_sha256": {name: _file_sha256(path) for name, path in files.items()},
        "module_paths": {name: str(path) for name, path in files.items()},
        "split_unit": "complete synthetic structure (all repeated seeds)",
        "economic_evaluation_unit": (
            "all generated runs in the assigned partition, including runs with zero eligible "
            "pattern occurrences (zero strategy return)"
        ),
        "economic_inference_unit": "sampled structure; average repeated-seed totals within structure",
        "confirmatory_family": "two validation-selected primary models: observable and oracle; Bonferroni size 2",
        "model_selection_rule": (
            "highest validation ROC-AUC; tie -> lowest validation Brier score; "
            "tie -> highest validation mean run additive net return points; tie -> model name"
        ),
        "ai_incremental_value_test": (
            "paired independent-structure test of test-set AI-filter additive net return points minus the "
            "all-pattern classical baseline; inferential interpretation is reserved for the "
            "primary model selected on validation"
        ),
        "label_semantics": (
            "net directional return > 0 after simulated spread + pre-registered brokerage/slippage; "
            "entry at detector earliest_execution_index, exit close/reference at entry+horizon-1"
        ),
        "future_excursion_semantics": (
            "transaction OHLC extremes relative to entry quote mid, including zero at entry; "
            "descriptive only, not executable quote-path MAE/MFE"
        ),
        "explainability_caveat": (
            "SHAP and condition-response outputs identify predictive associations in the sampled "
            "synthetic design; controlled counterfactual experiments are required for causal claims"
        ),
        # Filled atomically only after both raw dataset artifacts have been built successfully.
        "dataset_artifacts": None,
    }
    (output_dir / "ai_research_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def finalize_research_manifest(output_dir: Path) -> None:
    """Bind the frozen code/design contract to the exact raw dataset artifacts.

    The raw pattern dataset is intentionally immutable after this point.  Train/validation/test
    assignments are written to a separate file so a later training run cannot silently rewrite the
    data whose SHA-256 was frozen at generation time.
    """

    manifest_path = output_dir / "ai_research_manifest.json"
    runs_path = output_dir / "synthetic_runs.csv"
    if not manifest_path.exists() or not runs_path.exists():
        raise FileNotFoundError("cannot finalize AI manifest before raw dataset artifacts exist")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("AI research manifest must be a JSON object")
    dataset_name = _raw_dataset_filename(payload)
    dataset_path = output_dir / dataset_name
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset_rows = max(0, sum(1 for _ in handle) - 1)
    with runs_path.open("r", encoding="utf-8") as handle:
        run_rows = max(0, sum(1 for _ in handle) - 1)
    payload["dataset_artifacts"] = {
        dataset_name: {
            "sha256": _file_sha256(dataset_path),
            "rows": dataset_rows,
            "bytes": dataset_path.stat().st_size,
        },
        "synthetic_runs.csv": {
            "sha256": _file_sha256(runs_path),
            "rows": run_rows,
            "bytes": runs_path.stat().st_size,
        },
    }
    payload["dataset_finalized_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)



def _raw_dataset_filename(manifest: Mapping[str, Any]) -> str:
    name = manifest.get("raw_dataset_filename", "pattern_training_dataset.csv")
    if name not in {"pattern_training_dataset.csv", "synthetic_pattern_dataset.csv"}:
        raise ValueError("unsupported raw dataset filename in AI manifest")
    return name


def load_frozen_research_manifest(
    path: Path,
) -> tuple[AIExperimentConfig, WorldSamplingRanges, CostModel, Mapping[str, Any]]:
    """Load the exact dataset-generation/model-selection contract for a train-only run."""

    if not path.exists():
        raise FileNotFoundError(
            f"frozen AI research manifest not found at {path}; build the dataset in this output "
            "directory before running the train command"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("AI research manifest must be a JSON object")
    try:
        experiment = AIExperimentConfig(**dict(payload["experiment"]))
        range_payload = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in dict(payload["world_sampling_ranges"]).items()
        }
        ranges = WorldSamplingRanges(**range_payload)
        costs = CostModel(**dict(payload["cost_model"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("AI research manifest is incomplete or incompatible") from exc

    if payload.get("ai_procedure_contract") != research_procedure_contract(experiment, costs):
        raise ValueError("AI procedure contract differs from the frozen dataset settings")
    reference = payload.get("reference_statistical_procedure")
    if reference is not None:
        if not isinstance(reference, Mapping):
            raise ValueError("frozen reference statistical procedure must be an object")
        procedure = _procedure_from_manifest(reference)
        if reference.get("procedure_hash") != procedure_hash(procedure):
            raise ValueError("frozen reference statistical procedure_hash no longer matches")
        if (procedure.steps_per_candle != experiment.candle_steps
                or experiment.label_horizon_candles not in procedure.horizons
                or procedure.overlap_policy != "skip_new" or procedure.cost_model != costs):
            raise ValueError("frozen reference statistical procedure settings differ from AI")

    expected_versions = {
        "ai_pipeline_version": AI_PIPELINE_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "candle_builder_version": CANDLE_BUILDER_VERSION,
        "detector_version": DETECTOR_VERSION,
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_versions.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "frozen AI manifest version mismatch; do not train a stored dataset with a different "
            f"research stack: {mismatches}"
        )
    current_detector_hash = detect_pattern_universe([], DEFAULT_PATTERN_CONFIG).config_hash
    if payload.get("detector_config_hash") != current_detector_hash:
        raise ValueError("frozen detector configuration differs from the stored AI dataset contract")

    current_files = {
        "ai_pipeline": _module_path(write_research_manifest),
        "simulator": _module_path(SyntheticMarketConfig),
        "candle_builder": _module_path(build_candles),
        "pattern_detector": _module_path(detect_pattern_universe),
        "experiment_runner": _module_path(net_trade_return),
    }
    stored_hashes = payload.get("module_sha256")
    if not isinstance(stored_hashes, Mapping):
        raise ValueError("frozen AI manifest is missing module SHA-256 fingerprints")
    if "dataset_generator" in stored_hashes:
        current_files["dataset_generator"] = Path(__file__).with_name("ai_pattern_dataset_generator.py")
    hash_mismatches = {
        name: (stored_hashes.get(name), _file_sha256(path))
        for name, path in current_files.items()
        if stored_hashes.get(name) != _file_sha256(path)
    }
    if hash_mismatches:
        raise ValueError(
            "research code changed after the dataset manifest was frozen; rebuild the dataset "
            f"instead of silently retraining under different code: {hash_mismatches}"
        )

    artifacts = payload.get("dataset_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(
            "frozen AI manifest has no finalized dataset fingerprints; rebuild the dataset before "
            "running train-only mode"
        )
    for filename in (_raw_dataset_filename(payload), "synthetic_runs.csv"):
        record = artifacts.get(filename)
        if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"frozen AI manifest is missing the fingerprint for {filename}")
        artifact_path = path.parent / filename
        if not artifact_path.exists():
            raise FileNotFoundError(f"frozen AI dataset artifact is missing: {artifact_path}")
        observed_hash = _file_sha256(artifact_path)
        if observed_hash != record["sha256"]:
            raise ValueError(
                f"frozen AI dataset artifact changed after generation: {filename}; rebuild rather "
                "than training on an untracked mutation"
            )
    return experiment, ranges, costs, payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, train, and test AI models that estimate when causal classical chart patterns "
            "are profitable in the frozen synthetic market research stack."
        )
    )
    parser.add_argument("command", choices=("build-dataset", "train", "all"), nargs="?", default="all")
    parser.add_argument("--structures", type=int, default=1_000)
    parser.add_argument("--seeds-per-structure", type=int, default=1)
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Legacy alias: sets --structures=N and --seeds-per-structure=1.",
    )
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--candle-steps", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--market-lookback", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-world-fraction", type=float, default=0.10)
    parser.add_argument("--efficient-null-share", type=float, default=0.50)
    parser.add_argument("--min-active-worlds", type=int, default=1)
    parser.add_argument("--max-active-worlds", type=int, default=4)
    parser.add_argument("--random-background-probability", type=float, default=0.45)
    parser.add_argument("--world-weight-concentration", type=float, default=1.0)
    parser.add_argument("--minimum-world-weight", type=float, default=0.04)
    parser.add_argument("--brokerage-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--min-validation-trades", type=int, default=25)
    parser.add_argument("--min-validation-runs", type=int, default=5)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    parser.add_argument("--mlp-batch-size", type=int, default=512)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-patience", type=int, default=15)
    parser.add_argument("--jobs", type=int, default=1,
                        help="Generation workers: Atlas supports 1 or 2; -1 uses all logical CPUs only in legacy mode.")
    parser.add_argument("--feature-set", choices=("observable", "oracle", "both"), default="both")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Persist completed simulations and reuse them when rerunning the same command/directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/Atlas/atlas_1000000x2"))
    parser.add_argument("--git-commit", default=None)
    parser.add_argument(
        "--reference-procedure-manifest", type=Path, default=None,
        help="Verify shared execution/cost settings against a statistical procedure manifest; "
             "the sampled AI universe and portfolio baseline remain a separate experiment.",
    )
    from atlas_pipeline import add_streaming_arguments
    return add_streaming_arguments(parser)


def experiment_from_args(args: argparse.Namespace) -> AIExperimentConfig:
    structures = args.structures
    seeds_per_structure = args.seeds_per_structure
    if args.simulations is not None:
        if args.simulations <= 0:
            raise ValueError("--simulations must be > 0")
        if args.seeds_per_structure != 1:
            raise ValueError("--simulations cannot be combined with --seeds-per-structure != 1")
        structures = args.simulations
        seeds_per_structure = 1
    return AIExperimentConfig(
        structures=structures,
        seeds_per_structure=seeds_per_structure,
        steps_per_run=args.steps,
        candle_steps=args.candle_steps,
        label_horizon_candles=args.horizon,
        market_feature_lookback_candles=args.market_lookback,
        base_seed=args.seed,
        null_world_fraction=args.null_world_fraction,
        efficient_null_share=args.efficient_null_share,
        min_active_worlds=args.min_active_worlds,
        max_active_worlds=args.max_active_worlds,
        random_background_probability=args.random_background_probability,
        world_weight_concentration=args.world_weight_concentration,
        minimum_world_weight=args.minimum_world_weight,
        validation_threshold_min=args.threshold_min,
        validation_threshold_max=args.threshold_max,
        validation_threshold_step=args.threshold_step,
        min_validation_trades=args.min_validation_trades,
        min_validation_runs=args.min_validation_runs,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_learning_rate=args.mlp_learning_rate,
        mlp_patience=args.mlp_patience,
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found at {path}; run build-dataset first")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"dataset at {path} is empty")
    return frame


def _run_cli_impl(args: argparse.Namespace) -> None:
    if not getattr(args, "legacy_in_memory", False):
        from atlas_pipeline import run_streaming_cli
        run_streaming_cli(args)
        return
    if args.structures * args.seeds_per_structure > 10000 or (args.simulations or 0) > 10000:
        raise ValueError("Large experiments require the default streaming pipeline")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ai_research_manifest.json"
    dataset_path = output_dir / "pattern_training_dataset.csv"

    if args.command == "train":
        experiment, ranges, costs, _manifest = load_frozen_research_manifest(manifest_path)
        dataset_path = output_dir / _raw_dataset_filename(_manifest)
        if args.reference_procedure_manifest is not None:
            reference = validate_reference_procedure_manifest(args.reference_procedure_manifest, experiment, costs)
            frozen_reference = _manifest.get("reference_statistical_procedure")
            if (frozen_reference is not None
                    and reference["procedure_hash"] != frozen_reference["procedure_hash"]):
                raise ValueError("training reference procedure differs from the frozen dataset reference")
            (output_dir / "training_reference_procedure.json").write_text(
                json.dumps(reference, indent=2, sort_keys=True), encoding="utf-8"
            )
        print("train-only mode: using the frozen experiment/cost contract from ai_research_manifest.json", flush=True)
    else:
        experiment = experiment_from_args(args)
        ranges = WorldSamplingRanges()
        costs = CostModel(
            brokerage_bps_per_side=args.brokerage_bps,
            slippage_bps_per_side=args.slippage_bps,
        )
        write_research_manifest(
            output_dir,
            experiment,
            ranges,
            costs,
            git_commit=args.git_commit,
            reference_procedure_manifest=args.reference_procedure_manifest,
        )

    if args.command in {"build-dataset", "all"}:
        print(
            "AI study role: independent sampled mechanism discovery; headline rates are not "
            "directly comparable to the statistical scenario-grid experiment.", flush=True,
        )
        print(
            f"AI dataset design: {experiment.structures} structures × "
            f"{experiment.seeds_per_structure} seeds = {experiment.total_runs} runs",
            flush=True,
        )
        dataset, runs = build_training_dataset(
            experiment,
            output_dir,
            jobs=args.jobs,
            costs=costs,
            ranges=ranges,
            git_commit=args.git_commit,
            checkpoint=args.checkpoint,
        )
        save_dataset_diagnostics(dataset, output_dir, runs)
        finalize_research_manifest(output_dir)
    else:
        dataset = _load_dataset(dataset_path)
        runs = _load_dataset(output_dir / "synthetic_runs.csv")

    if args.command in {"train", "all"}:
        # Always recreate the split deterministically from the complete generated structure/run
        # universe.  The immutable raw dataset remains untouched after its SHA-256 is frozen.
        dataset = dataset.drop(columns=["split"], errors="ignore")
        split_dataset, split_manifest = split_by_structure(
            dataset,
            experiment,
            run_universe=runs,
        )
        split_dataset.to_csv(output_dir / "pattern_training_dataset_with_split.csv", index=False)
        split_manifest.to_csv(output_dir / "structure_split_manifest.csv", index=False)
        split_lookup = dict(zip(split_manifest["structure_id"].astype(str), split_manifest["split"]))
        run_universe = runs.copy()
        run_universe["structure_id"] = run_universe["structure_id"].astype(str)
        run_universe["run_id"] = run_universe["run_id"].astype(str)
        run_universe["split"] = run_universe["structure_id"].map(split_lookup)
        if run_universe["split"].isna().any():
            raise AssertionError("run universe contains an unassigned structure")
        run_universe.to_csv(output_dir / "run_split_manifest.csv", index=False)

        feature_sets = ("observable", "oracle") if args.feature_set == "both" else (args.feature_set,)
        comparisons: list[pd.DataFrame] = []
        for feature_set in feature_sets:
            print(f"training models with feature_set={feature_set}", flush=True)
            comparison = train_and_evaluate_models(
                split_dataset,
                experiment,
                output_dir,
                feature_set=feature_set,
                run_universe=run_universe,
            )
            comparisons.append(comparison)
            print(
                comparison[
                    [
                        "model",
                        "roc_auc",
                        "brier_score",
                        "precision_at_threshold",
                        "selected_success_rate",
                        "mean_selected_net_return",
                        "mean_run_net_return_points",
                    ]
                ].to_string(index=False),
                flush=True,
            )
        combined = pd.concat(comparisons, ignore_index=True)
        combined.to_csv(output_dir / "model_comparison_all.csv", index=False)
        print(f"AI research results written to {output_dir.resolve()}", flush=True)


def run_cli(args: argparse.Namespace) -> None:
    """Mark a reused result directory incomplete before any fallible work begins."""

    if not getattr(args, "legacy_in_memory", False):
        # The streaming runner validates D: before creating anything and owns
        # its result-directory lock/status. Do not mutate a second status here.
        _run_cli_impl(args)
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "ai_run_status.json"

    def write_status(status: str, *, valid: bool = False, error: str | None = None) -> None:
        status_path.write_text(json.dumps({
            "status": status, "confirmatory_results_valid": valid,
            "command": args.command, "error": error,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2, sort_keys=True), encoding="utf-8")

    write_status("RUNNING")
    try:
        _run_cli_impl(args)
    except Exception as exc:
        status = ("INSUFFICIENT_VALIDATION_SUPPORT" if "INSUFFICIENT_VALIDATION_SUPPORT" in str(exc)
                  else "FAILED")
        write_status(status, error=str(exc))
        raise
    write_status("EVALUATION_COMPLETE" if args.command in {"train", "all"} else "DATASET_COMPLETE",
                 valid=args.command in {"train", "all"})


def main() -> None:
    parser = build_argument_parser()
    run_cli(parser.parse_args())


__all__ = [
    "AI_PIPELINE_VERSION",
    "WorldSamplingRanges",
    "AIExperimentConfig",
    "RunSpec",
    "OBSERVABLE_NUMERIC_FEATURES",
    "ORACLE_NUMERIC_FEATURES",
    "sample_run_specs",
    "create_pattern_row",
    "build_training_dataset",
    "iter_pattern_runs",
    "assign_structure_splits",
    "split_by_structure",
    "feature_columns",
    "train_and_evaluate_models",
    "finalize_research_manifest",
    "load_frozen_research_manifest",
]


if __name__ == "__main__":
    main()
