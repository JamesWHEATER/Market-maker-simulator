"""Causal synthetic market-maker simulator for chart-pattern research.

The simulator is designed for controlled experiments rather than for producing one visually
plausible price path.  Nine market mechanisms can be activated individually or in arbitrary
mixtures.  Active worlds generate independent Poisson trader arrivals, so a mixture represents a
heterogeneous population instead of an average synthetic trader.  Exogenous random streams are
stable by name, which supports common-random-number comparisons across counterfactual scenarios.

Important research contracts
----------------------------
* Every trading decision is causal: it uses only state available before the current execution.
* Latent fundamental value is not exposed to ordinary value/adaptive traders when an information
  world is active; only the designated informed component can use it.
* ``random_null_mode='efficient'`` provides a microstructure-free price null, while
  ``'microstructure'`` keeps random order flow but preserves bid/ask and inventory effects.
* Simulator snapshots preserve intrastep OHLCV, allowing ``chart_renderer.build_candles`` to retain
  the range created by multiple independent trader arrivals.
* Configuration and run fingerprints are deterministic and exported with every experiment.

The default parameters are deliberately baseline parameters, not claimed empirical estimates.
They should be frozen before a hypothesis test and calibrated/varied only in explicit experiment
specifications.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from chart_renderer import CANDLE_BUILDER_VERSION, RENDERER_VERSION, PriceCandle, build_candles, write_interactive_candlestick_html


PRICE_EPS = sys.float_info.min  # numerical positive floor only; never an economic price threshold
SIMULATOR_VERSION = "3.5.0"
# Random-stream identity is intentionally decoupled from code versioning.  Keep this frozen
# unless a deliberate stochastic-stream redesign is part of the experiment specification.
RNG_STREAM_VERSION = "3.0.0"
WORLD_NAMES = (
    "random",
    "rule_based",
    "emotional",
    "information",
    "mean_reversion",
    "momentum",
    "regime",
    "liquidity",
    "adaptive",
)
WORLD_ALIASES = {
    "mathematical": "rule_based",
    "math": "rule_based",
    "behavioural": "emotional",
    "behavioral": "emotional",
    "info": "information",
    "mean-reversion": "mean_reversion",
    "meanreversion": "mean_reversion",
    "trend": "momentum",
    "regime_switching": "regime",
    "regime-switching": "regime",
    "order_flow": "liquidity",
    "order-flow": "liquidity",
    "evolutionary": "adaptive",
}


@dataclass(frozen=True, slots=True)
class _FrozenMapping(Mapping[str, Any]):
    """Small immutable mapping that remains deterministic and pickle-friendly."""

    _items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "_FrozenMapping":
        return cls(tuple((str(key), value) for key, value in values.items()))

    def __getitem__(self, key: str) -> Any:
        for existing, value in self._items:
            if existing == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_tanh(value: float) -> float:
    if value >= 20.0:
        return 1.0
    if value <= -20.0:
        return -1.0
    return math.tanh(value)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _sign(value: float, eps: float = 1e-15) -> float:
    if value > eps:
        return 1.0
    if value < -eps:
        return -1.0
    return 0.0


def _normalize_world_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    normalized = WORLD_ALIASES.get(normalized, normalized)
    if normalized not in WORLD_NAMES:
        raise ValueError(f"Unknown world {name!r}. Valid worlds: {', '.join(WORLD_NAMES)}")
    return normalized


def _real(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {numeric!r}")
    return numeric


def _validate_probability(name: str, value: object) -> float:
    numeric = _real(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {numeric}")
    return numeric


def _validate_positive(name: str, value: object, *, allow_zero: bool = False) -> float:
    numeric = _real(name, value)
    valid = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}, got {numeric}")
    return numeric


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}")
    return int(value)


def _stable_seed(seed: int, label: str) -> int:
    payload = f"{seed}|{label}|{RNG_STREAM_VERSION}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _rng(seed: int, label: str) -> random.Random:
    return random.Random(_stable_seed(seed, label))


def _poisson_count(rng: random.Random, rate: float, cap: int) -> tuple[int, bool]:
    """Exact Poisson-process count on a unit interval, capped for pathological configurations."""

    if rate <= 0.0:
        return 0, False
    elapsed = 0.0
    count = 0
    while count <= cap:
        elapsed += rng.expovariate(rate)
        if elapsed > 1.0:
            return count, False
        count += 1
    return cap, True


def _floor_to_tick(price: float, tick: float) -> float:
    units = math.floor((price + tick * 1e-12) / tick)
    return max(tick, units * tick)


def _ceil_to_tick(price: float, tick: float) -> float:
    units = math.ceil((price - tick * 1e-12) / tick)
    return max(tick, units * tick)


def _price_after_log_move(price: float, log_move: float, context: str) -> float:
    """Apply an unconstrained model log move, failing loudly outside supported float range."""

    price = _validate_positive(context, price)
    log_move = _real(f"{context} log move", log_move)
    new_log = math.log(price) + log_move
    min_log = math.log(sys.float_info.min)
    max_log = math.log(sys.float_info.max) - 1e-12
    if not min_log <= new_log <= max_log:
        raise FloatingPointError(
            f"{context} would move price outside the supported positive finite range; "
            "reduce the configured drift/volatility/shock scale"
        )
    result = math.exp(new_log)
    if not math.isfinite(result) or result < PRICE_EPS:
        raise FloatingPointError(f"{context} produced an unsupported price {result!r}")
    return result


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _json_ready(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration/result metadata cannot contain NaN or infinity")
        return value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class RandomWorldConfig:
    """Uninformed noise traders. ``signal_strength`` scales a diagnostic noise draw only."""

    # Retained for backward compatibility/provenance. Random executions are always fair coins;
    # this value is stored as a diagnostic noise draw and never masquerades as information.
    signal_strength: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_strength", _validate_positive("random.signal_strength", self.signal_strength, allow_zero=True))


@dataclass(frozen=True, slots=True)
class RuleBasedWorldConfig:
    """Mechanical traders driven only by already-observed prices."""

    short_ma: int = 8
    long_ma: int = 24
    breakout_lookback: int = 20
    zscore_lookback: int = 24
    zscore_scale: float = 1.5
    ma_weight: float = 0.40
    breakout_weight: float = 0.35
    contrarian_weight: float = 0.25
    activity_boost: float = 0.50
    strict_execution: bool = True
    strict_signal_threshold: float = 0.10

    def __post_init__(self) -> None:
        if isinstance(self.short_ma, bool) or not isinstance(self.short_ma, int) or self.short_ma < 2:
            raise ValueError("rule_based.short_ma must be an integer >= 2")
        if isinstance(self.long_ma, bool) or not isinstance(self.long_ma, int) or self.long_ma <= self.short_ma:
            raise ValueError("rule_based.long_ma must be an integer > short_ma")
        for name in ("breakout_lookback", "zscore_lookback"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise ValueError(f"rule_based.{name} must be an integer >= 3")
        object.__setattr__(self, "zscore_scale", _validate_positive("rule_based.zscore_scale", self.zscore_scale))
        for name in ("ma_weight", "breakout_weight", "contrarian_weight"):
            object.__setattr__(self, name, _validate_positive(f"rule_based.{name}", getattr(self, name), allow_zero=True))
        if self.ma_weight + self.breakout_weight + self.contrarian_weight <= 0.0:
            raise ValueError("rule-based strategy weights cannot all be zero")
        object.__setattr__(self, "activity_boost", _validate_positive("rule_based.activity_boost", self.activity_boost, allow_zero=True))
        object.__setattr__(self, "strict_signal_threshold", _validate_probability("rule_based.strict_signal_threshold", self.strict_signal_threshold))
        if not isinstance(self.strict_execution, bool):
            raise ValueError("rule_based.strict_execution must be boolean")


@dataclass(frozen=True, slots=True)
class EmotionalWorldConfig:
    """Behavioral feedback: FOMO, fear, greed and drawdown response."""

    decay: float = 0.90
    return_scale: float = 0.0015
    fomo_sensitivity: float = 0.22
    fear_sensitivity: float = 0.25
    greed_sensitivity: float = 0.12
    drawdown_sensitivity: float = 0.035
    drawdown_scale: float = 0.05
    signal_scale: float = 1.20
    size_boost: float = 1.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "decay", _validate_probability("emotional.decay", self.decay))
        object.__setattr__(self, "return_scale", _validate_positive("emotional.return_scale", self.return_scale))
        for name in (
            "fomo_sensitivity", "fear_sensitivity", "greed_sensitivity",
            "drawdown_sensitivity", "drawdown_scale", "signal_scale", "size_boost",
        ):
            object.__setattr__(self, name, _validate_positive(f"emotional.{name}", getattr(self, name), allow_zero=True))
        if self.drawdown_scale == 0.0 or self.signal_scale == 0.0:
            raise ValueError("emotional.drawdown_scale and signal_scale must be > 0")


@dataclass(frozen=True, slots=True)
class InformationWorldConfig:
    """Latent fundamental shocks plus a designated partially informed trader population."""

    event_probability: float = 0.0025
    shock_std_fraction: float = 0.018
    min_shock_fraction: float = 0.004
    public_diffusion_rate: float = 0.035
    informed_fraction: float = 0.12
    signal_gap_fraction: float = 0.010
    event_size_boost: float = 1.50
    event_decay: float = 0.93

    def __post_init__(self) -> None:
        for name in ("event_probability", "public_diffusion_rate", "informed_fraction", "event_decay"):
            object.__setattr__(self, name, _validate_probability(f"information.{name}", getattr(self, name)))
        for name in ("shock_std_fraction", "min_shock_fraction", "signal_gap_fraction", "event_size_boost"):
            object.__setattr__(self, name, _validate_positive(f"information.{name}", getattr(self, name), allow_zero=True))
        if self.signal_gap_fraction == 0.0 or self.shock_std_fraction == 0.0:
            raise ValueError("information.signal_gap_fraction and shock_std_fraction must be > 0")


@dataclass(frozen=True, slots=True)
class MeanReversionWorldConfig:
    """Value traders pull price toward the *observable* public fundamental estimate."""

    deviation_scale_fraction: float = 0.006
    deadband_fraction: float = 0.0005
    size_boost: float = 0.70

    def __post_init__(self) -> None:
        object.__setattr__(self, "deviation_scale_fraction", _validate_positive("mean_reversion.deviation_scale_fraction", self.deviation_scale_fraction))
        object.__setattr__(self, "deadband_fraction", _validate_positive("mean_reversion.deadband_fraction", self.deadband_fraction, allow_zero=True))
        object.__setattr__(self, "size_boost", _validate_positive("mean_reversion.size_boost", self.size_boost, allow_zero=True))


@dataclass(frozen=True, slots=True)
class MomentumWorldConfig:
    """Trend followers extrapolate already-observed transaction returns."""

    lookback: int = 12
    return_scale: float = 0.006
    deadband: float = 0.0002
    size_boost: float = 0.85

    def __post_init__(self) -> None:
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError("momentum.lookback must be an integer >= 2")
        object.__setattr__(self, "return_scale", _validate_positive("momentum.return_scale", self.return_scale))
        object.__setattr__(self, "deadband", _validate_positive("momentum.deadband", self.deadband, allow_zero=True))
        object.__setattr__(self, "size_boost", _validate_positive("momentum.size_boost", self.size_boost, allow_zero=True))


@dataclass(frozen=True, slots=True)
class RegimeWorldConfig:
    """Persistent hidden macro/microstructure regimes with causal Markov switching."""

    stay_probability: float = 0.992
    trend_drift_fraction_per_step: float = 0.00009
    panic_drift_fraction_per_step: float = 0.00025
    high_vol_multiplier: float = 3.0
    panic_vol_multiplier: float = 2.4
    euphoria_vol_multiplier: float = 1.8
    panic_liquidity_multiplier: float = 0.45
    high_vol_liquidity_multiplier: float = 0.70
    regime_signal_strength: float = 0.85

    def __post_init__(self) -> None:
        object.__setattr__(self, "stay_probability", _validate_probability("regime.stay_probability", self.stay_probability))
        for name in (
            "trend_drift_fraction_per_step", "panic_drift_fraction_per_step",
            "high_vol_multiplier", "panic_vol_multiplier", "euphoria_vol_multiplier",
            "panic_liquidity_multiplier", "high_vol_liquidity_multiplier", "regime_signal_strength",
        ):
            object.__setattr__(self, name, _validate_positive(f"regime.{name}", getattr(self, name), allow_zero=True))
        if min(self.high_vol_multiplier, self.panic_vol_multiplier, self.euphoria_vol_multiplier) <= 0.0:
            raise ValueError("regime volatility multipliers must be > 0")
        if min(self.panic_liquidity_multiplier, self.high_vol_liquidity_multiplier) <= 0.0:
            raise ValueError("regime liquidity multipliers must be > 0")


@dataclass(frozen=True, slots=True)
class LiquidityWorldConfig:
    """Stochastic depth, round-number support/resistance and stop cascades."""

    persistence: float = 0.965
    log_liquidity_vol: float = 0.08
    min_liquidity: float = 0.30
    max_liquidity: float = 3.00
    round_number_interval: float = 0.0
    round_proximity_fraction: float = 0.0018
    support_resistance_strength: float = 0.45
    stop_cascade_strength: float = 0.90
    stop_cascade_size_boost: float = 1.80

    def __post_init__(self) -> None:
        object.__setattr__(self, "persistence", _validate_probability("liquidity.persistence", self.persistence))
        for name in (
            "log_liquidity_vol", "min_liquidity", "max_liquidity", "round_number_interval",
            "round_proximity_fraction", "support_resistance_strength", "stop_cascade_strength",
            "stop_cascade_size_boost",
        ):
            object.__setattr__(self, name, _validate_positive(f"liquidity.{name}", getattr(self, name), allow_zero=True))
        if self.min_liquidity <= 0.0 or self.max_liquidity < self.min_liquidity:
            raise ValueError("liquidity bounds are invalid")


@dataclass(frozen=True, slots=True)
class AdaptiveWorldConfig:
    """Evolutionary traders reweight causal strategies by realised past success."""

    score_decay: float = 0.965
    learning_rate: float = 0.16
    return_scale: float = 0.0015
    temperature: float = 0.35
    exploration_weight: float = 0.05
    size_boost: float = 0.70

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_decay", _validate_probability("adaptive.score_decay", self.score_decay))
        object.__setattr__(self, "learning_rate", _validate_positive("adaptive.learning_rate", self.learning_rate, allow_zero=True))
        object.__setattr__(self, "return_scale", _validate_positive("adaptive.return_scale", self.return_scale))
        object.__setattr__(self, "temperature", _validate_positive("adaptive.temperature", self.temperature))
        object.__setattr__(self, "exploration_weight", _validate_probability("adaptive.exploration_weight", self.exploration_weight))
        object.__setattr__(self, "size_boost", _validate_positive("adaptive.size_boost", self.size_boost, allow_zero=True))


@dataclass(frozen=True, slots=True)
class SyntheticMarketConfig:
    """Complete configuration for one or any mixture of the nine synthetic market worlds.

    ``world_weights`` are relative component strengths and trader-population shares. Multiplying
    every weight by the same constant leaves the mixture unchanged; overall order-arrival activity
    is controlled separately by ``base_arrival_rate``. Market-wide mechanisms (information opacity,
    regime effects and stochastic liquidity) are smoothly strength-weighted as well, so adding a
    component with a tiny weight cannot silently activate its structural effect at full strength.
    """

    steps: int = 10_000
    seed: int = 42
    initial_fair_value: float = 100.0

    # Legacy absolute controls are retained.  Fractional overrides are preferred for scale-free
    # research and default to the legacy values divided by the initial price when omitted.
    fair_value_step_vol: float = 0.03
    fair_value_step_vol_fraction: float | None = None
    base_fundamental_drift: float = 0.0
    base_fundamental_drift_fraction: float | None = None

    base_spread: float = 0.04
    base_spread_fraction: float | None = None
    min_tick: float = 0.01
    inventory_skew: float = 0.002
    inventory_skew_fraction: float | None = None
    inventory_skew_cap_fraction: float = 0.025
    inventory_spread_sensitivity: float = 1.25
    max_abs_inventory: int = 500
    inventory_hedge_trigger_fraction: float = 0.80
    inventory_hedge_fraction: float = 0.35
    hedge_cost_fraction: float = 0.00005

    min_order_size: int = 1
    max_order_size: int = 10
    max_order_size_multiplier: float = 5.0
    base_arrival_rate: float = 2.0
    max_arrivals_per_world_per_step: int = 64
    decision_signal_strength: float = 2.1
    decision_noise: float = 0.75

    fundamental_anchor_strength: float = 0.025
    order_impact_fraction: float = 0.00032
    microstructure_noise_fraction: float = 0.000025
    max_single_step_mid_move_fraction: float = 0.035

    strict_random_null: bool = True
    random_null_mode: str = "efficient"  # efficient | microstructure
    world_weights: Mapping[str, float] = field(default_factory=lambda: {"random": 1.0})

    random_world: RandomWorldConfig = field(default_factory=RandomWorldConfig)
    rule_based: RuleBasedWorldConfig = field(default_factory=RuleBasedWorldConfig)
    emotional: EmotionalWorldConfig = field(default_factory=EmotionalWorldConfig)
    information: InformationWorldConfig = field(default_factory=InformationWorldConfig)
    mean_reversion: MeanReversionWorldConfig = field(default_factory=MeanReversionWorldConfig)
    momentum: MomentumWorldConfig = field(default_factory=MomentumWorldConfig)
    regime: RegimeWorldConfig = field(default_factory=RegimeWorldConfig)
    liquidity: LiquidityWorldConfig = field(default_factory=LiquidityWorldConfig)
    adaptive: AdaptiveWorldConfig = field(default_factory=AdaptiveWorldConfig)

    def __post_init__(self) -> None:
        _positive_int("steps", self.steps)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        nested_types = (
            ("random_world", RandomWorldConfig),
            ("rule_based", RuleBasedWorldConfig),
            ("emotional", EmotionalWorldConfig),
            ("information", InformationWorldConfig),
            ("mean_reversion", MeanReversionWorldConfig),
            ("momentum", MomentumWorldConfig),
            ("regime", RegimeWorldConfig),
            ("liquidity", LiquidityWorldConfig),
            ("adaptive", AdaptiveWorldConfig),
        )
        for field_name, expected_type in nested_types:
            if not isinstance(getattr(self, field_name), expected_type):
                raise ValueError(f"{field_name} must be a {expected_type.__name__}")
        object.__setattr__(self, "initial_fair_value", _validate_positive("initial_fair_value", self.initial_fair_value))
        object.__setattr__(self, "fair_value_step_vol", _validate_positive("fair_value_step_vol", self.fair_value_step_vol, allow_zero=True))
        if self.fair_value_step_vol_fraction is not None:
            object.__setattr__(self, "fair_value_step_vol_fraction", _validate_positive("fair_value_step_vol_fraction", self.fair_value_step_vol_fraction, allow_zero=True))
        object.__setattr__(self, "base_fundamental_drift", _real("base_fundamental_drift", self.base_fundamental_drift))
        if self.base_fundamental_drift_fraction is not None:
            object.__setattr__(self, "base_fundamental_drift_fraction", _real("base_fundamental_drift_fraction", self.base_fundamental_drift_fraction))

        for name in ("base_spread", "min_tick"):
            object.__setattr__(self, name, _validate_positive(name, getattr(self, name)))
        if self.min_tick > self.initial_fair_value:
            raise ValueError("min_tick cannot exceed initial_fair_value")
        if self.base_spread_fraction is not None:
            object.__setattr__(self, "base_spread_fraction", _validate_positive("base_spread_fraction", self.base_spread_fraction))
        object.__setattr__(self, "inventory_skew", _validate_positive("inventory_skew", self.inventory_skew, allow_zero=True))
        if self.inventory_skew_fraction is not None:
            object.__setattr__(self, "inventory_skew_fraction", _validate_positive("inventory_skew_fraction", self.inventory_skew_fraction, allow_zero=True))
        object.__setattr__(
            self,
            "inventory_skew_cap_fraction",
            _validate_probability("inventory_skew_cap_fraction", self.inventory_skew_cap_fraction),
        )
        if self.inventory_skew_cap_fraction >= 1.0:
            raise ValueError("inventory_skew_cap_fraction must be < 1 to keep reservation prices positive")
        object.__setattr__(self, "inventory_spread_sensitivity", _validate_positive("inventory_spread_sensitivity", self.inventory_spread_sensitivity, allow_zero=True))
        _positive_int("max_abs_inventory", self.max_abs_inventory)
        object.__setattr__(self, "inventory_hedge_trigger_fraction", _validate_probability("inventory_hedge_trigger_fraction", self.inventory_hedge_trigger_fraction))
        if self.inventory_hedge_trigger_fraction <= 0.0:
            raise ValueError("inventory_hedge_trigger_fraction must be > 0")
        object.__setattr__(self, "inventory_hedge_fraction", _validate_probability("inventory_hedge_fraction", self.inventory_hedge_fraction))
        object.__setattr__(self, "hedge_cost_fraction", _validate_probability("hedge_cost_fraction", self.hedge_cost_fraction))
        if self.hedge_cost_fraction >= 1.0:
            raise ValueError("hedge_cost_fraction must be < 1")

        _positive_int("min_order_size", self.min_order_size)
        _positive_int("max_order_size", self.max_order_size)
        if self.max_order_size < self.min_order_size:
            raise ValueError("max_order_size must be >= min_order_size")
        object.__setattr__(self, "max_order_size_multiplier", _validate_positive("max_order_size_multiplier", self.max_order_size_multiplier))
        object.__setattr__(self, "base_arrival_rate", _validate_positive("base_arrival_rate", self.base_arrival_rate, allow_zero=True))
        _positive_int("max_arrivals_per_world_per_step", self.max_arrivals_per_world_per_step)
        object.__setattr__(self, "decision_signal_strength", _validate_positive("decision_signal_strength", self.decision_signal_strength, allow_zero=True))
        object.__setattr__(self, "decision_noise", _validate_positive("decision_noise", self.decision_noise, allow_zero=True))

        object.__setattr__(self, "fundamental_anchor_strength", _validate_probability("fundamental_anchor_strength", self.fundamental_anchor_strength))
        object.__setattr__(self, "order_impact_fraction", _validate_positive("order_impact_fraction", self.order_impact_fraction, allow_zero=True))
        object.__setattr__(self, "microstructure_noise_fraction", _validate_positive("microstructure_noise_fraction", self.microstructure_noise_fraction, allow_zero=True))
        object.__setattr__(self, "max_single_step_mid_move_fraction", _validate_positive("max_single_step_mid_move_fraction", self.max_single_step_mid_move_fraction))

        if not isinstance(self.strict_random_null, bool):
            raise ValueError("strict_random_null must be boolean")
        mode = str(self.random_null_mode).strip().lower()
        if mode not in {"efficient", "microstructure"}:
            raise ValueError("random_null_mode must be 'efficient' or 'microstructure'")
        object.__setattr__(self, "random_null_mode", mode)

        if not isinstance(self.world_weights, Mapping):
            raise ValueError("world_weights must be a mapping")
        cleaned: dict[str, float] = {}
        for raw_name, raw_weight in self.world_weights.items():
            name = _normalize_world_name(str(raw_name))
            weight = _validate_positive(f"world weight for {name!r}", raw_weight, allow_zero=True)
            if weight > 0.0:
                cleaned[name] = cleaned.get(name, 0.0) + weight
        if not cleaned:
            raise ValueError("At least one world must have a positive weight")
        ordered = {name: cleaned[name] for name in WORLD_NAMES if name in cleaned}
        object.__setattr__(self, "world_weights", _FrozenMapping.from_mapping(ordered))

    @property
    def active_worlds(self) -> tuple[str, ...]:
        return tuple(self.world_weights)

    @property
    def normalized_world_weights(self) -> Mapping[str, float]:
        total = sum(self.world_weights.values())
        return _FrozenMapping.from_mapping({name: weight / total for name, weight in self.world_weights.items()})

    @property
    def is_random_only(self) -> bool:
        return self.active_worlds == ("random",)

    @property
    def is_strict_random_null(self) -> bool:
        return self.strict_random_null and self.is_random_only

    @property
    def is_efficient_random_null(self) -> bool:
        return self.is_strict_random_null and self.random_null_mode == "efficient"

    @property
    def is_microstructure_random_null(self) -> bool:
        return self.is_strict_random_null and self.random_null_mode == "microstructure"

    @property
    def effective_fundamental_vol_fraction(self) -> float:
        if self.fair_value_step_vol_fraction is not None:
            return self.fair_value_step_vol_fraction
        return self.fair_value_step_vol / self.initial_fair_value

    @property
    def effective_fundamental_drift_fraction(self) -> float:
        if self.base_fundamental_drift_fraction is not None:
            return self.base_fundamental_drift_fraction
        return self.base_fundamental_drift / self.initial_fair_value

    @property
    def effective_base_spread_fraction(self) -> float:
        if self.base_spread_fraction is not None:
            return self.base_spread_fraction
        return self.base_spread / self.initial_fair_value

    @property
    def effective_inventory_skew_fraction(self) -> float:
        if self.inventory_skew_fraction is not None:
            return self.inventory_skew_fraction
        return self.inventory_skew / self.initial_fair_value

    @property
    def required_history_length(self) -> int:
        return max(
            130,
            self.rule_based.long_ma + 2,
            self.rule_based.breakout_lookback + 3,
            self.rule_based.zscore_lookback + 2,
            self.momentum.lookback + 2,
        )


@dataclass(frozen=True, slots=True)
class RandomMarketConfig:
    """Backward-compatible configuration for the original random-only simulator API."""

    steps: int = 10_000
    seed: int = 42
    initial_fair_value: float = 100.0
    fair_value_step_vol: float = 0.03
    base_spread: float = 0.04
    inventory_skew: float = 0.002
    min_order_size: int = 1
    max_order_size: int = 10

    def to_synthetic_config(self) -> SyntheticMarketConfig:
        return SyntheticMarketConfig(
            steps=self.steps,
            seed=self.seed,
            initial_fair_value=self.initial_fair_value,
            fair_value_step_vol=self.fair_value_step_vol,
            base_spread=self.base_spread,
            inventory_skew=self.inventory_skew,
            min_order_size=self.min_order_size,
            max_order_size=self.max_order_size,
            world_weights={"random": 1.0},
            strict_random_null=True,
            random_null_mode="microstructure",
        )


@dataclass(slots=True)
class MarketMaker:
    base_spread_fraction: float
    inventory_skew_fraction: float
    min_tick: float
    inventory_skew_cap_fraction: float
    inventory_spread_sensitivity: float
    max_abs_inventory: int
    cash: float = 0.0
    inventory: int = 0

    def __post_init__(self) -> None:
        _validate_positive("maker.base_spread_fraction", self.base_spread_fraction)
        _validate_positive("maker.inventory_skew_fraction", self.inventory_skew_fraction, allow_zero=True)
        _validate_positive("maker.min_tick", self.min_tick)
        _validate_positive("maker.inventory_skew_cap_fraction", self.inventory_skew_cap_fraction, allow_zero=True)
        _validate_positive("maker.inventory_spread_sensitivity", self.inventory_spread_sensitivity, allow_zero=True)
        _positive_int("maker.max_abs_inventory", self.max_abs_inventory)
        if abs(self.inventory) > self.max_abs_inventory:
            raise ValueError("initial maker inventory exceeds max_abs_inventory")

    def quote(self, reference_price: float, *, spread_multiplier: float = 1.0) -> tuple[float, float]:
        reference = max(self.min_tick, _real("reference_price", reference_price))
        spread_multiplier = _validate_positive("spread_multiplier", spread_multiplier)
        inventory_ratio = self.inventory / self.max_abs_inventory
        raw_skew_fraction = self.inventory_skew_fraction * self.inventory
        bounded_skew_fraction = _clamp(
            raw_skew_fraction,
            -self.inventory_skew_cap_fraction,
            self.inventory_skew_cap_fraction,
        )
        reservation = max(self.min_tick, reference * (1.0 - bounded_skew_fraction))
        inventory_spread = 1.0 + self.inventory_spread_sensitivity * abs(inventory_ratio) ** 2
        effective_spread = max(
            self.min_tick,
            reference * self.base_spread_fraction * spread_multiplier * inventory_spread,
        )
        bid = _floor_to_tick(reservation - effective_spread / 2.0, self.min_tick)
        ask = _ceil_to_tick(reservation + effective_spread / 2.0, self.min_tick)
        if ask <= bid:
            ask = bid + self.min_tick
        return bid, ask

    def fill_capacity(self, taker_side: str) -> int:
        if taker_side == "buy":
            return self.inventory + self.max_abs_inventory
        if taker_side == "sell":
            return self.max_abs_inventory - self.inventory
        raise ValueError("taker_side must be 'buy' or 'sell'")

    def execute_taker(self, taker_side: str, price: float, requested_size: int) -> int:
        if isinstance(requested_size, bool) or not isinstance(requested_size, int) or requested_size <= 0:
            raise ValueError("requested_size must be a positive integer")
        execution_price = _validate_positive("execution price", price)
        filled = min(requested_size, max(0, self.fill_capacity(taker_side)))
        if filled <= 0:
            return 0
        if taker_side == "buy":
            self.cash += execution_price * filled
            self.inventory -= filled
        elif taker_side == "sell":
            self.cash -= execution_price * filled
            self.inventory += filled
        else:
            raise ValueError("taker_side must be 'buy' or 'sell'")
        if abs(self.inventory) > self.max_abs_inventory:
            raise RuntimeError("maker inventory hard limit was violated")
        return filled

    def hedge_excess(
        self,
        reference_price: float,
        *,
        trigger_fraction: float,
        hedge_fraction: float,
        cost_fraction: float,
    ) -> tuple[int, float]:
        """Reduce inventory beyond the soft limit; return signed hedge quantity and cost."""

        trigger = max(1, int(math.floor(self.max_abs_inventory * trigger_fraction)))
        excess = abs(self.inventory) - trigger
        if excess <= 0 or hedge_fraction <= 0.0:
            return 0, 0.0
        quantity = max(1, int(math.ceil(excess * hedge_fraction)))
        quantity = min(quantity, abs(self.inventory))
        mark = _validate_positive("hedge reference price", reference_price)
        cost_fraction = _validate_positive("hedge cost fraction", cost_fraction, allow_zero=True)
        if self.inventory > 0:
            # Sell long inventory externally at a small cost to the reference price.
            execution = mark * (1.0 - cost_fraction)
            self.cash += execution * quantity
            self.inventory -= quantity
            signed_quantity = -quantity
        else:
            execution = mark * (1.0 + cost_fraction)
            self.cash -= execution * quantity
            self.inventory += quantity
            signed_quantity = quantity
        explicit_cost = abs(execution - mark) * quantity
        return signed_quantity, explicit_cost

    def mark_to_market_pnl(self, mark_price: float) -> float:
        return self.cash + self.inventory * _validate_positive("mark_price", mark_price)


@dataclass(frozen=True, slots=True)
class WorldSignal:
    signal: float = 0.0
    size_multiplier: float = 1.0
    activity_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    impact_multiplier: float = 1.0
    metadata: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal = _real("WorldSignal.signal", self.signal)
        object.__setattr__(self, "signal", _clamp(signal, -1.0, 1.0))
        for name in ("size_multiplier", "activity_multiplier", "spread_multiplier", "impact_multiplier"):
            object.__setattr__(self, name, _validate_positive(f"WorldSignal.{name}", getattr(self, name)))
        cleaned: dict[str, float | str] = {}
        for key, value in self.metadata.items():
            if isinstance(value, str):
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = _real(f"metadata[{key!r}]", value)
        object.__setattr__(self, "metadata", _FrozenMapping.from_mapping(cleaned))


@dataclass(frozen=True, slots=True)
class OrderIntent:
    source_world: str
    taker_side: str
    requested_size: int
    signal: float
    buy_probability: float
    impact_multiplier: float


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    step: int
    fair_value: float
    reference_price: float
    post_trade_reference_price: float
    bid: float
    ask: float
    taker_side: str
    order_size: int
    trade_price: float
    maker_inventory: int
    maker_cash: float
    maker_pnl: float
    # End-of-step executable quote around ``post_trade_reference_price``.  These fields make a
    # close-of-candle synthetic exit explicit instead of approximating the closing spread from the
    # quote observed at the beginning of the last simulator step.  Defaults preserve compatibility
    # with older stored snapshots.
    post_trade_bid: float = 0.0
    post_trade_ask: float = 0.0
    maker_pnl_reference: float = 0.0
    aggregate_signal: float = 0.0
    buy_probability: float = 0.5
    signed_order_flow: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    trade_count: int = 0
    rejected_volume: int = 0
    arrival_cap_hit: bool = False
    hedge_volume: int = 0
    hedge_cost: float = 0.0
    liquidity: float = 1.0
    regime: str = "none"
    sentiment: float = 0.0
    fomo: float = 0.0
    fear: float = 0.0
    greed: float = 0.0
    public_fundamental: float = 0.0
    information_gap: float = 0.0
    information_event: bool = False
    step_open: float = 0.0
    step_high: float = 0.0
    step_low: float = 0.0
    step_close: float = 0.0
    traded_volume: int = 0
    active_worlds: str = "random"
    simulation_seed: int = 0
    scenario_id: str = ""
    run_id: str = ""
    world_weights_json: str = "{}"
    world_signals_json: str = "{}"
    world_flow_json: str = "{}"
    simulator_version: str = SIMULATOR_VERSION

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def post_trade_spread(self) -> float:
        if self.post_trade_bid > 0.0 and self.post_trade_ask >= self.post_trade_bid:
            return self.post_trade_ask - self.post_trade_bid
        return self.spread

    # OHLCV aliases consumed by chart_renderer without coupling it to this class.
    @property
    def open(self) -> float:
        return self.step_open if self.step_open > 0.0 else self.trade_price

    @property
    def high(self) -> float:
        return self.step_high if self.step_high > 0.0 else self.trade_price

    @property
    def low(self) -> float:
        return self.step_low if self.step_low > 0.0 else self.trade_price

    @property
    def close(self) -> float:
        return self.step_close if self.step_close > 0.0 else self.trade_price

    @property
    def volume(self) -> int:
        return self.traded_volume


@dataclass(slots=True)
class SimulationState:
    fundamental_value: float
    reference_price: float
    public_fundamental: float
    last_observed_price: float
    liquidity: float = 1.0
    regime: str = "neutral"
    fomo: float = 0.0
    fear: float = 0.0
    greed: float = 0.0
    sentiment: float = 0.0
    event_intensity: float = 0.0
    information_event: bool = False
    recent_trade_prices: list[float] = field(default_factory=list)
    recent_returns: list[float] = field(default_factory=list)
    rolling_peak_price: float = 0.0
    adaptive_scores: dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.0,
            "mean_reversion": 0.0,
            "breakout": 0.0,
            "noise": 0.0,
        }
    )
    adaptive_previous_signals: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RandomStreams:
    fundamental: random.Random
    information: random.Random
    regime: random.Random
    liquidity: random.Random
    microstructure: random.Random
    execution_order: random.Random
    adaptive_strategy: random.Random
    world_arrival: dict[str, random.Random]
    world_signal: dict[str, random.Random]
    world_decision: dict[str, random.Random]
    world_size: dict[str, random.Random]

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStreams":
        return cls(
            fundamental=_rng(seed, "fundamental"),
            information=_rng(seed, "information_environment"),
            regime=_rng(seed, "regime_environment"),
            liquidity=_rng(seed, "liquidity_environment"),
            microstructure=_rng(seed, "microstructure_noise"),
            execution_order=_rng(seed, "execution_order"),
            adaptive_strategy=_rng(seed, "adaptive_strategy"),
            world_arrival={name: _rng(seed, f"arrival:{name}") for name in WORLD_NAMES},
            world_signal={name: _rng(seed, f"signal:{name}") for name in WORLD_NAMES},
            world_decision={name: _rng(seed, f"decision:{name}") for name in WORLD_NAMES},
            world_size={name: _rng(seed, f"size:{name}") for name in WORLD_NAMES},
        )


@dataclass(frozen=True, slots=True)
class RegimeProfile:
    drift_fraction: float = 0.0
    volatility_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0
    signal: float = 0.0


REGIME_NAMES = (
    "neutral",
    "trend_up",
    "trend_down",
    "mean_reverting",
    "high_volatility",
    "panic",
    "euphoria",
)


def _recent_return(prices: Sequence[float], lookback: int) -> float:
    if len(prices) < 2:
        return 0.0
    span = min(max(1, lookback), len(prices) - 1)
    start = prices[-span - 1]
    end = prices[-1]
    if start <= 0.0 or end <= 0.0:
        return 0.0
    return math.log(end / start)


def _sample_std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def _moving_average_signal(prices: Sequence[float], short: int, long: int) -> float:
    if len(prices) < long:
        return 0.0
    short_avg = mean(prices[-short:])
    long_avg = mean(prices[-long:])
    returns = [
        math.log(prices[index] / prices[index - 1])
        for index in range(max(1, len(prices) - long + 1), len(prices))
        if prices[index - 1] > 0.0 and prices[index] > 0.0
    ]
    scale = max(_sample_std(returns) * math.sqrt(max(1, short)), 1e-8)
    relative_gap = (short_avg - long_avg) / max(long_avg, PRICE_EPS)
    return _safe_tanh(relative_gap / scale)


def _breakout_signal(prices: Sequence[float], lookback: int) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    current = prices[-1]
    previous_window = prices[-lookback - 1 : -1]
    high = max(previous_window)
    low = min(previous_window)
    width = max(high - low, current * 1e-8)
    if current > high:
        return _clamp((current - high) / width * 4.0 + 0.55, 0.0, 1.0)
    if current < low:
        return -_clamp((low - current) / width * 4.0 + 0.55, 0.0, 1.0)
    position = (current - low) / width
    return _clamp((position - 0.5) * 0.30, -0.15, 0.15)


def _zscore_contrarian_signal(prices: Sequence[float], lookback: int, zscore_scale: float) -> float:
    if len(prices) < lookback:
        return 0.0
    window = prices[-lookback:]
    center = mean(window)
    std = _sample_std(window)
    if std <= max(abs(center) * 1e-15, PRICE_EPS):
        return 0.0
    z_score = (window[-1] - center) / std
    return -_safe_tanh(z_score / zscore_scale)


def _fundamental_reversion_signal(
    reference_price: float,
    observable_fundamental: float,
    cfg: MeanReversionWorldConfig,
) -> float:
    gap_fraction = (observable_fundamental - reference_price) / max(reference_price, PRICE_EPS)
    if abs(gap_fraction) <= cfg.deadband_fraction:
        return 0.0
    effective = gap_fraction - math.copysign(cfg.deadband_fraction, gap_fraction)
    return _safe_tanh(effective / cfg.deviation_scale_fraction)


def _momentum_signal(prices: Sequence[float], cfg: MomentumWorldConfig) -> float:
    trailing_return = _recent_return(prices, cfg.lookback)
    if abs(trailing_return) <= cfg.deadband:
        return 0.0
    effective = trailing_return - math.copysign(cfg.deadband, trailing_return)
    return _safe_tanh(effective / cfg.return_scale)


def _effective_round_interval(price: float, configured_interval: float) -> float:
    if configured_interval > 0.0:
        return configured_interval
    price = max(price, PRICE_EPS)
    magnitude = 10.0 ** math.floor(math.log10(price))
    return max(PRICE_EPS, magnitude * 0.05)


def _crossed_round_level(previous: float, current: float, interval: float) -> bool:
    if previous <= 0.0 or current <= 0.0 or previous == current:
        return False
    low, high = sorted((previous, current))
    first_level = math.ceil(low / interval) * interval
    tolerance = max(PRICE_EPS, abs(interval) * 1e-12)
    return first_level <= high + tolerance and not math.isclose(
        first_level, low, rel_tol=0.0, abs_tol=tolerance
    )


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    scaled = {name: score / temperature for name, score in scores.items()}
    maximum = max(scaled.values())
    exponentials = {name: math.exp(_clamp(value - maximum, -700.0, 700.0)) for name, value in scaled.items()}
    total = sum(exponentials.values())
    if total <= 0.0:
        equal = 1.0 / len(exponentials)
        return {name: equal for name in exponentials}
    return {name: value / total for name, value in exponentials.items()}


def _regime_profile(regime: str, cfg: RegimeWorldConfig) -> RegimeProfile:
    if regime == "trend_up":
        return RegimeProfile(cfg.trend_drift_fraction_per_step, 1.15, 1.0, cfg.regime_signal_strength)
    if regime == "trend_down":
        return RegimeProfile(-cfg.trend_drift_fraction_per_step, 1.15, 1.0, -cfg.regime_signal_strength)
    if regime == "mean_reverting":
        return RegimeProfile(volatility_multiplier=0.85)
    if regime == "high_volatility":
        return RegimeProfile(volatility_multiplier=cfg.high_vol_multiplier, liquidity_multiplier=cfg.high_vol_liquidity_multiplier)
    if regime == "panic":
        return RegimeProfile(
            drift_fraction=-cfg.panic_drift_fraction_per_step,
            volatility_multiplier=cfg.panic_vol_multiplier,
            liquidity_multiplier=cfg.panic_liquidity_multiplier,
            signal=-cfg.regime_signal_strength,
        )
    if regime == "euphoria":
        return RegimeProfile(
            drift_fraction=cfg.panic_drift_fraction_per_step * 0.70,
            volatility_multiplier=cfg.euphoria_vol_multiplier,
            liquidity_multiplier=0.75,
            signal=cfg.regime_signal_strength,
        )
    return RegimeProfile()


def _update_regime(state: SimulationState, rng: random.Random, cfg: RegimeWorldConfig) -> RegimeProfile:
    if rng.random() > cfg.stay_probability:
        alternatives = [name for name in REGIME_NAMES if name != state.regime]
        state.regime = rng.choice(alternatives)
    return _regime_profile(state.regime, cfg)


def _update_emotions(state: SimulationState, cfg: EmotionalWorldConfig) -> None:
    last_return = state.recent_returns[-1] if state.recent_returns else 0.0
    scaled_up = max(last_return, 0.0) / cfg.return_scale
    scaled_down = max(-last_return, 0.0) / cfg.return_scale
    last_price = state.recent_trade_prices[-1] if state.recent_trade_prices else state.reference_price
    if state.recent_trade_prices:
        state.rolling_peak_price = max(state.recent_trade_prices[-128:])
    else:
        state.rolling_peak_price = last_price
    drawdown = max(0.0, 1.0 - last_price / max(state.rolling_peak_price, PRICE_EPS))
    trend = max(_recent_return(state.recent_trade_prices, 8), 0.0) / max(cfg.return_scale * 4.0, PRICE_EPS)
    state.fomo = _clamp(cfg.decay * state.fomo + cfg.fomo_sensitivity * scaled_up, 0.0, 8.0)
    state.fear = _clamp(
        cfg.decay * state.fear + cfg.fear_sensitivity * scaled_down
        + cfg.drawdown_sensitivity * drawdown / cfg.drawdown_scale,
        0.0,
        8.0,
    )
    state.greed = _clamp(cfg.decay * state.greed + cfg.greed_sensitivity * trend, 0.0, 8.0)
    state.sentiment = _safe_tanh((state.fomo + 0.65 * state.greed - state.fear) / cfg.signal_scale)


def _weighted_multiplier(multiplier: float, strength: float) -> float:
    """Interpolate a positive multiplicative effect toward one in log space."""

    multiplier = _validate_positive("multiplier", multiplier)
    strength = _validate_probability("component strength", strength)
    return math.exp(strength * math.log(multiplier))


def _update_information_environment(
    state: SimulationState,
    rng: random.Random,
    cfg: InformationWorldConfig,
    strength: float,
) -> None:
    """Update latent/public information with a smooth mixture-strength interpretation."""

    strength = _validate_probability("information strength", strength)
    state.information_event = False
    event_probability = cfg.event_probability * strength
    if event_probability > 0.0 and rng.random() < event_probability:
        shock = rng.gauss(0.0, cfg.shock_std_fraction)
        if abs(shock) < cfg.min_shock_fraction:
            direction = -1.0 if shock < 0.0 else 1.0
            if shock == 0.0:
                direction = -1.0 if rng.random() < 0.5 else 1.0
            shock = direction * cfg.min_shock_fraction
        state.fundamental_value = _price_after_log_move(
            state.fundamental_value, shock, "information shock"
        )
        state.event_intensity = max(
            state.event_intensity,
            min(4.0, abs(shock) / cfg.shock_std_fraction),
        )
        state.information_event = True
    else:
        state.event_intensity *= cfg.event_decay

    # At information strength 0 the fundamental is effectively public immediately; at strength 1
    # the configured slower diffusion applies. This avoids a tiny information weight making the
    # entire baseline fundamental process suddenly opaque.
    effective_diffusion = 1.0 - strength * (1.0 - cfg.public_diffusion_rate)
    log_public = math.log(max(state.public_fundamental, PRICE_EPS))
    log_true = math.log(max(state.fundamental_value, PRICE_EPS))
    log_public += effective_diffusion * (log_true - log_public)
    state.public_fundamental = max(PRICE_EPS, math.exp(log_public))


def _update_liquidity(
    state: SimulationState,
    rng: random.Random,
    cfg: LiquidityWorldConfig,
    regime_profile: RegimeProfile,
    *,
    liquidity_strength: float,
    regime_strength: float,
) -> None:
    """Evolve liquidity around a strength-weighted regime target without recursive drift."""

    liquidity_strength = _validate_probability("liquidity strength", liquidity_strength)
    regime_strength = _validate_probability("regime strength", regime_strength)
    regime_target = _weighted_multiplier(regime_profile.liquidity_multiplier, regime_strength)
    target = _clamp(regime_target, cfg.min_liquidity, cfg.max_liquidity)
    target_log = math.log(target)
    current_log = math.log(_clamp(state.liquidity, cfg.min_liquidity, cfg.max_liquidity))
    # Variance scales approximately linearly with component strength, hence sqrt(strength) on sigma.
    innovation_sigma = cfg.log_liquidity_vol * math.sqrt(liquidity_strength)
    innovation = rng.gauss(0.0, innovation_sigma) if innovation_sigma > 0.0 else 0.0
    effective_persistence = cfg.persistence * liquidity_strength
    next_log = target_log + effective_persistence * (current_log - target_log) + innovation
    state.liquidity = _clamp(math.exp(next_log), cfg.min_liquidity, cfg.max_liquidity)


def _random_world_signal(rng: random.Random, cfg: RandomWorldConfig) -> WorldSignal:
    noise_draw = (-1.0 if rng.random() < 0.5 else 1.0) * cfg.signal_strength
    return WorldSignal(signal=0.0, metadata={"noise_draw": noise_draw})


def _rule_based_world_signal(state: SimulationState, cfg: RuleBasedWorldConfig) -> WorldSignal:
    prices = state.recent_trade_prices
    ma_signal = _moving_average_signal(prices, cfg.short_ma, cfg.long_ma)
    breakout = _breakout_signal(prices, cfg.breakout_lookback)
    contrarian = _zscore_contrarian_signal(prices, cfg.zscore_lookback, cfg.zscore_scale)
    total_weight = cfg.ma_weight + cfg.breakout_weight + cfg.contrarian_weight
    raw = (
        cfg.ma_weight * ma_signal + cfg.breakout_weight * breakout + cfg.contrarian_weight * contrarian
    ) / total_weight
    return WorldSignal(
        signal=raw,
        activity_multiplier=1.0 + cfg.activity_boost * abs(raw),
        metadata={"ma": ma_signal, "breakout": breakout, "contrarian": contrarian},
    )


def _emotional_world_signal(state: SimulationState, cfg: EmotionalWorldConfig) -> WorldSignal:
    intensity = _clamp((state.fomo + state.fear + state.greed) / 8.0, 0.0, 1.0)
    return WorldSignal(
        signal=state.sentiment,
        size_multiplier=1.0 + cfg.size_boost * intensity,
        metadata={"fomo": state.fomo, "fear": state.fear, "greed": state.greed},
    )


def _information_world_signal(state: SimulationState, cfg: InformationWorldConfig) -> WorldSignal:
    informed_target = state.fundamental_value
    public_target = state.public_fundamental
    visible_target = cfg.informed_fraction * informed_target + (1.0 - cfg.informed_fraction) * public_target
    gap_fraction = (visible_target - state.reference_price) / max(state.reference_price, PRICE_EPS)
    signal = _safe_tanh(gap_fraction / cfg.signal_gap_fraction)
    return WorldSignal(
        signal=signal,
        size_multiplier=1.0 + cfg.event_size_boost * min(1.0, state.event_intensity),
        metadata={
            "visible_target": visible_target,
            "public_fundamental": state.public_fundamental,
            "event_intensity": state.event_intensity,
        },
    )


def _mean_reversion_world_signal(state: SimulationState, cfg: MeanReversionWorldConfig) -> WorldSignal:
    signal = _fundamental_reversion_signal(state.reference_price, state.public_fundamental, cfg)
    return WorldSignal(
        signal=signal,
        size_multiplier=1.0 + cfg.size_boost * abs(signal),
        metadata={
            "observable_valuation_gap_fraction": (
                (state.public_fundamental - state.reference_price) / max(state.reference_price, PRICE_EPS)
            )
        },
    )


def _momentum_world_signal(state: SimulationState, cfg: MomentumWorldConfig) -> WorldSignal:
    signal = _momentum_signal(state.recent_trade_prices, cfg)
    return WorldSignal(
        signal=signal,
        size_multiplier=1.0 + cfg.size_boost * abs(signal),
        metadata={"trailing_return": _recent_return(state.recent_trade_prices, cfg.lookback)},
    )


def _regime_world_signal(
    state: SimulationState,
    profile: RegimeProfile,
    cfg: RegimeWorldConfig,
    mean_reversion_cfg: MeanReversionWorldConfig,
) -> WorldSignal:
    signal = profile.signal
    if state.regime == "mean_reverting":
        signal = _fundamental_reversion_signal(
            state.reference_price,
            state.public_fundamental,
            mean_reversion_cfg,
        ) * cfg.regime_signal_strength
    return WorldSignal(
        signal=signal,
        size_multiplier=1.0 + 0.60 * abs(signal),
        metadata={"regime": state.regime},
    )


def _liquidity_world_signal(state: SimulationState, cfg: LiquidityWorldConfig) -> WorldSignal:
    prices = state.recent_trade_prices
    signal = 0.0
    size_boost = 0.0
    mode = "neutral"
    if len(prices) >= 2:
        previous, current = prices[-2], prices[-1]
        interval = _effective_round_interval(current, cfg.round_number_interval)
        recent_direction = _sign(math.log(current / previous)) if previous > 0.0 and current > 0.0 else 0.0
        if _crossed_round_level(previous, current, interval):
            signal = recent_direction * cfg.stop_cascade_strength
            size_boost = cfg.stop_cascade_size_boost
            mode = "stop_cascade"
        else:
            nearest_level = round(current / interval) * interval
            proximity = abs(current - nearest_level) / max(current, PRICE_EPS)
            if proximity <= cfg.round_proximity_fraction and recent_direction != 0.0:
                if current <= nearest_level and recent_direction > 0.0:
                    signal = -cfg.support_resistance_strength
                    mode = "round_resistance"
                elif current >= nearest_level and recent_direction < 0.0:
                    signal = cfg.support_resistance_strength
                    mode = "round_support"
    return WorldSignal(
        signal=signal,
        size_multiplier=1.0 + size_boost * abs(signal),
        metadata={"liquidity_mode": mode, "liquidity": state.liquidity},
    )


def _adaptive_strategy_signals(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
) -> dict[str, float]:
    return {
        "momentum": _momentum_signal(state.recent_trade_prices, config.momentum),
        "mean_reversion": _fundamental_reversion_signal(
            state.reference_price,
            state.public_fundamental,
            config.mean_reversion,
        ),
        "breakout": _breakout_signal(state.recent_trade_prices, config.rule_based.breakout_lookback),
        "noise": -1.0 if rng.random() < 0.5 else 1.0,
    }


def _update_adaptive_scores(state: SimulationState, cfg: AdaptiveWorldConfig) -> None:
    last_return = state.recent_returns[-1] if state.recent_returns else 0.0
    scaled_return = _clamp(last_return / cfg.return_scale, -5.0, 5.0)
    for name in state.adaptive_scores:
        previous_signal = state.adaptive_previous_signals.get(name, 0.0)
        reward = previous_signal * scaled_return
        state.adaptive_scores[name] = cfg.score_decay * state.adaptive_scores[name] + cfg.learning_rate * reward


def _adaptive_world_signal(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
) -> WorldSignal:
    _update_adaptive_scores(state, config.adaptive)
    signals = _adaptive_strategy_signals(state, rng, config)
    weights = _softmax(state.adaptive_scores, config.adaptive.temperature)
    exploration = config.adaptive.exploration_weight
    if weights and exploration > 0.0:
        count = len(weights)
        weights = {
            name: (1.0 - exploration) * weight + exploration / count
            for name, weight in weights.items()
        }
    combined = sum(weights[name] * signals[name] for name in weights)
    state.adaptive_previous_signals = dict(signals)
    metadata: dict[str, float] = {}
    for name in sorted(weights):
        metadata[f"weight_{name}"] = weights[name]
        metadata[f"score_{name}"] = state.adaptive_scores[name]
    return WorldSignal(
        signal=combined,
        size_multiplier=1.0 + config.adaptive.size_boost * abs(combined),
        metadata=metadata,
    )


def _collect_world_signals(
    state: SimulationState,
    streams: RandomStreams,
    config: SyntheticMarketConfig,
    regime_profile: RegimeProfile,
    active_worlds: Sequence[str] | None = None,
) -> dict[str, WorldSignal]:
    signals: dict[str, WorldSignal] = {}
    worlds = tuple(active_worlds) if active_worlds is not None else config.active_worlds
    for world in worlds:
        if world == "random":
            signals[world] = _random_world_signal(streams.world_signal[world], config.random_world)
        elif world == "rule_based":
            signals[world] = _rule_based_world_signal(state, config.rule_based)
        elif world == "emotional":
            signals[world] = _emotional_world_signal(state, config.emotional)
        elif world == "information":
            signals[world] = _information_world_signal(state, config.information)
        elif world == "mean_reversion":
            signals[world] = _mean_reversion_world_signal(state, config.mean_reversion)
        elif world == "momentum":
            signals[world] = _momentum_world_signal(state, config.momentum)
        elif world == "regime":
            signals[world] = _regime_world_signal(state, regime_profile, config.regime, config.mean_reversion)
        elif world == "liquidity":
            signals[world] = _liquidity_world_signal(state, config.liquidity)
        elif world == "adaptive":
            signals[world] = _adaptive_world_signal(state, streams.adaptive_strategy, config)
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled world: {world}")
    return signals


def _aggregate_world_signals(
    world_signals: Mapping[str, WorldSignal],
    normalized_weights: Mapping[str, float],
) -> tuple[float, float, float]:
    aggregate_signal = sum(normalized_weights[name] * world_signals[name].signal for name in world_signals)

    def geometric(attribute: str) -> float:
        return math.exp(
            sum(
                normalized_weights[name] * math.log(max(float(getattr(signal, attribute)), PRICE_EPS))
                for name, signal in world_signals.items()
            )
        )

    return _clamp(aggregate_signal, -1.0, 1.0), geometric("spread_multiplier"), geometric("impact_multiplier")


def _world_signals_json(signals: Mapping[str, WorldSignal]) -> str:
    compact: dict[str, Any] = {}
    for name, signal in signals.items():
        compact[name] = {
            "signal": signal.signal,
            "size_multiplier": signal.size_multiplier,
            "activity_multiplier": signal.activity_multiplier,
            "spread_multiplier": signal.spread_multiplier,
            "impact_multiplier": signal.impact_multiplier,
            **dict(signal.metadata),
        }
    return _canonical_json(compact)


def _world_flow_json(flow: Mapping[str, Mapping[str, int]]) -> str:
    return _canonical_json({name: dict(values) for name, values in flow.items()})


def _update_fundamental(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
    profile: RegimeProfile,
    regime_strength: float,
) -> None:
    regime_strength = _validate_probability("regime strength", regime_strength)
    drift = config.effective_fundamental_drift_fraction + regime_strength * profile.drift_fraction
    volatility_multiplier = _weighted_multiplier(profile.volatility_multiplier, regime_strength)
    volatility = config.effective_fundamental_vol_fraction * volatility_multiplier
    innovation = rng.gauss(0.0, volatility) if volatility > 0.0 else 0.0
    log_increment = drift - 0.5 * volatility * volatility + innovation
    state.fundamental_value = _price_after_log_move(
        state.fundamental_value, log_increment, "fundamental evolution"
    )


def _update_reference_before_quote(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
    regime_strength: float,
) -> None:
    if config.is_strict_random_null:
        state.reference_price = state.fundamental_value
        return
    regime_strength = _validate_probability("regime strength", regime_strength)
    anchor = config.fundamental_anchor_strength
    if state.regime == "mean_reverting" and regime_strength > 0.0:
        anchor = min(1.0, anchor * (1.0 + 2.0 * regime_strength))
    # Anchor only to public information. The hidden fundamental can affect price only through the
    # explicitly informed trader population in the information world.
    target = max(state.public_fundamental, PRICE_EPS)
    log_reference = math.log(max(state.reference_price, PRICE_EPS))
    log_reference += anchor * (math.log(target) - log_reference)
    if config.microstructure_noise_fraction > 0.0:
        log_reference += rng.gauss(0.0, config.microstructure_noise_fraction)
    state.reference_price = max(PRICE_EPS, math.exp(log_reference))


def _apply_order_impact(
    reference_price: float,
    signed_order_flow: int,
    average_base_order_size: float,
    impact_multiplier: float,
    config: SyntheticMarketConfig,
    accumulated_step_log_move: float,
) -> tuple[float, float]:
    """Apply one fill's impact without exceeding the configured cumulative step cap."""

    if config.is_strict_random_null or config.order_impact_fraction <= 0.0 or signed_order_flow == 0:
        return reference_price, 0.0
    normalized_flow = signed_order_flow / max(average_base_order_size, 1.0)
    log_move = config.order_impact_fraction * normalized_flow * impact_multiplier
    limit = config.max_single_step_mid_move_fraction
    log_move = _clamp(
        log_move,
        -limit - accumulated_step_log_move,
        limit - accumulated_step_log_move,
    )
    return _price_after_log_move(reference_price, log_move, "order impact"), log_move


def _record_observed_price(state: SimulationState, price: float, max_history: int) -> None:
    price = _validate_positive("observed price", price)
    if state.recent_trade_prices:
        previous = state.recent_trade_prices[-1]
        state.recent_returns.append(math.log(price / previous))
    state.recent_trade_prices.append(price)
    state.last_observed_price = price
    if len(state.recent_trade_prices) > max_history:
        del state.recent_trade_prices[:-max_history]
    if len(state.recent_returns) > max_history:
        del state.recent_returns[:-max_history]


def _event_buy_probability(
    world: str,
    signal: WorldSignal,
    rng: random.Random,
    config: SyntheticMarketConfig,
) -> tuple[float, str]:
    if world == "random":
        side = "buy" if rng.random() < 0.5 else "sell"
        return 0.5, side
    if (
        world == "rule_based"
        and config.rule_based.strict_execution
        and abs(signal.signal) >= config.rule_based.strict_signal_threshold
    ):
        if signal.signal > 0.0:
            return 1.0, "buy"
        if signal.signal < 0.0:
            return 0.0, "sell"
    noisy_score = config.decision_signal_strength * signal.signal
    if config.decision_noise > 0.0:
        noisy_score += rng.gauss(0.0, config.decision_noise)
    probability = _sigmoid(noisy_score)
    side = "buy" if rng.random() < probability else "sell"
    return probability, side


def _generate_order_intents(
    signals: Mapping[str, WorldSignal],
    streams: RandomStreams,
    config: SyntheticMarketConfig,
    normalized_weights: Mapping[str, float] | None = None,
) -> tuple[list[OrderIntent], dict[str, bool]]:
    intents: list[OrderIntent] = []
    capped: dict[str, bool] = {}
    normalized = normalized_weights if normalized_weights is not None else config.normalized_world_weights
    maximum_size = max(1, int(math.ceil(config.max_order_size * config.max_order_size_multiplier)))
    for world in config.active_worlds:
        signal = signals[world]
        arrival_rate = config.base_arrival_rate * normalized[world] * signal.activity_multiplier
        count, was_capped = _poisson_count(
            streams.world_arrival[world],
            arrival_rate,
            config.max_arrivals_per_world_per_step,
        )
        capped[world] = was_capped
        for _ in range(count):
            probability, side = _event_buy_probability(world, signal, streams.world_decision[world], config)
            base_size = streams.world_size[world].randint(config.min_order_size, config.max_order_size)
            requested = max(1, min(maximum_size, int(round(base_size * signal.size_multiplier))))
            intents.append(
                OrderIntent(
                    source_world=world,
                    taker_side=side,
                    requested_size=requested,
                    signal=signal.signal,
                    buy_probability=probability,
                    impact_multiplier=signal.impact_multiplier,
                )
            )
    streams.execution_order.shuffle(intents)
    return intents, capped


def _config_payload(config: SyntheticMarketConfig, *, include_seed: bool = True) -> dict[str, Any]:
    payload = _json_ready(config)
    payload["world_weights"] = dict(config.world_weights)
    payload["normalized_world_weights"] = dict(config.normalized_world_weights)
    payload["active_worlds"] = list(config.active_worlds)
    payload["effective_fundamental_vol_fraction"] = config.effective_fundamental_vol_fraction
    payload["effective_fundamental_drift_fraction"] = config.effective_fundamental_drift_fraction
    payload["effective_base_spread_fraction"] = config.effective_base_spread_fraction
    payload["effective_inventory_skew_fraction"] = config.effective_inventory_skew_fraction
    payload["simulator_version"] = SIMULATOR_VERSION
    payload["rng_stream_version"] = RNG_STREAM_VERSION
    if not include_seed:
        payload.pop("seed", None)
    return payload


_BEHAVIOR_CONFIG_FIELDS = {
    "random_world": "random",
    "rule_based": "rule_based",
    "emotional": "emotional",
    "information": "information",
    "mean_reversion": "mean_reversion",
    "momentum": "momentum",
    "regime": "regime",
    "liquidity": "liquidity",
    "adaptive": "adaptive",
}
_BEHAVIOR_CROSS_FIELDS = {
    "adaptive": {
        "rule_based": frozenset({"breakout_lookback"}),
        "momentum": frozenset({"lookback", "return_scale", "deadband"}),
        "mean_reversion": frozenset({"deviation_scale_fraction", "deadband_fraction"}),
    },
    "regime": {
        "mean_reversion": frozenset({"deviation_scale_fraction", "deadband_fraction"}),
    },
}


def canonical_behavior_config(config: SyntheticMarketConfig) -> SyntheticMarketConfig:
    """Canonicalize every configuration representation that produces the same market path.

    The original, user-authored config is still written as provenance.  IDs and experimental
    structure grouping use this behavioral form so inactive knobs, shadowed legacy values, and
    proportional population weights cannot create false independent scenarios.
    """

    active = set(config.active_worlds)
    relevance: dict[str, frozenset[str] | None] = {}
    for config_field, world_name in _BEHAVIOR_CONFIG_FIELDS.items():
        # Random signal strength changes diagnostic noise metadata only, not orders or prices.
        if world_name in active and world_name != "random":
            relevance[config_field] = None
    for active_world, dependencies in _BEHAVIOR_CROSS_FIELDS.items():
        if active_world not in active:
            continue
        for config_field, names in dependencies.items():
            if relevance.get(config_field) is None and config_field in relevance:
                continue
            relevance[config_field] = frozenset(
                set(relevance.get(config_field, frozenset()) or ()) | set(names)
            )

    nested: dict[str, Any] = {}
    for config_field in _BEHAVIOR_CONFIG_FIELDS:
        current = getattr(config, config_field)
        default = type(current)()
        names = relevance.get(config_field)
        if names is None and config_field in relevance:
            nested[config_field] = current
        elif names:
            nested[config_field] = replace(
                default,
                **{name: getattr(current, name) for name in names},
            )
        else:
            nested[config_field] = default

    defaults = SyntheticMarketConfig()
    strict_random = bool(config.strict_random_null) if config.is_random_only else False
    random_mode = config.random_null_mode if strict_random else defaults.random_null_mode
    global_updates: dict[str, Any] = {
        "world_weights": dict(config.normalized_world_weights),
        "fair_value_step_vol": defaults.fair_value_step_vol,
        "fair_value_step_vol_fraction": config.effective_fundamental_vol_fraction,
        "base_fundamental_drift": defaults.base_fundamental_drift,
        "base_fundamental_drift_fraction": config.effective_fundamental_drift_fraction,
        "base_spread": defaults.base_spread,
        "base_spread_fraction": config.effective_base_spread_fraction,
        "inventory_skew": defaults.inventory_skew,
        "inventory_skew_fraction": config.effective_inventory_skew_fraction,
        "strict_random_null": strict_random,
        "random_null_mode": random_mode,
        **nested,
    }
    if config.is_random_only:
        # Random-world sides are explicit fair coins and bypass the score/noise decision model.
        global_updates["decision_signal_strength"] = defaults.decision_signal_strength
        global_updates["decision_noise"] = defaults.decision_noise
    if strict_random:
        # These mechanisms are bypassed by both strict-null modes.
        for name in (
            "order_impact_fraction",
            "max_single_step_mid_move_fraction",
            "fundamental_anchor_strength",
            "microstructure_noise_fraction",
        ):
            global_updates[name] = getattr(defaults, name)
    if strict_random and random_mode == "efficient":
        # Efficient nulls execute at the fundamental with zero spread, so quote-shaping inputs are
        # observationally irrelevant even though inventory/volume mechanics remain active.
        global_updates.update(
            {
                "base_spread": defaults.base_spread,
                "base_spread_fraction": defaults.effective_base_spread_fraction,
                "min_tick": defaults.min_tick,
                "inventory_skew": defaults.inventory_skew,
                "inventory_skew_fraction": defaults.effective_inventory_skew_fraction,
                "inventory_skew_cap_fraction": defaults.inventory_skew_cap_fraction,
                "inventory_spread_sensitivity": defaults.inventory_spread_sensitivity,
            }
        )
    return replace(config, **global_updates)


def _identity_payload(config: SyntheticMarketConfig, *, include_seed: bool) -> dict[str, Any]:
    """Return a behavioral identity payload with relative world weights canonicalized."""

    return _config_payload(canonical_behavior_config(config), include_seed=include_seed)


def scenario_id(config: SyntheticMarketConfig) -> str:
    digest = hashlib.sha256(
        _canonical_json(_identity_payload(config, include_seed=False)).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def run_id(config: SyntheticMarketConfig) -> str:
    digest = hashlib.sha256(
        _canonical_json(_identity_payload(config, include_seed=True)).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def candle_dataset_id(
    config: SyntheticMarketConfig,
    steps_per_candle: int,
    *,
    include_partial: bool = False,
) -> str:
    """Fingerprint one detector-ready candle dataset construction."""

    _positive_int("steps_per_candle", steps_per_candle)
    if not isinstance(include_partial, bool):
        raise ValueError("include_partial must be boolean")
    payload = {
        "run_id": run_id(config),
        "steps_per_candle": steps_per_candle,
        "include_partial": include_partial,
        "candle_builder_version": CANDLE_BUILDER_VERSION,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def simulate_market(
    config: SyntheticMarketConfig,
    *,
    record_step_diagnostics: bool = True,
) -> list[MarketSnapshot]:
    """Simulate one deterministic-by-seed causal market path.

    ``record_step_diagnostics`` controls only the expensive per-step JSON diagnostic payloads
    (``world_signals_json`` and ``world_flow_json``).  It never changes prices, order flow,
    inventory, P&L, run/scenario identity, or candle construction.  Large Monte Carlo and AI
    sweeps should normally set it to ``False``; single-path forensic runs can keep the default.
    """

    if not isinstance(record_step_diagnostics, bool):
        raise ValueError("record_step_diagnostics must be boolean")

    streams = RandomStreams.from_seed(config.seed)
    maker = MarketMaker(
        base_spread_fraction=config.effective_base_spread_fraction,
        inventory_skew_fraction=config.effective_inventory_skew_fraction,
        min_tick=config.min_tick,
        inventory_skew_cap_fraction=config.inventory_skew_cap_fraction,
        inventory_spread_sensitivity=config.inventory_spread_sensitivity,
        max_abs_inventory=config.max_abs_inventory,
    )
    state = SimulationState(
        fundamental_value=config.initial_fair_value,
        reference_price=config.initial_fair_value,
        public_fundamental=config.initial_fair_value,
        last_observed_price=config.initial_fair_value,
        rolling_peak_price=config.initial_fair_value,
    )
    history: list[MarketSnapshot] = []
    average_base_order_size = (config.min_order_size + config.max_order_size) / 2.0
    # These are immutable for a run.  Precomputing them removes a surprising amount of Python
    # allocation/property work from large Monte Carlo sweeps without changing the stochastic path.
    active_worlds = config.active_worlds
    active_world_set = frozenset(active_worlds)
    normalized_weights = config.normalized_world_weights
    regime_strength = float(normalized_weights.get("regime", 0.0))
    information_strength = float(normalized_weights.get("information", 0.0))
    liquidity_strength = float(normalized_weights.get("liquidity", 0.0))
    required_history_length = config.required_history_length
    active_worlds_label = "+".join(active_worlds)
    weights_json = _canonical_json(dict(config.world_weights))
    scenario = scenario_id(config)
    run = run_id(config)

    for step in range(1, config.steps + 1):
        # Persistent behavioral states use only observations completed before this step.
        if "emotional" in active_world_set:
            _update_emotions(state, config.emotional)

        if "regime" in active_world_set:
            regime_profile = _update_regime(state, streams.regime, config.regime)
        else:
            state.regime = "none"
            regime_profile = RegimeProfile()

        _update_fundamental(
            state, streams.fundamental, config, regime_profile, regime_strength
        )

        if information_strength > 0.0:
            _update_information_environment(
                state, streams.information, config.information, information_strength
            )
        else:
            state.information_event = False
            state.event_intensity = 0.0
            state.public_fundamental = state.fundamental_value

        if liquidity_strength > 0.0:
            _update_liquidity(
                state,
                streams.liquidity,
                config.liquidity,
                regime_profile,
                liquidity_strength=liquidity_strength,
                regime_strength=regime_strength,
            )
        else:
            state.liquidity = _clamp(
                _weighted_multiplier(regime_profile.liquidity_multiplier, regime_strength),
                0.25,
                4.0,
            )

        _update_reference_before_quote(
            state, streams.microstructure, config, regime_strength
        )
        quote_reference_price = state.reference_price

        world_signals = _collect_world_signals(
            state,
            streams,
            config,
            regime_profile,
            active_worlds=active_worlds,
        )
        aggregate_signal, aggregate_spread_multiplier, aggregate_impact_multiplier = _aggregate_world_signals(
            world_signals,
            normalized_weights,
        )

        liquidity = max(state.liquidity, PRICE_EPS)
        aggregate_spread_multiplier *= 1.0 / math.sqrt(liquidity)
        aggregate_impact_multiplier *= 1.0 / liquidity
        if regime_strength > 0.0:
            aggregate_spread_multiplier *= math.sqrt(
                _weighted_multiplier(regime_profile.volatility_multiplier, regime_strength)
            )

        if config.is_efficient_random_null:
            initial_bid = initial_ask = state.fundamental_value
        else:
            initial_bid, initial_ask = maker.quote(
                quote_reference_price,
                spread_multiplier=aggregate_spread_multiplier,
            )

        intents, capped_worlds = _generate_order_intents(
            world_signals,
            streams,
            config,
            normalized_weights=normalized_weights,
        )
        executions: list[float] = []
        buy_volume = 0
        sell_volume = 0
        rejected_volume = 0
        filled_trades = 0
        intent_probability_sum = 0.0
        accumulated_step_impact = 0.0
        flow_by_world: dict[str, dict[str, int]] | None = None
        if record_step_diagnostics:
            flow_by_world = {
                world: {
                    "arrivals": 0,
                    "trades": 0,
                    "buy_volume": 0,
                    "sell_volume": 0,
                    "rejected_volume": 0,
                }
                for world in active_worlds
            }

        for intent in intents:
            if flow_by_world is not None:
                flow_by_world[intent.source_world]["arrivals"] += 1
            intent_probability_sum += intent.buy_probability
            if config.is_efficient_random_null:
                execution_price = state.fundamental_value
            else:
                bid, ask = maker.quote(
                    state.reference_price,
                    spread_multiplier=aggregate_spread_multiplier,
                )
                execution_price = ask if intent.taker_side == "buy" else bid

            filled = maker.execute_taker(intent.taker_side, execution_price, intent.requested_size)
            rejected = intent.requested_size - filled
            if rejected:
                rejected_volume += rejected
                if flow_by_world is not None:
                    flow_by_world[intent.source_world]["rejected_volume"] += rejected
            if filled <= 0:
                continue

            filled_trades += 1
            if flow_by_world is not None:
                flow_by_world[intent.source_world]["trades"] += 1
            executions.append(execution_price)
            signed_flow = filled if intent.taker_side == "buy" else -filled
            if intent.taker_side == "buy":
                buy_volume += filled
                if flow_by_world is not None:
                    flow_by_world[intent.source_world]["buy_volume"] += filled
            else:
                sell_volume += filled
                if flow_by_world is not None:
                    flow_by_world[intent.source_world]["sell_volume"] += filled

            state.reference_price, applied_impact = _apply_order_impact(
                state.reference_price,
                signed_flow,
                average_base_order_size,
                aggregate_impact_multiplier * intent.impact_multiplier,
                config,
                accumulated_step_impact,
            )
            accumulated_step_impact += applied_impact
            if abs(accumulated_step_impact) > config.max_single_step_mid_move_fraction + 1e-15:
                raise AssertionError("cumulative order impact exceeded the configured step cap")

        hedge_signed, hedge_cost = maker.hedge_excess(
            state.reference_price,
            trigger_fraction=config.inventory_hedge_trigger_fraction,
            hedge_fraction=config.inventory_hedge_fraction,
            cost_fraction=config.hedge_cost_fraction,
        )
        hedge_volume = abs(hedge_signed)

        # Store the executable quote at the end of the step as well as the opening quote.  This is
        # diagnostic/read-only and consumes no randomness, so it cannot alter the simulated path.
        if config.is_efficient_random_null:
            post_trade_bid = post_trade_ask = state.fundamental_value
        else:
            post_trade_bid, post_trade_ask = maker.quote(
                state.reference_price,
                spread_multiplier=aggregate_spread_multiplier,
            )

        traded_volume = buy_volume + sell_volume
        signed_order_flow = buy_volume - sell_volume
        if signed_order_flow > 0:
            net_side = "buy"
        elif signed_order_flow < 0:
            net_side = "sell"
        else:
            net_side = "neutral"

        if config.is_efficient_random_null:
            observed_close = state.fundamental_value
            if executions:
                executions = [state.fundamental_value for _ in executions]
        elif executions:
            observed_close = executions[-1]
        else:
            observed_close = state.last_observed_price

        if executions:
            step_open = executions[0]
            step_high = max(executions)
            step_low = min(executions)
            step_close = executions[-1] if not config.is_efficient_random_null else observed_close
        else:
            step_open = step_high = step_low = step_close = observed_close

        _record_observed_price(state, observed_close, required_history_length)
        average_probability = intent_probability_sum / len(intents) if intents else 0.5

        # A cap is a safety event and should be visible in stored diagnostics, never silent.
        if flow_by_world is not None:
            for world, was_capped in capped_worlds.items():
                if was_capped:
                    flow_by_world[world]["arrival_cap_hit"] = 1

        history.append(
            MarketSnapshot(
                step=step,
                fair_value=state.fundamental_value,
                reference_price=quote_reference_price,
                post_trade_reference_price=state.reference_price,
                bid=initial_bid,
                ask=initial_ask,
                taker_side=net_side,
                order_size=traded_volume,
                trade_price=observed_close,
                maker_inventory=maker.inventory,
                maker_cash=maker.cash,
                maker_pnl=maker.mark_to_market_pnl(state.fundamental_value),
                post_trade_bid=post_trade_bid,
                post_trade_ask=post_trade_ask,
                maker_pnl_reference=maker.mark_to_market_pnl(state.reference_price),
                aggregate_signal=aggregate_signal,
                buy_probability=average_probability,
                signed_order_flow=signed_order_flow,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                trade_count=filled_trades,
                rejected_volume=rejected_volume,
                arrival_cap_hit=any(capped_worlds.values()),
                hedge_volume=hedge_volume,
                hedge_cost=hedge_cost,
                liquidity=state.liquidity,
                regime=state.regime,
                sentiment=state.sentiment,
                fomo=state.fomo,
                fear=state.fear,
                greed=state.greed,
                public_fundamental=state.public_fundamental,
                information_gap=state.fundamental_value - state.public_fundamental,
                information_event=state.information_event,
                step_open=step_open,
                step_high=step_high,
                step_low=step_low,
                step_close=step_close,
                traded_volume=traded_volume,
                active_worlds=active_worlds_label,
                simulation_seed=config.seed,
                scenario_id=scenario,
                run_id=run,
                world_weights_json=weights_json,
                world_signals_json=(
                    _world_signals_json(world_signals) if record_step_diagnostics else "{}"
                ),
                world_flow_json=(
                    _world_flow_json(flow_by_world) if flow_by_world is not None else "{}"
                ),
            )
        )

    return history


def simulate_random_market(
    config: RandomMarketConfig,
    *,
    record_step_diagnostics: bool = True,
) -> list[MarketSnapshot]:
    """Backward-compatible random-only entry point."""
    return simulate_market(
        config.to_synthetic_config(),
        record_step_diagnostics=record_step_diagnostics,
    )


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    mean_x = mean(x)
    mean_y = mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def _skewness(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    center = mean(values)
    std = pstdev(values)
    if std <= 0.0:
        return 0.0
    return mean(((value - center) / std) ** 3 for value in values)


def _excess_kurtosis(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    center = mean(values)
    std = pstdev(values)
    if std <= 0.0:
        return 0.0
    return mean(((value - center) / std) ** 4 for value in values) - 3.0


def summarize(history: Sequence[MarketSnapshot]) -> dict[str, float]:
    if not history:
        return {}
    prices = [snapshot.close for snapshot in history]
    returns = [
        math.log(prices[index] / prices[index - 1])
        for index in range(1, len(prices))
        if prices[index - 1] > 0.0 and prices[index] > 0.0
    ]
    pnl_path = [snapshot.maker_pnl for snapshot in history]
    # The maker starts with zero cash and zero inventory, so drawdown must include the
    # pre-trading P&L baseline rather than beginning at the first post-trade observation.
    peak_pnl = 0.0
    max_drawdown = 0.0
    for pnl in pnl_path:
        peak_pnl = max(peak_pnl, pnl)
        max_drawdown = max(max_drawdown, peak_pnl - pnl)

    buy_volume = sum(snapshot.buy_volume for snapshot in history)
    sell_volume = sum(snapshot.sell_volume for snapshot in history)
    total_volume = buy_volume + sell_volume
    total_trades = sum(snapshot.trade_count for snapshot in history)
    rejected = sum(snapshot.rejected_volume for snapshot in history)
    arrival_cap_steps = sum(int(snapshot.arrival_cap_hit) for snapshot in history)
    hedge_volume = sum(snapshot.hedge_volume for snapshot in history)
    information_events = sum(1 for snapshot in history if snapshot.information_event)
    log_price_errors = [
        math.log(snapshot.close / snapshot.fair_value)
        for snapshot in history
        if snapshot.close > 0.0 and snapshot.fair_value > 0.0
    ]
    illiquidity_terms = [
        abs(returns[index - 1]) / history[index].traded_volume
        for index in range(1, len(history))
        if history[index].traded_volume > 0 and index - 1 < len(returns)
    ]
    return {
        "steps": float(len(history)),
        "final_fair_value": history[-1].fair_value,
        "final_trade_price": history[-1].close,
        "final_reference_price": history[-1].post_trade_reference_price,
        "final_maker_inventory": float(history[-1].maker_inventory),
        "max_abs_maker_inventory": float(max(abs(snapshot.maker_inventory) for snapshot in history)),
        "final_maker_pnl": history[-1].maker_pnl,
        "average_spread": mean(snapshot.spread for snapshot in history),
        "average_spread_fraction": mean(
            snapshot.spread / max(snapshot.reference_price, PRICE_EPS) for snapshot in history
        ),
        "average_abs_inventory": mean(abs(snapshot.maker_inventory) for snapshot in history),
        "realized_volatility_per_step": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_log_return_per_step": mean(returns) if returns else 0.0,
        "return_lag1_autocorrelation": _lag1_autocorrelation(returns),
        "return_skewness": _skewness(returns),
        "return_excess_kurtosis": _excess_kurtosis(returns),
        "max_pnl_drawdown": max_drawdown,
        "buy_volume": float(buy_volume),
        "sell_volume": float(sell_volume),
        "signed_order_flow": float(buy_volume - sell_volume),
        "total_volume": float(total_volume),
        "trade_count": float(total_trades),
        "rejected_volume": float(rejected),
        "arrival_cap_steps": float(arrival_cap_steps),
        "hedge_volume": float(hedge_volume),
        "average_liquidity": mean(snapshot.liquidity for snapshot in history),
        "average_abs_signal": mean(abs(snapshot.aggregate_signal) for snapshot in history),
        "average_buy_probability": mean(snapshot.buy_probability for snapshot in history),
        "information_events": float(information_events),
        "price_fundamental_log_rmse": math.sqrt(mean(error * error for error in log_price_errors)) if log_price_errors else 0.0,
        "amihud_like_illiquidity": mean(illiquidity_terms) if illiquidity_terms else 0.0,
    }


def write_config_json(
    config: SyntheticMarketConfig,
    output_path: Path,
    *,
    experiment_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist an exact simulator specification, identifiers and optional construction metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(config, include_seed=True)
    payload["scenario_id"] = scenario_id(config)
    payload["run_id"] = run_id(config)
    payload["candle_builder_version"] = CANDLE_BUILDER_VERSION
    payload["renderer_version"] = RENDERER_VERSION
    if experiment_metadata is not None:
        if not isinstance(experiment_metadata, Mapping):
            raise ValueError("experiment_metadata must be a mapping when supplied")
        payload["experiment_metadata"] = _json_ready(experiment_metadata)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("cannot export NaN or infinity")
    return format(value, ".17g")


def write_history_csv(history: Sequence[MarketSnapshot], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "open", "high", "low", "close", "volume",
        "fair_value", "reference_price", "post_trade_reference_price", "bid", "ask", "spread",
        "post_trade_bid", "post_trade_ask", "post_trade_spread",
        "taker_side", "order_size", "signed_order_flow", "buy_volume", "sell_volume", "trade_count",
        "rejected_volume", "arrival_cap_hit", "hedge_volume", "hedge_cost", "trade_price", "maker_inventory", "maker_cash",
        "maker_pnl", "maker_pnl_reference", "aggregate_signal", "buy_probability", "liquidity", "regime",
        "sentiment", "fomo", "fear", "greed", "public_fundamental", "information_gap",
        "information_event", "active_worlds", "simulation_seed", "scenario_id", "run_id",
        "world_weights_json", "world_signals_json", "world_flow_json", "simulator_version",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in history:
            row: dict[str, Any] = {
                "step": snapshot.step,
                "open": _format_float(snapshot.open),
                "high": _format_float(snapshot.high),
                "low": _format_float(snapshot.low),
                "close": _format_float(snapshot.close),
                "volume": snapshot.volume,
                "fair_value": _format_float(snapshot.fair_value),
                "reference_price": _format_float(snapshot.reference_price),
                "post_trade_reference_price": _format_float(snapshot.post_trade_reference_price),
                "bid": _format_float(snapshot.bid),
                "ask": _format_float(snapshot.ask),
                "spread": _format_float(snapshot.spread),
                "post_trade_bid": _format_float(snapshot.post_trade_bid),
                "post_trade_ask": _format_float(snapshot.post_trade_ask),
                "post_trade_spread": _format_float(snapshot.post_trade_spread),
                "taker_side": snapshot.taker_side,
                "order_size": snapshot.order_size,
                "signed_order_flow": snapshot.signed_order_flow,
                "buy_volume": snapshot.buy_volume,
                "sell_volume": snapshot.sell_volume,
                "trade_count": snapshot.trade_count,
                "rejected_volume": snapshot.rejected_volume,
                "arrival_cap_hit": int(snapshot.arrival_cap_hit),
                "hedge_volume": snapshot.hedge_volume,
                "hedge_cost": _format_float(snapshot.hedge_cost),
                "trade_price": _format_float(snapshot.trade_price),
                "maker_inventory": snapshot.maker_inventory,
                "maker_cash": _format_float(snapshot.maker_cash),
                "maker_pnl": _format_float(snapshot.maker_pnl),
                "maker_pnl_reference": _format_float(snapshot.maker_pnl_reference),
                "aggregate_signal": _format_float(snapshot.aggregate_signal),
                "buy_probability": _format_float(snapshot.buy_probability),
                "liquidity": _format_float(snapshot.liquidity),
                "regime": snapshot.regime,
                "sentiment": _format_float(snapshot.sentiment),
                "fomo": _format_float(snapshot.fomo),
                "fear": _format_float(snapshot.fear),
                "greed": _format_float(snapshot.greed),
                "public_fundamental": _format_float(snapshot.public_fundamental),
                "information_gap": _format_float(snapshot.information_gap),
                "information_event": int(snapshot.information_event),
                "active_worlds": snapshot.active_worlds,
                "simulation_seed": snapshot.simulation_seed,
                "scenario_id": snapshot.scenario_id,
                "run_id": snapshot.run_id,
                "world_weights_json": snapshot.world_weights_json,
                "world_signals_json": snapshot.world_signals_json,
                "world_flow_json": snapshot.world_flow_json,
                "simulator_version": snapshot.simulator_version,
            }
            writer.writerow(row)


def write_candles_csv(
    candles: Sequence[PriceCandle],
    output_path: Path,
    *,
    dataset_id: str | None = None,
) -> None:
    """Export detector-ready OHLCV candles without precision-destroying decimal rounding."""

    if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id.strip()):
        raise ValueError("dataset_id must be a non-empty string when supplied")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candle_number", "start_step", "end_step", "open", "high", "low", "close",
        "volume", "is_complete",
    ]
    if dataset_id is not None:
        fieldnames.append("dataset_id")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for candle in candles:
            row: dict[str, Any] = {
                "candle_number": candle.candle_number,
                "start_step": candle.start_step,
                "end_step": candle.end_step,
                "open": _format_float(candle.open),
                "high": _format_float(candle.high),
                "low": _format_float(candle.low),
                "close": _format_float(candle.close),
                "volume": candle.volume,
                "is_complete": int(candle.is_complete),
            }
            if dataset_id is not None:
                row["dataset_id"] = dataset_id
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    run: int
    seed: int
    scenario_id: str
    run_id: str
    summary: Mapping[str, float]


def run_monte_carlo(
    config: SyntheticMarketConfig,
    runs: int,
    *,
    seed_start: int | None = None,
) -> list[MonteCarloResult]:
    if isinstance(runs, bool) or not isinstance(runs, int) or runs <= 0:
        raise ValueError("runs must be an integer > 0")
    first_seed = config.seed if seed_start is None else seed_start
    if isinstance(first_seed, bool) or not isinstance(first_seed, int):
        raise ValueError("seed_start must be an integer")
    results: list[MonteCarloResult] = []
    for run_index in range(runs):
        seed = first_seed + run_index
        run_config = replace(config, seed=seed)
        history = simulate_market(run_config, record_step_diagnostics=False)
        results.append(
            MonteCarloResult(
                run=run_index + 1,
                seed=seed,
                scenario_id=scenario_id(run_config),
                run_id=run_id(run_config),
                summary=_FrozenMapping.from_mapping(summarize(history)),
            )
        )
    return results


def summarize_monte_carlo(results: Sequence[MonteCarloResult]) -> dict[str, float]:
    if not results:
        return {}
    shared_keys = sorted(set.intersection(*(set(result.summary) for result in results)))
    output: dict[str, float] = {"runs": float(len(results))}
    for key in shared_keys:
        values = [float(result.summary[key]) for result in results]
        output[f"mean_{key}"] = mean(values)
        output[f"sd_{key}"] = pstdev(values) if len(values) > 1 else 0.0
    return output


def write_monte_carlo_csv(results: Sequence[MonteCarloResult], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        output_path.write_text("run,seed,scenario_id,run_id\n", encoding="utf-8")
        return
    shared_keys = sorted(set.intersection(*(set(result.summary) for result in results)))
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["run", "seed", "scenario_id", "run_id", *shared_keys])
        writer.writeheader()
        for result in results:
            row = {
                "run": result.run,
                "seed": result.seed,
                "scenario_id": result.scenario_id,
                "run_id": result.run_id,
                **{key: _format_float(float(result.summary[key])) for key in shared_keys},
            }
            writer.writerow(row)


def parse_world_specification(
    worlds_text: str,
    weight_specs: Sequence[str] | None = None,
) -> dict[str, float]:
    names = [part.strip() for part in worlds_text.split(",") if part.strip()]
    if not names:
        raise ValueError("--worlds must contain at least one world")
    weights: dict[str, float] = {}
    for raw_name in names:
        name = _normalize_world_name(raw_name)
        weights[name] = 1.0
    for specification in weight_specs or ():
        if "=" not in specification:
            raise ValueError(f"Invalid --world-weight {specification!r}; expected WORLD=WEIGHT")
        raw_name, raw_weight = specification.split("=", 1)
        name = _normalize_world_name(raw_name)
        if name not in weights:
            raise ValueError(f"Weight specified for inactive world {name!r}; add it to --worlds first")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid weight in {specification!r}") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"World weights must be finite and > 0, got {weight}")
        weights[name] = weight
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causal heterogeneous-agent synthetic market-maker simulator."
    )
    parser.add_argument(
        "--worlds",
        default="random",
        help="Comma-separated worlds. Valid: " + ", ".join(WORLD_NAMES) + ". Alias 'mathematical' maps to rule_based.",
    )
    parser.add_argument("--world-weight", action="append", default=[], metavar="WORLD=WEIGHT", help="Relative mechanism/population weight for an active world; repeatable.")
    parser.add_argument("--list-worlds", action="store_true")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-fair-value", type=float, default=100.0)
    parser.add_argument("--fair-value-step-vol", type=float, default=0.03, help="Legacy absolute volatility scale; overridden by --fair-value-step-vol-fraction.")
    parser.add_argument("--fair-value-step-vol-fraction", type=float, default=None, help="Preferred scale-free log-volatility per simulation step.")
    parser.add_argument("--base-spread", type=float, default=0.04, help="Legacy absolute spread at the initial price.")
    parser.add_argument("--base-spread-fraction", type=float, default=None, help="Preferred scale-free base spread fraction.")
    parser.add_argument("--inventory-skew", type=float, default=0.002, help="Legacy absolute inventory skew at the initial price.")
    parser.add_argument("--inventory-skew-fraction", type=float, default=None, help="Preferred scale-free inventory skew fraction per unit inventory.")
    parser.add_argument("--max-abs-inventory", type=int, default=500)
    parser.add_argument("--min-order-size", type=int, default=1)
    parser.add_argument("--max-order-size", type=int, default=10)
    parser.add_argument("--base-arrival-rate", type=float, default=2.0, help="Expected total baseline trader arrivals per step before state-dependent activity multipliers.")
    parser.add_argument("--random-null-mode", choices=("efficient", "microstructure"), default="efficient")
    parser.add_argument("--no-strict-random-null", action="store_true", help="Allow permanent impact/feedback in a random-only run; this is no longer a formal null.")
    parser.add_argument("--candle-steps", type=int, default=5)
    parser.add_argument("--include-partial-candle", action="store_true", help="Keep the final short candle. Disabled by default for fixed-duration backtests.")
    parser.add_argument("--visible-candles", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_market.csv"))
    parser.add_argument("--candles-output", type=Path, default=Path("results/synthetic_market_candles.csv"))
    parser.add_argument("--config-output", type=Path, default=None, help="Scenario metadata JSON. Defaults beside the primary output.")
    parser.add_argument("--chart-output", type=Path, default=Path("results/synthetic_market_candles.html"))
    parser.add_argument("--monte-carlo-runs", type=int, default=0, help="If >0, run this many seeds instead of producing a single charted path.")
    parser.add_argument("--monte-carlo-output", type=Path, default=Path("results/monte_carlo_summary.csv"))
    return parser.parse_args()


def _chart_title(config: SyntheticMarketConfig) -> str:
    world_text = " + ".join(name.replace("_", " ").title() for name in config.active_worlds)
    null_suffix = f" · {config.random_null_mode.title()} Null" if config.is_strict_random_null else ""
    return f"Synthetic Market — {world_text}{null_suffix}"


def main() -> None:
    args = parse_args()
    if args.list_worlds:
        print("Available worlds:")
        for name in WORLD_NAMES:
            print(f"  - {name}")
        return

    world_weights = parse_world_specification(args.worlds, args.world_weight)
    config = SyntheticMarketConfig(
        steps=args.steps,
        seed=args.seed,
        initial_fair_value=args.initial_fair_value,
        fair_value_step_vol=args.fair_value_step_vol,
        fair_value_step_vol_fraction=args.fair_value_step_vol_fraction,
        base_spread=args.base_spread,
        base_spread_fraction=args.base_spread_fraction,
        inventory_skew=args.inventory_skew,
        inventory_skew_fraction=args.inventory_skew_fraction,
        max_abs_inventory=args.max_abs_inventory,
        min_order_size=args.min_order_size,
        max_order_size=args.max_order_size,
        base_arrival_rate=args.base_arrival_rate,
        strict_random_null=not args.no_strict_random_null,
        random_null_mode=args.random_null_mode,
        world_weights=world_weights,
    )

    config_output = args.config_output
    if config_output is None:
        base_output = args.monte_carlo_output if args.monte_carlo_runs > 0 else args.output
        config_output = base_output.with_suffix(".config.json")
    if args.monte_carlo_runs > 0:
        write_config_json(
            config,
            config_output,
            experiment_metadata={
                "mode": "monte_carlo",
                "monte_carlo_runs": args.monte_carlo_runs,
                "monte_carlo_output": str(args.monte_carlo_output),
            },
        )
        results = run_monte_carlo(config, args.monte_carlo_runs)
        write_monte_carlo_csv(results, args.monte_carlo_output)
        summary = summarize_monte_carlo(results)
        print("Monte Carlo simulation complete")
        print(f"Worlds: {dict(config.world_weights)}")
        print(f"Scenario ID: {scenario_id(config)}")
        print(f"Runs: {args.monte_carlo_runs}")
        print(f"CSV written to: {args.monte_carlo_output}")
        print(f"Scenario config written to: {config_output}")
        for key, value in summary.items():
            print(f"{key}: {value:.10g}")
        return

    dataset = candle_dataset_id(
        config, args.candle_steps, include_partial=args.include_partial_candle
    )
    write_config_json(
        config,
        config_output,
        experiment_metadata={
            "mode": "single_path",
            "candle_construction": {
                "steps_per_candle": args.candle_steps,
                "include_partial": args.include_partial_candle,
                "dataset_id": dataset,
            },
            "outputs": {
                "history_csv": str(args.output),
                "candles_csv": str(args.candles_output),
                "chart_html": str(args.chart_output),
            },
        },
    )
    history = simulate_market(config)
    write_history_csv(history, args.output)
    candles = build_candles(
        history,
        args.candle_steps,
        include_partial=args.include_partial_candle,
        require_contiguous_steps=True,
    )
    write_candles_csv(candles, args.candles_output, dataset_id=dataset)
    write_interactive_candlestick_html(
        candles,
        args.chart_output,
        visible_candles=args.visible_candles,
        title=_chart_title(config),
    )

    summary = summarize(history)
    print("Synthetic market simulation complete")
    print(f"Worlds: {dict(config.world_weights)}")
    print(f"Scenario ID: {scenario_id(config)}")
    print(f"Run ID: {run_id(config)}")
    print(f"Candle dataset ID: {dataset}")
    print(f"History CSV written to: {args.output}")
    print(f"Candle CSV written to: {args.candles_output}")
    print(f"Scenario config written to: {config_output}")
    print(f"Interactive candlestick chart written to: {args.chart_output}")
    print(f"Candles generated: {len(candles)}")
    for key, value in summary.items():
        print(f"{key}: {value:.10g}")


__all__ = [
    "SIMULATOR_VERSION", "RNG_STREAM_VERSION", "WORLD_NAMES", "RandomWorldConfig", "RuleBasedWorldConfig",
    "EmotionalWorldConfig", "InformationWorldConfig", "MeanReversionWorldConfig",
    "MomentumWorldConfig", "RegimeWorldConfig", "LiquidityWorldConfig", "AdaptiveWorldConfig",
    "SyntheticMarketConfig", "RandomMarketConfig", "MarketMaker", "WorldSignal", "MarketSnapshot",
    "MonteCarloResult", "canonical_behavior_config", "scenario_id", "run_id", "candle_dataset_id", "simulate_market", "simulate_random_market",
    "summarize", "run_monte_carlo", "summarize_monte_carlo", "write_config_json",
    "write_history_csv", "write_candles_csv", "write_monte_carlo_csv", "parse_world_specification",
]


if __name__ == "__main__":
    main()
