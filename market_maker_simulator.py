from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from chart_renderer import build_candles, write_interactive_candlestick_html


MIN_PRICE = 0.01
SIMULATOR_VERSION = "2.0.0"
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_tanh(value: float) -> float:
    if value >= 20.0:
        return 1.0
    if value <= -20.0:
        return -1.0
    return math.tanh(value)


def _sigmoid(value: float) -> float:
    if value >= 0:
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
        raise ValueError(
            f"Unknown world {name!r}. Valid worlds: {', '.join(WORLD_NAMES)}"
        )
    return normalized


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _validate_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not valid or not math.isfinite(value):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}, got {value}")


@dataclass(frozen=True)
class RandomWorldConfig:
    """Pure noise-order component used as the null/baseline world."""

    signal_strength: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive("random.signal_strength", self.signal_strength, allow_zero=True)


@dataclass(frozen=True)
class RuleBasedWorldConfig:
    """Mechanical traders driven only by past/available prices."""

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
        if self.short_ma < 2:
            raise ValueError("rule_based.short_ma must be >= 2")
        if self.long_ma <= self.short_ma:
            raise ValueError("rule_based.long_ma must be > short_ma")
        if self.breakout_lookback < 3 or self.zscore_lookback < 3:
            raise ValueError("rule-based lookbacks must be >= 3")
        _validate_positive("rule_based.zscore_scale", self.zscore_scale)
        for name, value in (
            ("ma_weight", self.ma_weight),
            ("breakout_weight", self.breakout_weight),
            ("contrarian_weight", self.contrarian_weight),
        ):
            _validate_positive(f"rule_based.{name}", value, allow_zero=True)
        if self.ma_weight + self.breakout_weight + self.contrarian_weight <= 0:
            raise ValueError("rule-based strategy weights cannot all be zero")
        _validate_positive("rule_based.activity_boost", self.activity_boost, allow_zero=True)
        if not 0.0 <= self.strict_signal_threshold <= 1.0:
            raise ValueError("rule_based.strict_signal_threshold must be between 0 and 1")


@dataclass(frozen=True)
class EmotionalWorldConfig:
    """Behavioral feedback: FOMO, fear, greed, drawdown panic and decay."""

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
        _validate_probability("emotional.decay", self.decay)
        _validate_positive("emotional.return_scale", self.return_scale)
        for name, value in (
            ("fomo_sensitivity", self.fomo_sensitivity),
            ("fear_sensitivity", self.fear_sensitivity),
            ("greed_sensitivity", self.greed_sensitivity),
            ("drawdown_sensitivity", self.drawdown_sensitivity),
            ("drawdown_scale", self.drawdown_scale),
            ("signal_scale", self.signal_scale),
            ("size_boost", self.size_boost),
        ):
            _validate_positive(f"emotional.{name}", value, allow_zero=True)


@dataclass(frozen=True)
class InformationWorldConfig:
    """Latent fundamental shocks with gradual public information diffusion."""

    event_probability: float = 0.0025
    shock_std_fraction: float = 0.018
    min_shock_fraction: float = 0.004
    public_diffusion_rate: float = 0.035
    informed_fraction: float = 0.12
    signal_gap_fraction: float = 0.010
    event_size_boost: float = 1.50
    event_decay: float = 0.93

    def __post_init__(self) -> None:
        _validate_probability("information.event_probability", self.event_probability)
        _validate_probability("information.public_diffusion_rate", self.public_diffusion_rate)
        _validate_probability("information.informed_fraction", self.informed_fraction)
        _validate_probability("information.event_decay", self.event_decay)
        for name, value in (
            ("shock_std_fraction", self.shock_std_fraction),
            ("min_shock_fraction", self.min_shock_fraction),
            ("signal_gap_fraction", self.signal_gap_fraction),
            ("event_size_boost", self.event_size_boost),
        ):
            _validate_positive(f"information.{name}", value, allow_zero=True)
        if self.signal_gap_fraction == 0:
            raise ValueError("information.signal_gap_fraction must be > 0")


@dataclass(frozen=True)
class MeanReversionWorldConfig:
    """Value traders pull market price toward latent fundamental value."""

    deviation_scale_fraction: float = 0.006
    deadband_fraction: float = 0.0005
    size_boost: float = 0.70

    def __post_init__(self) -> None:
        _validate_positive("mean_reversion.deviation_scale_fraction", self.deviation_scale_fraction)
        _validate_positive("mean_reversion.deadband_fraction", self.deadband_fraction, allow_zero=True)
        _validate_positive("mean_reversion.size_boost", self.size_boost, allow_zero=True)


@dataclass(frozen=True)
class MomentumWorldConfig:
    """Trend followers extrapolate only already-observed returns."""

    lookback: int = 12
    return_scale: float = 0.006
    deadband: float = 0.0002
    size_boost: float = 0.85

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("momentum.lookback must be >= 2")
        _validate_positive("momentum.return_scale", self.return_scale)
        _validate_positive("momentum.deadband", self.deadband, allow_zero=True)
        _validate_positive("momentum.size_boost", self.size_boost, allow_zero=True)


@dataclass(frozen=True)
class RegimeWorldConfig:
    """Persistent hidden market regimes with causal Markov switching."""

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
        _validate_probability("regime.stay_probability", self.stay_probability)
        for name, value in (
            ("trend_drift_fraction_per_step", self.trend_drift_fraction_per_step),
            ("panic_drift_fraction_per_step", self.panic_drift_fraction_per_step),
            ("high_vol_multiplier", self.high_vol_multiplier),
            ("panic_vol_multiplier", self.panic_vol_multiplier),
            ("euphoria_vol_multiplier", self.euphoria_vol_multiplier),
            ("panic_liquidity_multiplier", self.panic_liquidity_multiplier),
            ("high_vol_liquidity_multiplier", self.high_vol_liquidity_multiplier),
            ("regime_signal_strength", self.regime_signal_strength),
        ):
            _validate_positive(f"regime.{name}", value, allow_zero=True)


@dataclass(frozen=True)
class LiquidityWorldConfig:
    """Stochastic liquidity, round-number clustering and stop cascades."""

    persistence: float = 0.965
    log_liquidity_vol: float = 0.08
    min_liquidity: float = 0.30
    max_liquidity: float = 3.00
    round_number_interval: float = 0.0  # 0 = automatic scale-aware round-number spacing
    round_proximity_fraction: float = 0.0018
    support_resistance_strength: float = 0.45
    stop_cascade_strength: float = 0.90
    stop_cascade_size_boost: float = 1.80

    def __post_init__(self) -> None:
        _validate_probability("liquidity.persistence", self.persistence)
        for name, value in (
            ("log_liquidity_vol", self.log_liquidity_vol),
            ("min_liquidity", self.min_liquidity),
            ("max_liquidity", self.max_liquidity),
            ("round_number_interval", self.round_number_interval),
            ("round_proximity_fraction", self.round_proximity_fraction),
            ("support_resistance_strength", self.support_resistance_strength),
            ("stop_cascade_strength", self.stop_cascade_strength),
            ("stop_cascade_size_boost", self.stop_cascade_size_boost),
        ):
            _validate_positive(f"liquidity.{name}", value, allow_zero=True)
        if self.min_liquidity <= 0 or self.max_liquidity < self.min_liquidity:
            raise ValueError("liquidity bounds are invalid")
        if self.round_number_interval < 0:
            raise ValueError("round_number_interval must be >= 0 (0 selects automatic spacing)")


@dataclass(frozen=True)
class AdaptiveWorldConfig:
    """Evolutionary agents reweight strategies according to past realised success."""

    score_decay: float = 0.965
    learning_rate: float = 0.16
    return_scale: float = 0.0015
    temperature: float = 0.35
    exploration_weight: float = 0.05
    size_boost: float = 0.70

    def __post_init__(self) -> None:
        _validate_probability("adaptive.score_decay", self.score_decay)
        _validate_positive("adaptive.learning_rate", self.learning_rate, allow_zero=True)
        _validate_positive("adaptive.return_scale", self.return_scale)
        _validate_positive("adaptive.temperature", self.temperature)
        _validate_probability("adaptive.exploration_weight", self.exploration_weight)
        _validate_positive("adaptive.size_boost", self.size_boost, allow_zero=True)


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Configuration for one or any mixture of the nine synthetic market worlds.

    `world_weights` can contain one world, any subset, or all nine worlds. Weights
    are relative rather than probabilities, so e.g. {"emotional": 2, "momentum": 1}
    gives the emotional component twice the influence of momentum.
    """

    steps: int = 10_000
    seed: int = 42
    initial_fair_value: float = 100.0
    fair_value_step_vol: float = 0.03
    base_fundamental_drift: float = 0.0

    base_spread: float = 0.04
    min_tick: float = 0.01
    inventory_skew: float = 0.002
    inventory_skew_cap_fraction: float = 0.025
    inventory_spread_sensitivity: float = 0.00020

    min_order_size: int = 1
    max_order_size: int = 10
    max_order_size_multiplier: float = 5.0
    decision_signal_strength: float = 2.1
    decision_noise: float = 0.75

    fundamental_anchor_strength: float = 0.025
    order_impact_fraction: float = 0.00032
    microstructure_noise_fraction: float = 0.000025
    max_single_step_mid_move_fraction: float = 0.035

    strict_random_null: bool = True
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
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        _validate_positive("initial_fair_value", self.initial_fair_value)
        _validate_positive("fair_value_step_vol", self.fair_value_step_vol, allow_zero=True)
        _validate_positive("base_spread", self.base_spread)
        _validate_positive("min_tick", self.min_tick)
        _validate_positive("inventory_skew", self.inventory_skew, allow_zero=True)
        _validate_positive("inventory_skew_cap_fraction", self.inventory_skew_cap_fraction, allow_zero=True)
        _validate_positive("inventory_spread_sensitivity", self.inventory_spread_sensitivity, allow_zero=True)
        if self.min_order_size <= 0 or self.max_order_size < self.min_order_size:
            raise ValueError("order-size bounds are invalid")
        _validate_positive("max_order_size_multiplier", self.max_order_size_multiplier)
        _validate_positive("decision_signal_strength", self.decision_signal_strength, allow_zero=True)
        _validate_positive("decision_noise", self.decision_noise, allow_zero=True)
        _validate_probability("fundamental_anchor_strength", self.fundamental_anchor_strength)
        _validate_positive("order_impact_fraction", self.order_impact_fraction, allow_zero=True)
        _validate_positive("microstructure_noise_fraction", self.microstructure_noise_fraction, allow_zero=True)
        _validate_positive("max_single_step_mid_move_fraction", self.max_single_step_mid_move_fraction)

        cleaned: dict[str, float] = {}
        for raw_name, raw_weight in self.world_weights.items():
            name = _normalize_world_name(str(raw_name))
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"world weight for {name!r} must be finite and >= 0")
            if weight > 0:
                cleaned[name] = cleaned.get(name, 0.0) + weight
        if not cleaned:
            raise ValueError("At least one world must have a positive weight")
        object.__setattr__(self, "world_weights", cleaned)

    @property
    def active_worlds(self) -> tuple[str, ...]:
        return tuple(name for name in WORLD_NAMES if self.world_weights.get(name, 0.0) > 0)

    @property
    def is_strict_random_null(self) -> bool:
        return self.strict_random_null and self.active_worlds == ("random",)


@dataclass(frozen=True)
class RandomMarketConfig:
    """Backward-compatible configuration for the original random-only API."""

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
        )


@dataclass
class MarketMaker:
    base_spread: float
    inventory_skew: float
    min_tick: float = 0.01
    inventory_skew_cap_fraction: float = 0.025
    inventory_spread_sensitivity: float = 0.00020
    cash: float = 0.0
    inventory: int = 0

    def quote(
        self,
        reference_price: float,
        *,
        spread_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        reference_price = max(MIN_PRICE, float(reference_price))
        spread_multiplier = max(0.05, float(spread_multiplier))

        raw_skew = self.inventory_skew * self.inventory
        skew_cap = self.inventory_skew_cap_fraction * reference_price
        bounded_skew = _clamp(raw_skew, -skew_cap, skew_cap)
        reservation_price = max(MIN_PRICE, reference_price - bounded_skew)

        inventory_spread = 1.0 + self.inventory_spread_sensitivity * abs(self.inventory)
        effective_spread = max(
            self.min_tick,
            self.base_spread * spread_multiplier * inventory_spread,
        )
        half_spread = effective_spread / 2.0
        bid = max(MIN_PRICE, reservation_price - half_spread)
        ask = max(bid + self.min_tick, reservation_price + half_spread)
        return bid, ask

    def sell_to_taker(self, price: float, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.cash += price * size
        self.inventory -= size

    def buy_from_taker(self, price: float, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.cash -= price * size
        self.inventory += size

    def mark_to_market_pnl(self, mark_price: float) -> float:
        return self.cash + self.inventory * mark_price


@dataclass(frozen=True)
class WorldSignal:
    signal: float = 0.0
    size_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    impact_multiplier: float = 1.0
    metadata: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.signal):
            raise ValueError("WorldSignal.signal must be finite")
        object.__setattr__(self, "signal", _clamp(float(self.signal), -1.0, 1.0))
        for attr in ("size_multiplier", "spread_multiplier", "impact_multiplier"):
            value = float(getattr(self, attr))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"WorldSignal.{attr} must be finite and > 0")


@dataclass(frozen=True)
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
    aggregate_signal: float = 0.0
    buy_probability: float = 0.5
    signed_order_flow: int = 0
    liquidity: float = 1.0
    regime: str = "none"
    sentiment: float = 0.0
    fomo: float = 0.0
    fear: float = 0.0
    greed: float = 0.0
    public_fundamental: float = 0.0
    information_gap: float = 0.0
    information_event: bool = False
    active_worlds: str = "random"
    simulation_seed: int = 0
    world_weights_json: str = "{}"
    world_signals_json: str = "{}"

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class SimulationState:
    fundamental_value: float
    reference_price: float
    public_fundamental: float
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


@dataclass(frozen=True)
class RandomStreams:
    """Independent deterministic RNG streams for controlled experiments.

    Adding or removing a behavioral world must not silently change the base
    fundamental random walk merely because that world consumes extra random
    numbers. Fixed sub-streams preserve common random numbers across worlds.
    """

    fundamental: random.Random
    decision: random.Random
    order_size: random.Random
    random_world: random.Random
    information: random.Random
    regime: random.Random
    liquidity: random.Random
    adaptive: random.Random
    microstructure: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStreams":
        offsets = (
            11_003, 23_017, 37_033, 41_041, 53_051,
            67_067, 79_081, 83_089, 97_097,
        )
        streams = [random.Random(seed * 1_000_003 + offset) for offset in offsets]
        return cls(*streams)


@dataclass(frozen=True)
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
    if start <= 0 or end <= 0:
        return 0.0
    return math.log(end / start)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return pstdev(values)


def _moving_average_signal(prices: Sequence[float], short: int, long: int) -> float:
    if len(prices) < long:
        return 0.0
    short_avg = mean(prices[-short:])
    long_avg = mean(prices[-long:])
    returns = [
        math.log(prices[i] / prices[i - 1])
        for i in range(max(1, len(prices) - long + 1), len(prices))
        if prices[i - 1] > 0 and prices[i] > 0
    ]
    scale = max(_sample_std(returns) * math.sqrt(max(1, short)), 1e-5)
    relative_gap = (short_avg - long_avg) / max(long_avg, MIN_PRICE)
    return _safe_tanh(relative_gap / scale)


def _breakout_signal(prices: Sequence[float], lookback: int) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    current = prices[-1]
    previous_window = prices[-lookback - 1 : -1]
    high = max(previous_window)
    low = min(previous_window)
    width = max(high - low, current * 1e-5)
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
    if std <= 1e-12:
        return 0.0
    z = (window[-1] - center) / std
    return -_safe_tanh(z / zscore_scale)


def _fundamental_reversion_signal(
    reference_price: float,
    fundamental_value: float,
    cfg: MeanReversionWorldConfig,
) -> float:
    gap_fraction = (fundamental_value - reference_price) / max(reference_price, MIN_PRICE)
    if abs(gap_fraction) <= cfg.deadband_fraction:
        return 0.0
    effective_gap = gap_fraction - math.copysign(cfg.deadband_fraction, gap_fraction)
    return _safe_tanh(effective_gap / cfg.deviation_scale_fraction)


def _momentum_signal(prices: Sequence[float], cfg: MomentumWorldConfig) -> float:
    trailing_return = _recent_return(prices, cfg.lookback)
    if abs(trailing_return) <= cfg.deadband:
        return 0.0
    effective = trailing_return - math.copysign(cfg.deadband, trailing_return)
    return _safe_tanh(effective / cfg.return_scale)


def _effective_round_interval(price: float, configured_interval: float) -> float:
    if configured_interval > 0:
        return configured_interval
    price = max(price, MIN_PRICE)
    magnitude = 10.0 ** math.floor(math.log10(price))
    return max(MIN_PRICE, magnitude * 0.05)


def _crossed_round_level(previous: float, current: float, interval: float) -> bool:
    if previous <= 0 or current <= 0 or previous == current:
        return False
    low, high = sorted((previous, current))
    first_level = math.ceil(low / interval) * interval
    return first_level <= high and not math.isclose(first_level, low, rel_tol=0.0, abs_tol=1e-12)


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    scaled = {name: score / temperature for name, score in scores.items()}
    maximum = max(scaled.values())
    exps = {name: math.exp(_clamp(value - maximum, -700.0, 700.0)) for name, value in scaled.items()}
    total = sum(exps.values())
    if total <= 0:
        equal = 1.0 / len(exps)
        return {name: equal for name in exps}
    return {name: value / total for name, value in exps.items()}


def _regime_profile(regime: str, cfg: RegimeWorldConfig) -> RegimeProfile:
    if regime == "trend_up":
        return RegimeProfile(
            drift_fraction=cfg.trend_drift_fraction_per_step,
            volatility_multiplier=1.15,
            signal=cfg.regime_signal_strength,
        )
    if regime == "trend_down":
        return RegimeProfile(
            drift_fraction=-cfg.trend_drift_fraction_per_step,
            volatility_multiplier=1.15,
            signal=-cfg.regime_signal_strength,
        )
    if regime == "mean_reverting":
        return RegimeProfile(volatility_multiplier=0.85, signal=0.0)
    if regime == "high_volatility":
        return RegimeProfile(
            volatility_multiplier=cfg.high_vol_multiplier,
            liquidity_multiplier=cfg.high_vol_liquidity_multiplier,
            signal=0.0,
        )
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
        # Fear is driven by a recent drawdown, not an all-time peak that would make
        # a single old crash permanently poison the emotional state.
        state.rolling_peak_price = max(state.recent_trade_prices[-128:])
    else:
        state.rolling_peak_price = last_price
    drawdown = max(0.0, 1.0 - last_price / max(state.rolling_peak_price, MIN_PRICE))

    trend = max(_recent_return(state.recent_trade_prices, 8), 0.0) / max(cfg.return_scale * 4.0, 1e-12)

    state.fomo = _clamp(
        cfg.decay * state.fomo + cfg.fomo_sensitivity * scaled_up,
        0.0,
        8.0,
    )
    state.fear = _clamp(
        cfg.decay * state.fear
        + cfg.fear_sensitivity * scaled_down
        + cfg.drawdown_sensitivity * drawdown / cfg.drawdown_scale,
        0.0,
        8.0,
    )
    state.greed = _clamp(
        cfg.decay * state.greed + cfg.greed_sensitivity * trend,
        0.0,
        8.0,
    )
    raw_sentiment = state.fomo + 0.65 * state.greed - state.fear
    state.sentiment = _safe_tanh(raw_sentiment / max(cfg.signal_scale, 1e-12))


def _update_information_environment(
    state: SimulationState,
    rng: random.Random,
    cfg: InformationWorldConfig,
) -> float:
    """Apply a current-time information shock and return the fundamental shock amount."""
    state.information_event = False
    shock = 0.0
    if rng.random() < cfg.event_probability:
        raw_fraction = rng.gauss(0.0, cfg.shock_std_fraction)
        if abs(raw_fraction) < cfg.min_shock_fraction:
            direction = -1.0 if raw_fraction < 0 else 1.0
            if raw_fraction == 0:
                direction = rng.choice((-1.0, 1.0))
            raw_fraction = direction * cfg.min_shock_fraction
        shock = state.fundamental_value * raw_fraction
        state.fundamental_value = max(MIN_PRICE, state.fundamental_value + shock)
        state.event_intensity = max(
            state.event_intensity,
            min(4.0, abs(raw_fraction) / max(cfg.shock_std_fraction, 1e-12)),
        )
        state.information_event = True
    else:
        state.event_intensity *= cfg.event_decay

    state.public_fundamental += cfg.public_diffusion_rate * (
        state.fundamental_value - state.public_fundamental
    )
    state.public_fundamental = max(MIN_PRICE, state.public_fundamental)
    return shock


def _update_liquidity(
    state: SimulationState,
    rng: random.Random,
    cfg: LiquidityWorldConfig,
    regime_profile: RegimeProfile,
) -> None:
    log_liquidity = math.log(max(state.liquidity, 1e-9))
    innovation = rng.gauss(0.0, cfg.log_liquidity_vol)
    log_liquidity = cfg.persistence * log_liquidity + innovation
    raw = math.exp(log_liquidity) * regime_profile.liquidity_multiplier
    state.liquidity = _clamp(raw, cfg.min_liquidity, cfg.max_liquidity)


def _random_world_signal(rng: random.Random, cfg: RandomWorldConfig) -> WorldSignal:
    return WorldSignal(signal=rng.choice((-1.0, 1.0)) * cfg.signal_strength)


def _rule_based_world_signal(state: SimulationState, cfg: RuleBasedWorldConfig) -> WorldSignal:
    prices = state.recent_trade_prices
    ma_signal = _moving_average_signal(prices, cfg.short_ma, cfg.long_ma)
    breakout = _breakout_signal(prices, cfg.breakout_lookback)
    contrarian = _zscore_contrarian_signal(prices, cfg.zscore_lookback, cfg.zscore_scale)
    total_weight = cfg.ma_weight + cfg.breakout_weight + cfg.contrarian_weight
    raw = (
        cfg.ma_weight * ma_signal
        + cfg.breakout_weight * breakout
        + cfg.contrarian_weight * contrarian
    ) / total_weight
    intensity = abs(raw)
    return WorldSignal(
        signal=raw,
        size_multiplier=1.0 + cfg.activity_boost * intensity,
        metadata={
            "ma": ma_signal,
            "breakout": breakout,
            "contrarian": contrarian,
        },
    )


def _emotional_world_signal(state: SimulationState, cfg: EmotionalWorldConfig) -> WorldSignal:
    intensity = _clamp((state.fomo + state.fear + state.greed) / 8.0, 0.0, 1.0)
    return WorldSignal(
        signal=state.sentiment,
        size_multiplier=1.0 + cfg.size_boost * intensity,
        metadata={"fomo": state.fomo, "fear": state.fear, "greed": state.greed},
    )


def _information_world_signal(state: SimulationState, cfg: InformationWorldConfig) -> WorldSignal:
    visible_target = (
        cfg.informed_fraction * state.fundamental_value
        + (1.0 - cfg.informed_fraction) * state.public_fundamental
    )
    gap_fraction = (visible_target - state.reference_price) / max(state.reference_price, MIN_PRICE)
    signal = _safe_tanh(gap_fraction / cfg.signal_gap_fraction)
    size_multiplier = 1.0 + cfg.event_size_boost * min(1.0, state.event_intensity)
    return WorldSignal(
        signal=signal,
        size_multiplier=size_multiplier,
        metadata={
            "visible_target": visible_target,
            "public_fundamental": state.public_fundamental,
            "event_intensity": state.event_intensity,
        },
    )


def _mean_reversion_world_signal(state: SimulationState, cfg: MeanReversionWorldConfig) -> WorldSignal:
    signal = _fundamental_reversion_signal(state.reference_price, state.fundamental_value, cfg)
    return WorldSignal(
        signal=signal,
        size_multiplier=1.0 + cfg.size_boost * abs(signal),
        metadata={
            "valuation_gap_fraction": (
                (state.fundamental_value - state.reference_price)
                / max(state.reference_price, MIN_PRICE)
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
            state.fundamental_value,
            mean_reversion_cfg,
        ) * cfg.regime_signal_strength
    size_multiplier = 1.0 + 0.60 * abs(signal)
    return WorldSignal(
        signal=signal,
        size_multiplier=size_multiplier,
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
        recent_direction = _sign(math.log(current / previous)) if previous > 0 and current > 0 else 0.0
        if _crossed_round_level(previous, current, interval):
            signal = recent_direction * cfg.stop_cascade_strength
            size_boost = cfg.stop_cascade_size_boost
            mode = "stop_cascade"
        else:
            nearest_level = round(current / interval) * interval
            proximity = abs(current - nearest_level) / max(current, MIN_PRICE)
            if proximity <= cfg.round_proximity_fraction and recent_direction != 0.0:
                # Approaching from below behaves like resistance; approaching from above like support.
                if current <= nearest_level and recent_direction > 0:
                    signal = -cfg.support_resistance_strength
                    mode = "round_resistance"
                elif current >= nearest_level and recent_direction < 0:
                    signal = cfg.support_resistance_strength
                    mode = "round_support"

    liquidity = max(state.liquidity, 1e-9)
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
            state.fundamental_value,
            config.mean_reversion,
        ),
        "breakout": _breakout_signal(state.recent_trade_prices, config.rule_based.breakout_lookback),
        "noise": rng.choice((-1.0, 1.0)),
    }


def _update_adaptive_scores(state: SimulationState, cfg: AdaptiveWorldConfig) -> None:
    last_return = state.recent_returns[-1] if state.recent_returns else 0.0
    scaled_return = _clamp(last_return / cfg.return_scale, -5.0, 5.0)
    for name in state.adaptive_scores:
        previous_signal = state.adaptive_previous_signals.get(name, 0.0)
        reward = previous_signal * scaled_return
        state.adaptive_scores[name] = (
            cfg.score_decay * state.adaptive_scores[name]
            + cfg.learning_rate * reward
        )


def _adaptive_world_signal(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
) -> WorldSignal:
    _update_adaptive_scores(state, config.adaptive)
    signals = _adaptive_strategy_signals(state, rng, config)
    weights = _softmax(state.adaptive_scores, config.adaptive.temperature)

    n = len(weights)
    exploration = config.adaptive.exploration_weight
    if n > 0 and exploration > 0:
        weights = {
            name: (1.0 - exploration) * weight + exploration / n
            for name, weight in weights.items()
        }

    combined = sum(weights[name] * signals[name] for name in weights)
    state.adaptive_previous_signals = signals
    metadata: dict[str, float | str] = {}
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
    random_world_rng: random.Random,
    adaptive_rng: random.Random,
    config: SyntheticMarketConfig,
    regime_profile: RegimeProfile,
) -> dict[str, WorldSignal]:
    signals: dict[str, WorldSignal] = {}
    for world in config.active_worlds:
        if world == "random":
            signals[world] = _random_world_signal(random_world_rng, config.random_world)
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
            signals[world] = _regime_world_signal(
                state, regime_profile, config.regime, config.mean_reversion
            )
        elif world == "liquidity":
            signals[world] = _liquidity_world_signal(state, config.liquidity)
        elif world == "adaptive":
            signals[world] = _adaptive_world_signal(state, adaptive_rng, config)
        else:  # pragma: no cover - protected by config validation
            raise RuntimeError(f"Unhandled world: {world}")
    return signals


def _aggregate_world_signals(
    world_signals: Mapping[str, WorldSignal],
    world_weights: Mapping[str, float],
) -> tuple[float, float, float, float]:
    total_weight = sum(world_weights[name] for name in world_signals)
    if total_weight <= 0:
        return 0.0, 1.0, 1.0, 1.0

    aggregate_signal = sum(
        world_weights[name] * world_signals[name].signal for name in world_signals
    ) / total_weight

    # Geometric averaging prevents one component with a large multiplier from
    # exploding the combined scale while preserving multiplicative meaning.
    def geometric(attribute: str) -> float:
        weighted_logs = 0.0
        for name, signal in world_signals.items():
            value = max(float(getattr(signal, attribute)), 1e-9)
            weighted_logs += world_weights[name] * math.log(value)
        return math.exp(weighted_logs / total_weight)

    return (
        _clamp(aggregate_signal, -1.0, 1.0),
        geometric("size_multiplier"),
        geometric("spread_multiplier"),
        geometric("impact_multiplier"),
    )


def _world_signals_json(signals: Mapping[str, WorldSignal]) -> str:
    compact: dict[str, Any] = {}
    for name, signal in signals.items():
        compact[name] = {
            "signal": round(signal.signal, 8),
            "size_multiplier": round(signal.size_multiplier, 8),
            **{
                key: round(value, 8) if isinstance(value, float) else value
                for key, value in signal.metadata.items()
            },
        }
    return json.dumps(compact, separators=(",", ":"), sort_keys=True)


def _update_fundamental(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
    profile: RegimeProfile,
) -> None:
    drift = config.base_fundamental_drift
    if "regime" in config.active_worlds:
        drift += state.fundamental_value * profile.drift_fraction
    volatility = config.fair_value_step_vol * (
        profile.volatility_multiplier if "regime" in config.active_worlds else 1.0
    )
    state.fundamental_value = max(
        MIN_PRICE,
        state.fundamental_value + drift + rng.gauss(0.0, volatility),
    )


def _update_reference_before_quote(
    state: SimulationState,
    rng: random.Random,
    config: SyntheticMarketConfig,
) -> None:
    if config.is_strict_random_null:
        # Preserve a clean null: no persistent order-flow feedback. The original
        # simulator quotes around the current random-walk fair value each step.
        state.reference_price = state.fundamental_value
        return

    anchor = config.fundamental_anchor_strength
    if "regime" in config.active_worlds and state.regime == "mean_reverting":
        anchor = min(1.0, anchor * 3.0)

    state.reference_price += anchor * (
        state.fundamental_value - state.reference_price
    )
    if config.microstructure_noise_fraction > 0:
        state.reference_price *= math.exp(
            rng.gauss(0.0, config.microstructure_noise_fraction)
        )
    state.reference_price = max(MIN_PRICE, state.reference_price)


def _apply_order_impact(
    reference_price: float,
    signed_order_flow: int,
    average_base_order_size: float,
    impact_multiplier: float,
    config: SyntheticMarketConfig,
) -> float:
    if config.is_strict_random_null or config.order_impact_fraction <= 0:
        return reference_price

    normalized_flow = signed_order_flow / max(average_base_order_size, 1.0)
    log_move = config.order_impact_fraction * normalized_flow * impact_multiplier
    log_move = _clamp(
        log_move,
        -config.max_single_step_mid_move_fraction,
        config.max_single_step_mid_move_fraction,
    )
    return max(MIN_PRICE, reference_price * math.exp(log_move))


def _record_return(state: SimulationState, trade_price: float) -> None:
    if state.recent_trade_prices and state.recent_trade_prices[-1] > 0 and trade_price > 0:
        state.recent_returns.append(math.log(trade_price / state.recent_trade_prices[-1]))
    state.recent_trade_prices.append(trade_price)

    # Keep a bounded causal history: enough for all configured lookbacks plus margin.
    max_history = 512
    if len(state.recent_trade_prices) > max_history:
        del state.recent_trade_prices[:-max_history]
    if len(state.recent_returns) > max_history:
        del state.recent_returns[:-max_history]


def simulate_market(config: SyntheticMarketConfig) -> list[MarketSnapshot]:
    """Simulate one causal synthetic market path.

    All trading decisions use only current state and previously observed prices. No
    future price, future regime, or future event is consulted. The hidden state is
    recorded for research diagnostics but is not required by the chart/pattern layer.
    """

    streams = RandomStreams.from_seed(config.seed)
    maker = MarketMaker(
        base_spread=config.base_spread,
        inventory_skew=config.inventory_skew,
        min_tick=config.min_tick,
        inventory_skew_cap_fraction=config.inventory_skew_cap_fraction,
        inventory_spread_sensitivity=config.inventory_spread_sensitivity,
    )
    state = SimulationState(
        fundamental_value=config.initial_fair_value,
        reference_price=config.initial_fair_value,
        public_fundamental=config.initial_fair_value,
        rolling_peak_price=config.initial_fair_value,
    )
    history: list[MarketSnapshot] = []
    average_base_order_size = (config.min_order_size + config.max_order_size) / 2.0
    active_worlds_label = "+".join(config.active_worlds)
    world_weights_json = json.dumps(dict(config.world_weights), separators=(",", ":"), sort_keys=True)

    for step in range(1, config.steps + 1):
        # Update persistent states using only information available before this trade.
        if "emotional" in config.active_worlds:
            _update_emotions(state, config.emotional)

        if "regime" in config.active_worlds:
            regime_profile = _update_regime(state, streams.regime, config.regime)
        else:
            state.regime = "none"
            regime_profile = RegimeProfile()

        _update_fundamental(state, streams.fundamental, config, regime_profile)

        if "information" in config.active_worlds:
            _update_information_environment(state, streams.information, config.information)
        else:
            state.information_event = False
            state.event_intensity = 0.0
            state.public_fundamental = state.fundamental_value

        if "liquidity" in config.active_worlds:
            _update_liquidity(state, streams.liquidity, config.liquidity, regime_profile)
        else:
            # Regime still affects market depth even when the explicit liquidity
            # world is off; otherwise a panic regime would not be meaningfully panicky.
            state.liquidity = _clamp(regime_profile.liquidity_multiplier, 0.25, 4.0)

        _update_reference_before_quote(state, streams.microstructure, config)

        world_signals = _collect_world_signals(
            state,
            streams.random_world,
            streams.adaptive,
            config,
            regime_profile,
        )
        aggregate_signal, size_multiplier, spread_multiplier, impact_multiplier = (
            _aggregate_world_signals(world_signals, config.world_weights)
        )

        # Explicit liquidity enters execution separately from a world's directional
        # vote. Thin books widen spreads and amplify impact; deep books do the reverse.
        liquidity = max(state.liquidity, 1e-9)
        spread_multiplier *= 1.0 / math.sqrt(liquidity)
        impact_multiplier *= 1.0 / liquidity
        if "regime" in config.active_worlds:
            spread_multiplier *= max(1.0, math.sqrt(regime_profile.volatility_multiplier))

        if config.is_strict_random_null:
            # The random-world signal itself is an independent fair coin. Using it
            # directly keeps the recorded ground-truth signal consistent with the
            # executed side while remaining completely independent of past prices.
            taker_side = "buy" if aggregate_signal > 0 else "sell"
            buy_probability = 0.5
        elif (
            config.active_worlds == ("rule_based",)
            and config.rule_based.strict_execution
            and abs(aggregate_signal) >= config.rule_based.strict_signal_threshold
        ):
            # The mathematical world can be genuinely mechanical: once a configured
            # technical event is strong enough, the taker follows it deterministically.
            taker_side = "buy" if aggregate_signal > 0 else "sell"
            buy_probability = 1.0 if aggregate_signal > 0 else 0.0
        else:
            decision_score = (
                config.decision_signal_strength * aggregate_signal
                + streams.decision.gauss(0.0, config.decision_noise)
            )
            buy_probability = _sigmoid(decision_score)
            taker_side = "buy" if streams.decision.random() < buy_probability else "sell"

        base_size = streams.order_size.randint(config.min_order_size, config.max_order_size)
        bounded_multiplier = _clamp(size_multiplier, 0.20, config.max_order_size_multiplier)
        order_size = max(1, int(round(base_size * bounded_multiplier)))

        quote_reference_price = state.reference_price
        bid, ask = maker.quote(
            quote_reference_price,
            spread_multiplier=spread_multiplier,
        )
        if taker_side == "buy":
            trade_price = ask
            maker.sell_to_taker(trade_price, order_size)
            signed_order_flow = order_size
        else:
            trade_price = bid
            maker.buy_from_taker(trade_price, order_size)
            signed_order_flow = -order_size

        state.reference_price = _apply_order_impact(
            state.reference_price,
            signed_order_flow,
            average_base_order_size,
            impact_multiplier,
            config,
        )

        _record_return(state, trade_price)

        history.append(
            MarketSnapshot(
                step=step,
                fair_value=state.fundamental_value,
                reference_price=quote_reference_price,
                post_trade_reference_price=state.reference_price,
                bid=bid,
                ask=ask,
                taker_side=taker_side,
                order_size=order_size,
                trade_price=trade_price,
                maker_inventory=maker.inventory,
                maker_cash=maker.cash,
                maker_pnl=maker.mark_to_market_pnl(state.fundamental_value),
                aggregate_signal=aggregate_signal,
                buy_probability=buy_probability,
                signed_order_flow=signed_order_flow,
                liquidity=state.liquidity,
                regime=state.regime,
                sentiment=state.sentiment,
                fomo=state.fomo,
                fear=state.fear,
                greed=state.greed,
                public_fundamental=state.public_fundamental,
                information_gap=state.fundamental_value - state.public_fundamental,
                information_event=state.information_event,
                active_worlds=active_worlds_label,
                simulation_seed=config.seed,
                world_weights_json=world_weights_json,
                world_signals_json=_world_signals_json(world_signals),
            )
        )

    return history


def simulate_random_market(config: RandomMarketConfig) -> list[MarketSnapshot]:
    """Backward-compatible entry point matching the original simulator API."""
    return simulate_market(config.to_synthetic_config())


def summarize(history: Sequence[MarketSnapshot]) -> dict[str, float]:
    if not history:
        return {}

    trade_prices = [snapshot.trade_price for snapshot in history]
    returns = [
        math.log(trade_prices[index] / trade_prices[index - 1])
        for index in range(1, len(trade_prices))
        if trade_prices[index - 1] > 0 and trade_prices[index] > 0
    ]
    pnl_path = [snapshot.maker_pnl for snapshot in history]
    peak_pnl = pnl_path[0]
    max_drawdown = 0.0
    for pnl in pnl_path:
        peak_pnl = max(peak_pnl, pnl)
        max_drawdown = max(max_drawdown, peak_pnl - pnl)

    buy_volume = sum(s.order_size for s in history if s.taker_side == "buy")
    sell_volume = sum(s.order_size for s in history if s.taker_side == "sell")
    signed_flow = buy_volume - sell_volume

    information_events = sum(1 for s in history if s.information_event)
    return {
        "steps": float(len(history)),
        "final_fair_value": history[-1].fair_value,
        "final_trade_price": history[-1].trade_price,
        "final_reference_price": history[-1].post_trade_reference_price,
        "final_quote_reference_price": history[-1].reference_price,
        "final_maker_inventory": float(history[-1].maker_inventory),
        "final_maker_pnl": history[-1].maker_pnl,
        "average_spread": mean(s.spread for s in history),
        "average_abs_inventory": mean(abs(s.maker_inventory) for s in history),
        "realized_volatility_per_step": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_log_return_per_step": mean(returns) if returns else 0.0,
        "max_pnl_drawdown": max_drawdown,
        "buy_volume": float(buy_volume),
        "sell_volume": float(sell_volume),
        "signed_order_flow": float(signed_flow),
        "total_volume": float(buy_volume + sell_volume),
        "average_liquidity": mean(s.liquidity for s in history),
        "average_abs_signal": mean(abs(s.aggregate_signal) for s in history),
        "average_buy_probability": mean(s.buy_probability for s in history),
        "information_events": float(information_events),
    }


def write_config_json(config: SyntheticMarketConfig, output_path: Path) -> None:
    """Persist the exact scenario parameters needed to reproduce an experiment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["world_weights"] = dict(config.world_weights)
    payload["active_worlds"] = list(config.active_worlds)
    payload["simulator_version"] = SIMULATOR_VERSION
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_history_csv(history: Sequence[MarketSnapshot], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "fair_value",
        "reference_price",
        "post_trade_reference_price",
        "bid",
        "ask",
        "spread",
        "taker_side",
        "order_size",
        "signed_order_flow",
        "trade_price",
        "maker_inventory",
        "maker_cash",
        "maker_pnl",
        "aggregate_signal",
        "buy_probability",
        "liquidity",
        "regime",
        "sentiment",
        "fomo",
        "fear",
        "greed",
        "public_fundamental",
        "information_gap",
        "information_event",
        "active_worlds",
        "simulation_seed",
        "world_weights_json",
        "world_signals_json",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in history:
            row = {
                "step": snapshot.step,
                "fair_value": f"{snapshot.fair_value:.8f}",
                "reference_price": f"{snapshot.reference_price:.8f}",
                "post_trade_reference_price": f"{snapshot.post_trade_reference_price:.8f}",
                "bid": f"{snapshot.bid:.8f}",
                "ask": f"{snapshot.ask:.8f}",
                "spread": f"{snapshot.spread:.8f}",
                "taker_side": snapshot.taker_side,
                "order_size": snapshot.order_size,
                "signed_order_flow": snapshot.signed_order_flow,
                "trade_price": f"{snapshot.trade_price:.8f}",
                "maker_inventory": snapshot.maker_inventory,
                "maker_cash": f"{snapshot.maker_cash:.8f}",
                "maker_pnl": f"{snapshot.maker_pnl:.8f}",
                "aggregate_signal": f"{snapshot.aggregate_signal:.8f}",
                "buy_probability": f"{snapshot.buy_probability:.8f}",
                "liquidity": f"{snapshot.liquidity:.8f}",
                "regime": snapshot.regime,
                "sentiment": f"{snapshot.sentiment:.8f}",
                "fomo": f"{snapshot.fomo:.8f}",
                "fear": f"{snapshot.fear:.8f}",
                "greed": f"{snapshot.greed:.8f}",
                "public_fundamental": f"{snapshot.public_fundamental:.8f}",
                "information_gap": f"{snapshot.information_gap:.8f}",
                "information_event": int(snapshot.information_event),
                "active_worlds": snapshot.active_worlds,
                "simulation_seed": snapshot.simulation_seed,
                "world_weights_json": snapshot.world_weights_json,
                "world_signals_json": snapshot.world_signals_json,
            }
            writer.writerow(row)


@dataclass(frozen=True)
class MonteCarloResult:
    run: int
    seed: int
    summary: Mapping[str, float]


def run_monte_carlo(
    config: SyntheticMarketConfig,
    runs: int,
    *,
    seed_start: int | None = None,
) -> list[MonteCarloResult]:
    if runs <= 0:
        raise ValueError("runs must be > 0")
    first_seed = config.seed if seed_start is None else seed_start
    results: list[MonteCarloResult] = []
    for run_index in range(runs):
        seed = first_seed + run_index
        run_config = replace(config, seed=seed)
        history = simulate_market(run_config)
        results.append(
            MonteCarloResult(
                run=run_index + 1,
                seed=seed,
                summary=summarize(history),
            )
        )
    return results


def summarize_monte_carlo(results: Sequence[MonteCarloResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = sorted(set.intersection(*(set(result.summary.keys()) for result in results)))
    output: dict[str, float] = {"runs": float(len(results))}
    for key in keys:
        values = [float(result.summary[key]) for result in results]
        output[f"mean_{key}"] = mean(values)
        output[f"sd_{key}"] = pstdev(values) if len(values) > 1 else 0.0
    return output


def write_monte_carlo_csv(results: Sequence[MonteCarloResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        output_path.write_text("run,seed\n", encoding="utf-8")
        return
    keys = sorted(set.intersection(*(set(result.summary.keys()) for result in results)))
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["run", "seed", *keys])
        writer.writeheader()
        for result in results:
            writer.writerow({"run": result.run, "seed": result.seed, **result.summary})


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

    for spec in weight_specs or ():
        if "=" not in spec:
            raise ValueError(f"Invalid --world-weight {spec!r}; expected WORLD=WEIGHT")
        raw_name, raw_weight = spec.split("=", 1)
        name = _normalize_world_name(raw_name)
        if name not in weights:
            raise ValueError(
                f"Weight specified for inactive world {name!r}; add it to --worlds first"
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid weight in {spec!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"World weights must be finite and > 0, got {weight}")
        weights[name] = weight
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the modular synthetic market-maker simulator with any mixture of "
            "nine market worlds."
        )
    )
    parser.add_argument(
        "--worlds",
        default="random",
        help=(
            "Comma-separated worlds. Valid: "
            + ", ".join(WORLD_NAMES)
            + ". Alias 'mathematical' maps to rule_based."
        ),
    )
    parser.add_argument(
        "--world-weight",
        action="append",
        default=[],
        metavar="WORLD=WEIGHT",
        help="Relative weight for an active world; can be repeated.",
    )
    parser.add_argument("--list-worlds", action="store_true")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-fair-value", type=float, default=100.0)
    parser.add_argument("--fair-value-step-vol", type=float, default=0.03)
    parser.add_argument("--base-spread", type=float, default=0.04)
    parser.add_argument("--inventory-skew", type=float, default=0.002)
    parser.add_argument("--min-order-size", type=int, default=1)
    parser.add_argument("--max-order-size", type=int, default=10)
    parser.add_argument(
        "--no-strict-random-null",
        action="store_true",
        help="Allow persistent order-impact feedback even in a random-only run.",
    )
    parser.add_argument("--candle-steps", type=int, default=5)
    parser.add_argument("--visible-candles", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_market.csv"))
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help="Scenario metadata JSON. Defaults beside the CSV output.",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=Path("results/synthetic_market_candles.html"),
    )
    parser.add_argument(
        "--monte-carlo-runs",
        type=int,
        default=0,
        help="If >0, run this many independent seeds instead of one charted path.",
    )
    parser.add_argument(
        "--monte-carlo-output",
        type=Path,
        default=Path("results/monte_carlo_summary.csv"),
    )
    return parser.parse_args()


def _chart_title(config: SyntheticMarketConfig) -> str:
    world_text = " + ".join(name.replace("_", " ").title() for name in config.active_worlds)
    return f"Synthetic Market — {world_text}"


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
        base_spread=args.base_spread,
        inventory_skew=args.inventory_skew,
        min_order_size=args.min_order_size,
        max_order_size=args.max_order_size,
        strict_random_null=not args.no_strict_random_null,
        world_weights=world_weights,
    )

    config_output = args.config_output
    if config_output is None:
        base_output = args.monte_carlo_output if args.monte_carlo_runs > 0 else args.output
        config_output = base_output.with_suffix(".config.json")
    write_config_json(config, config_output)

    if args.monte_carlo_runs > 0:
        results = run_monte_carlo(config, args.monte_carlo_runs)
        write_monte_carlo_csv(results, args.monte_carlo_output)
        summary = summarize_monte_carlo(results)
        print("Monte Carlo simulation complete")
        print(f"Worlds: {config.world_weights}")
        print(f"Runs: {args.monte_carlo_runs}")
        print(f"CSV written to: {args.monte_carlo_output}")
        print(f"Scenario config written to: {config_output}")
        for key, value in summary.items():
            print(f"{key}: {value:.8f}")
        return

    history = simulate_market(config)
    write_history_csv(history, args.output)
    candles = build_candles(history, args.candle_steps)
    # Updated chart_renderer accepts an optional title; fall back cleanly if an
    # older renderer is present so this simulator remains backward compatible.
    try:
        write_interactive_candlestick_html(
            candles,
            args.chart_output,
            visible_candles=args.visible_candles,
            title=_chart_title(config),
        )
    except TypeError as exc:
        if "title" not in str(exc):
            raise
        write_interactive_candlestick_html(
            candles,
            args.chart_output,
            visible_candles=args.visible_candles,
        )

    summary = summarize(history)
    print("Synthetic market simulation complete")
    print(f"Worlds: {config.world_weights}")
    print(f"CSV written to: {args.output}")
    print(f"Scenario config written to: {config_output}")
    print(f"Interactive candlestick chart written to: {args.chart_output}")
    print(f"Candles generated: {len(candles)}")
    for key, value in summary.items():
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
