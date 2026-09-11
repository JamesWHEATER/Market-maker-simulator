"""Causal market-structure experiment runner for chart-pattern edge research.

This module is the downstream research layer for the frozen simulator / candle builder /
pattern detector stack.  It does **not** modify pattern geometry or use realised profitability to
change detector parameters.

Research design
---------------
* Enumerate every non-empty subset of the configured market worlds, optionally expanding each
  subset over a finite simplex lattice of population weights and an arbitrary pre-registered
  parameter grid.
* Generate independent discovery and validation seeds.
* Build identical-duration candles and call ``pattern_detector.detect_pattern_universe`` with a
  ``RunContext``.  Only ``DetectionResult.confirmed_breakout_events`` are strategy-time signals.
* Enter no earlier than ``earliest_execution_index`` and evaluate pre-registered forward horizons.
* Keep directional correctness separate from economic return.
* Compute net P&L after bid/ask spread, brokerage, and optional slippage.
* Skip new signals while a position is already open (default causal overlap policy).
* Compare each event with both unconditional opportunities and deterministic state-matched nulls.
* Aggregate inference across independent seeds rather than pretending events inside one path are
  independent observations.
* Apply Benjamini-Hochberg FDR correction in discovery and Holm family-wise correction to the
  pre-selected hypotheses in validation.
* Freeze the detector + statistical procedure in a manifest that can subsequently be applied to
  untouched real OHLCV files without retuning the detector or analysis settings.

"All possible variations" is mathematically impossible for continuous parameters.  This runner is
therefore exhaustive over the *finite grid specified before the experiment*.  With the default
settings it evaluates all 511 non-empty world subsets at equal weights.  ``--weight-denominator``
and ``--grid-json`` expand that finite design exhaustively.

The code is research infrastructure, not evidence of an edge by itself.  Statistical conclusions
must be based on the held-out validation partition and then on untouched real-market data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from chart_renderer import CANDLE_BUILDER_VERSION, PriceCandle, build_candles
from market_maker_simulator import (
    SIMULATOR_VERSION,
    RNG_STREAM_VERSION,
    WORLD_NAMES,
    SyntheticMarketConfig,
    canonical_behavior_config,
    candle_dataset_id,
    run_id as simulator_run_id,
    scenario_id as simulator_scenario_id,
    simulate_market,
)
from pattern_detector import (
    CANONICAL_PATTERN_NAMES,
    DEFAULT_PATTERN_CONFIG,
    DETECTOR_VERSION,
    DetectionResult,
    PatternDetection,
    RunContext,
    causal_signal_priority,
    detect_pattern_universe,
)


EXPERIMENT_RUNNER_VERSION = "1.7.0"
ALL_PATTERNS_LABEL = "__all_patterns__"
PATTERN_NAMES = CANONICAL_PATTERN_NAMES


# ---------------------------------------------------------------------------
# Immutable experiment configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Round-trip trading-cost assumptions expressed in basis points where applicable.

    Synthetic paths use their simulated bid/ask spread. Real files use explicit opening and
    closing bid/ask quotes, otherwise OHLC midpoints and ``real_spread_bps_fallback``.
    """

    brokerage_bps_per_side: float = 1.0
    slippage_bps_per_side: float = 0.0
    real_spread_bps_fallback: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "brokerage_bps_per_side",
            "slippage_bps_per_side",
            "real_spread_bps_fallback",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative real number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be a finite non-negative real number")
            object.__setattr__(self, name, numeric)


@dataclass(frozen=True, slots=True)
class ProcedureConfig:
    """Pre-registered statistical/execution procedure shared by discovery and validation."""

    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    steps_per_candle: int = 5
    overlap_policy: str = "skip_new"
    matched_nulls_per_event: int = 20
    minimum_matched_nulls_per_event: int = 5
    null_volatility_lookback: int = 20
    null_embargo_bars: int = 5
    alpha: float = 0.05
    confidence_level: float = 0.95
    minimum_events: int = 30
    minimum_seeds_with_events: int = 5
    discovery_top_k: int = 100
    cost_model: CostModel = CostModel()

    def __post_init__(self) -> None:
        if not isinstance(self.horizons, (tuple, list)) or not self.horizons or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.horizons
        ):
            raise ValueError("horizons must contain positive integers")
        horizons = tuple(sorted(set(self.horizons)))
        object.__setattr__(self, "horizons", horizons)
        if (
            isinstance(self.steps_per_candle, bool)
            or not isinstance(self.steps_per_candle, int)
            or self.steps_per_candle <= 0
        ):
            raise ValueError("steps_per_candle must be an integer > 0")
        if self.overlap_policy not in {"skip_new", "allow"}:
            raise ValueError("overlap_policy must be 'skip_new' or 'allow'")
        for name in (
            "matched_nulls_per_event",
            "minimum_matched_nulls_per_event",
            "null_volatility_lookback",
            "null_embargo_bars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.matched_nulls_per_event <= 0:
            raise ValueError("matched_nulls_per_event must be > 0")
        if self.minimum_matched_nulls_per_event <= 0:
            raise ValueError("minimum_matched_nulls_per_event must be > 0")
        if self.minimum_matched_nulls_per_event > self.matched_nulls_per_event:
            raise ValueError(
                "minimum_matched_nulls_per_event cannot exceed matched_nulls_per_event"
            )
        for name in ("alpha", "confidence_level"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
            object.__setattr__(self, name, value)
        for name in ("minimum_events", "minimum_seeds_with_events", "discovery_top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer > 0")
        if not isinstance(self.cost_model, CostModel):
            raise ValueError("cost_model must be a CostModel")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One seed-independent synthetic market structure."""

    structure_id: str
    label: str
    world_weights: tuple[tuple[str, float], ...]
    overrides: tuple[tuple[str, Any], ...] = ()

    def weights_dict(self) -> dict[str, float]:
        return dict(self.world_weights)

    def overrides_dict(self) -> dict[str, Any]:
        return dict(self.overrides)


@dataclass(frozen=True, slots=True)
class TradeObservation:
    pattern_name: str
    event_id: str
    horizon: int
    direction: int
    entry_index: int
    exit_index: int
    gross_return: float
    spread_slippage_cost: float
    brokerage_cost: float
    total_transaction_cost: float
    net_return: float
    directional_correct: bool
    unconditional_mean_net_return: float
    matched_null_mean_net_return: float
    matched_null_direction_accuracy: float
    matched_null_count: int
    geometry_fit_score: float

    @property
    def effect_vs_matched(self) -> float:
        return self.net_return - self.matched_null_mean_net_return

    @property
    def effect_vs_unconditional(self) -> float:
        return self.net_return - self.unconditional_mean_net_return


@dataclass(frozen=True, slots=True)
class RunMetric:
    """One inferential observation; legacy total_* fields hold additive fixed-notional points."""
    partition: str
    structure_id: str
    structure_label: str
    scenario_id: str
    run_id: str
    seed: int
    pattern_name: str
    horizon: int
    event_count: int
    matched_null_event_count: int
    skipped_overlap_count: int
    untradeable_event_count: int
    total_gross_return: float
    total_transaction_cost: float
    total_net_return: float
    mean_net_return: float
    median_net_return: float
    mean_effect_vs_matched: float
    mean_effect_vs_unconditional: float
    mean_matched_null_net_return: float
    mean_unconditional_net_return: float
    direction_correct_count: int
    direction_accuracy: float
    matched_null_direction_accuracy: float
    matched_event_direction_accuracy: float = float("nan")


@dataclass(frozen=True, slots=True)
class RealCandle:
    candle_number: int
    timestamp: datetime | str | int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class _PathEvaluationCache:
    trailing_volatility: list[float]
    all_tradeable_entries: list[int]
    eligible_by_horizon: dict[tuple[int, int], tuple[int, ...]]
    unconditional_by_horizon_direction: dict[tuple[int, int, int], tuple[float, float]]
    economic_prefix_fingerprints: tuple[str, ...]


# ---------------------------------------------------------------------------
# Stable hashing / provenance
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata cannot contain NaN or infinity")
        return value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _module_file_sha256(module_name: str) -> str:
    module = sys.modules.get(module_name)
    module_path = Path(getattr(module, "__file__", "") or "") if module is not None else Path()
    return _file_sha256(module_path) if module_path.is_file() else "unavailable"


def _detector_contract() -> dict[str, str]:
    empty = detect_pattern_universe([], DEFAULT_PATTERN_CONFIG)
    detector_module = Path(sys.modules[detect_pattern_universe.__module__].__file__ or "")
    checksum = _file_sha256(detector_module) if detector_module.is_file() else "unavailable"
    return {
        "detector_version": DETECTOR_VERSION,
        "detector_config_hash": empty.config_hash,
        "detector_sha256": checksum,
    }


def _procedure_payload(procedure: ProcedureConfig) -> dict[str, Any]:
    return {
        "runner_version": EXPERIMENT_RUNNER_VERSION,
        "runner_sha256": _file_sha256(Path(__file__)),
        "simulator_version": SIMULATOR_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "simulator_sha256": _module_file_sha256(SyntheticMarketConfig.__module__),
        "candle_builder_version": CANDLE_BUILDER_VERSION,
        "candle_builder_sha256": _module_file_sha256(build_candles.__module__),
        **_detector_contract(),
        "procedure": _jsonable(procedure),
        "discovery_correction": "Benjamini-Hochberg FDR",
        "validation_correction": "Holm family-wise error rate",
        "inference_unit": "independent_synthetic_seed_or_real_temporal_dependence_cluster",
        "return_accounting": "fixed_notional_additive_return_points; not compounded portfolio returns",
        "direction_accuracy_inference": "equal_weight_run_accuracy_t_interval; real temporal clusters",
        "primary_edge_test": (
            "max(one-sided p for positive total net return per independent run, "
            "one-sided p for positive matched-null excess); corrected across hypotheses"
        ),
        "trade_entry_semantics": "open_of_earliest_execution_index",
        "horizon_semantics": "horizon=1 exits at the same entry bar close; h exits at close(entry+h-1)",
        "null_matching": (
            "same-path nearest pre-entry volatility/spread state; event-scale warmup; "
            "exclude full signal return intervals plus embargo; provenance-independent causal seed"
        ),
    }


def procedure_hash(procedure: ProcedureConfig) -> str:
    return _sha256_text(_procedure_payload(procedure))[:20]


# ---------------------------------------------------------------------------
# Scenario-grid generation
# ---------------------------------------------------------------------------


def _positive_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for tail in _positive_compositions(total - first, parts - 1):
            yield (first, *tail)


def _world_weight_variations(worlds: tuple[str, ...], denominator: int) -> Iterable[dict[str, float]]:
    if denominator <= 0 or denominator < len(worlds):
        equal = 1.0 / len(worlds)
        yield {world: equal for world in worlds}
        return
    for composition in _positive_compositions(denominator, len(worlds)):
        yield {world: units / denominator for world, units in zip(worlds, composition)}


def _apply_overrides(config: SyntheticMarketConfig, overrides: Mapping[str, Any]) -> SyntheticMarketConfig:
    """Apply a dotted override set atomically, including coupled dataclass constraints."""

    tree: dict[str, Any] = {}
    for path, value in sorted(overrides.items()):
        if path in {"seed", "steps", "world_weights"}:
            raise ValueError(f"grid override {path!r} is reserved by the experiment runner")
        parts = path.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid parameter path {path!r}")
        node = tree
        for part in parts:
            if "__value__" in node:
                raise ValueError(f"overlapping configuration overrides include {path!r}")
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"overlapping configuration overrides include {path!r}")
        if node:
            raise ValueError(f"overlapping configuration overrides include {path!r}")
        node["__value__"] = value

    def apply_tree(obj: Any, updates: Mapping[str, Any], prefix: str = "") -> Any:
        changes: dict[str, Any] = {}
        for field_name, child_updates in updates.items():
            dotted = f"{prefix}.{field_name}" if prefix else field_name
            if not hasattr(obj, field_name):
                raise ValueError(f"unknown configuration field in grid: {dotted!r}")
            if "__value__" in child_updates:
                if len(child_updates) != 1:
                    raise ValueError(f"overlapping configuration overrides include {dotted!r}")
                changes[field_name] = child_updates["__value__"]
            else:
                child = getattr(obj, field_name)
                if not is_dataclass(child):
                    raise ValueError(f"configuration field {dotted!r} has no nested fields")
                changes[field_name] = apply_tree(child, child_updates, dotted)
        return replace(obj, **changes)

    return apply_tree(config, tree)


def _structure_id(config: SyntheticMarketConfig) -> str:
    """Canonical identity for one active market structure.

    Inactive world-specific configuration cannot affect a simulation and is reset before hashing,
    preventing a parameter grid from creating duplicate experiments with different labels/ids.
    """

    canonical = canonical_behavior_config(replace(
        config,
        steps=1,
        seed=0,
        world_weights=dict(config.normalized_world_weights),
    ))
    return "ST-" + simulator_scenario_id(canonical)


def _behavioral_overrides(
    baseline: SyntheticMarketConfig,
    target: SyntheticMarketConfig,
) -> dict[str, Any]:
    """Describe the canonical behavioral delta as dotted, atomically re-applicable fields."""

    output: dict[str, Any] = {}

    def visit(before: Any, after: Any, prefix: str = "") -> None:
        for item in fields(before):
            if not prefix and item.name in {"steps", "seed", "world_weights"}:
                continue
            left = getattr(before, item.name)
            right = getattr(after, item.name)
            path = f"{prefix}.{item.name}" if prefix else item.name
            if is_dataclass(left) and is_dataclass(right):
                visit(left, right, path)
            elif _jsonable(left) != _jsonable(right):
                output[path] = right

    visit(baseline, target)
    return output


def _scenario_label(weights: Mapping[str, float], overrides: Mapping[str, Any]) -> str:
    world_part = "+".join(f"{name}:{weights[name]:.4g}" for name in WORLD_NAMES if name in weights)
    if not overrides:
        return world_part
    override_part = ",".join(f"{k}={v}" for k, v in sorted(overrides.items()))
    return f"{world_part}|{override_part}"


def generate_scenarios(
    base_config: SyntheticMarketConfig,
    *,
    min_worlds: int = 1,
    max_worlds: int = len(WORLD_NAMES),
    weight_denominator: int = 0,
    parameter_grid: Mapping[str, Sequence[Any]] | None = None,
    max_scenarios: int = 0,
) -> list[ScenarioSpec]:
    """Enumerate the complete finite pre-registered synthetic structure grid."""

    if min_worlds < 1 or max_worlds > len(WORLD_NAMES) or min_worlds > max_worlds:
        raise ValueError("invalid min_worlds/max_worlds")
    if isinstance(weight_denominator, bool) or not isinstance(weight_denominator, int) or weight_denominator < 0:
        raise ValueError("weight_denominator must be an integer >= 0")
    parameter_grid = parameter_grid or {}
    grid_names = tuple(sorted(parameter_grid))
    grid_values: list[tuple[Any, ...]] = []
    for name in grid_names:
        values = tuple(parameter_grid[name])
        if not values:
            raise ValueError(f"parameter grid for {name!r} cannot be empty")
        grid_values.append(values)
    combinations = itertools.product(*grid_values) if grid_values else [()]
    override_sets = [dict(zip(grid_names, values)) for values in combinations]
    # Validate every complete grid point atomically even if its scoped world will be inactive in
    # all requested scenarios. This catches typos/invalid values without rejecting valid paired
    # updates merely because an intermediate dataclass state violates a coupled constraint.
    for overrides in override_sets:
        _apply_overrides(base_config, overrides)

    scenarios: list[ScenarioSpec] = []
    seen_structures: set[str] = set()
    for count in range(min_worlds, max_worlds + 1):
        for worlds in itertools.combinations(WORLD_NAMES, count):
            for weights in _world_weight_variations(worlds, weight_denominator):
                for overrides in override_sets:
                    raw_base = replace(base_config, steps=1, seed=0, world_weights=weights)
                    canonical_base = canonical_behavior_config(raw_base)
                    raw_target = _apply_overrides(raw_base, overrides)
                    cfg = canonical_behavior_config(raw_target)
                    active_overrides = _behavioral_overrides(canonical_base, cfg)
                    sid = _structure_id(cfg)
                    if sid in seen_structures:
                        continue
                    seen_structures.add(sid)
                    normalized = tuple((name, float(cfg.normalized_world_weights[name])) for name in cfg.active_worlds)
                    scenarios.append(
                        ScenarioSpec(
                            structure_id=sid,
                            label=_scenario_label(dict(normalized), active_overrides),
                            world_weights=normalized,
                            overrides=tuple(sorted(active_overrides.items())),
                        )
                    )
                    if max_scenarios and len(scenarios) >= max_scenarios:
                        return scenarios
    return scenarios


def _load_grid_json(path: Path | None) -> dict[str, tuple[Any, ...]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("grid JSON must be an object mapping dotted config paths to arrays")
    output: dict[str, tuple[Any, ...]] = {}
    for key, values in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("grid JSON keys must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid JSON value for {key!r} must be a non-empty array")
        output[key] = tuple(values)
    return output


# ---------------------------------------------------------------------------
# Price / spread / cost helpers
# ---------------------------------------------------------------------------


def trade_direction(event: PatternDetection) -> int | None:
    """Resolve execution direction only from a breakout known at signal availability.

    ``expected_direction`` is a descriptive geometry label and is deliberately ignored.  This
    gives every pattern family the same causal entry trigger.
    """

    if not event.breakout_confirmed_by_availability:
        return None
    breakout = event.metadata.get("breakout_direction_by_availability")
    if breakout == "bullish":
        return 1
    if breakout == "bearish":
        return -1
    return None


# Private compatibility alias for older imports inside this module/project.
_trade_direction = trade_direction


def _net_trade_return(
    entry_mid: float,
    exit_mid: float,
    direction: int,
    entry_spread_fraction: float,
    exit_spread_fraction: float,
    costs: CostModel,
) -> tuple[float, float, float, float, float]:
    """Return gross, spread/slippage cost, brokerage cost, total cost, net return.

    P&L is measured on one unit of the asset and normalised by the entry mid price.  Spread is paid
    as half-spread on entry and half-spread on exit.  Slippage is adverse on both sides.
    """

    for name, value in (
        ("entry_mid", entry_mid),
        ("exit_mid", exit_mid),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or +1")
    for name, value in (
        ("entry_spread_fraction", entry_spread_fraction),
        ("exit_spread_fraction", exit_spread_fraction),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    slippage = costs.slippage_bps_per_side / 10_000.0
    entry_half = 0.5 * entry_spread_fraction + slippage
    exit_half = 0.5 * exit_spread_fraction + slippage
    if entry_half >= 1.0 or exit_half >= 1.0:
        raise ValueError("spread/slippage is too large for a meaningful execution-price model")

    gross_return = direction * (exit_mid - entry_mid) / entry_mid
    if direction == 1:
        entry_exec = entry_mid * (1.0 + entry_half)
        exit_exec = exit_mid * (1.0 - exit_half)
        pre_broker_return = (exit_exec - entry_exec) / entry_mid
    else:
        entry_exec = entry_mid * (1.0 - entry_half)
        exit_exec = exit_mid * (1.0 + exit_half)
        pre_broker_return = (entry_exec - exit_exec) / entry_mid

    brokerage_rate = costs.brokerage_bps_per_side / 10_000.0
    brokerage_cost = brokerage_rate * (entry_exec + exit_exec) / entry_mid
    net_return = pre_broker_return - brokerage_cost
    spread_slippage_cost = gross_return - pre_broker_return
    total_cost = spread_slippage_cost + brokerage_cost
    # Identity check catches sign mistakes during future maintenance.
    if not math.isclose(gross_return - total_cost, net_return, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError("transaction-cost decomposition is inconsistent")
    return gross_return, spread_slippage_cost, brokerage_cost, total_cost, net_return


def _synthetic_execution_arrays(
    history: Sequence[Any],
    candles: Sequence[PriceCandle],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return exact synthetic quote midpoints and full-spread fractions for candle entry/exit.

    Entry is the market maker quote at the beginning of the candle's first simulator step.  Exit is
    the stored post-trade quote at the end of the candle's final step.  Using quote midpoints rather
    than the simulator's unskewed reference price preserves inventory-skew and tick effects: with
    zero additional slippage, ``net_trade_return`` reconstructs the actual bid/ask execution prices
    exactly from midpoint ± half-spread.

    Older snapshots without post-trade bid/ask fields remain readable through a conservative
    compatibility fallback based on the stored post-trade reference and opening end-step spread.
    """

    by_step = {snapshot.step: snapshot for snapshot in history}
    if len(by_step) != len(history):
        raise ValueError("synthetic history contains duplicate simulator step labels")
    entry_mids: list[float] = []
    exit_mids: list[float] = []
    entry_spreads: list[float] = []
    exit_spreads: list[float] = []
    for candle in candles:
        try:
            start = by_step[candle.start_step]
            end = by_step[candle.end_step]
        except KeyError as exc:
            raise ValueError("candle references a simulator step absent from history") from exc

        entry_bid = float(start.bid)
        entry_ask = float(start.ask)
        if not (
            math.isfinite(entry_bid)
            and math.isfinite(entry_ask)
            and entry_bid > 0.0
            and entry_ask >= entry_bid
        ):
            raise ValueError("synthetic opening quote is invalid")
        entry_mid = 0.5 * entry_bid + 0.5 * entry_ask

        post_bid = float(getattr(end, "post_trade_bid", 0.0) or 0.0)
        post_ask = float(getattr(end, "post_trade_ask", 0.0) or 0.0)
        if (
            math.isfinite(post_bid)
            and math.isfinite(post_ask)
            and post_bid > 0.0
            and post_ask >= post_bid
        ):
            exit_mid = 0.5 * post_bid + 0.5 * post_ask
            exit_spread = (post_ask - post_bid) / exit_mid
        else:
            if post_bid != 0.0 or post_ask != 0.0:
                raise ValueError("synthetic closing quote is invalid or incomplete")
            # Compatibility with v3.1/v3.2 snapshots.  New research runs always use the exact
            # post-trade quote above.
            exit_mid = max(float(end.post_trade_reference_price), sys.float_info.min)
            end_reference = max(float(end.reference_price), sys.float_info.min)
            exit_spread = max(0.0, float(end.spread) / end_reference)

        entry_mids.append(entry_mid)
        exit_mids.append(exit_mid)
        entry_spreads.append((entry_ask - entry_bid) / entry_mid)
        exit_spreads.append(max(0.0, exit_spread))
    return entry_mids, exit_mids, entry_spreads, exit_spreads


def net_trade_return(
    entry_mid: float,
    exit_mid: float,
    direction: int,
    entry_spread_fraction: float,
    exit_spread_fraction: float,
    costs: CostModel,
) -> tuple[float, float, float, float, float]:
    """Public shared execution-cost contract used by the statistical and AI layers."""

    return _net_trade_return(
        entry_mid,
        exit_mid,
        direction,
        entry_spread_fraction,
        exit_spread_fraction,
        costs,
    )


def synthetic_execution_arrays(
    history: Sequence[Any],
    candles: Sequence[PriceCandle],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Public shared synthetic execution arrays for downstream research layers."""

    return _synthetic_execution_arrays(history, candles)


def _trailing_volatility(closes: Sequence[float], lookback: int) -> list[float]:
    output = [0.0] * len(closes)
    if lookback <= 1:
        return output
    log_returns = [0.0]
    for index in range(1, len(closes)):
        log_returns.append(math.log(closes[index] / closes[index - 1]))
    for entry in range(len(closes)):
        # At the open of entry bar, only bars ending before this bar are complete.
        start = max(1, entry - lookback)
        sample = log_returns[start:entry]
        output[entry] = pstdev(sample) if len(sample) > 1 else 0.0
    return output


def _eligible_entry_indices(n_candles: int, horizon: int, minimum_entry: int) -> range:
    last_entry = n_candles - horizon
    if last_entry < minimum_entry:
        return range(0)
    return range(minimum_entry, last_entry + 1)


def _null_seed(economic_prefix: str, pattern_name: str, entry_index: int, horizon: int) -> int:
    """Sampling identity contains causal prices and strategy coordinates, never provenance."""
    payload = _canonical_json(
        [economic_prefix, pattern_name, entry_index, horizon, "matched-null-v2"]
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _event_minimum_entry(event: PatternDetection, result: DetectionResult) -> int:
    """Apply the same scale warm-up to a signal and its comparison population."""
    observation = event.metadata.get("observation_window_length")
    if observation is None:
        core = event.metadata.get("core_window_length")
        lengths = dict(result.scale_observation_lengths)
        observation = lengths.get(core, max(lengths.values(), default=1))
    delay = event.earliest_execution_index - event.available_at_index
    return max(1, int(observation) - 1 + delay)


def _economic_prefix_fingerprints(candles: Sequence[Any]) -> tuple[str, ...]:
    """OHLCV prefixes exclude filenames, run labels, timestamps and future data."""
    digest = hashlib.blake2b(digest_size=16)
    prefixes = [digest.hexdigest()]
    for candle in candles:
        values = [float(getattr(candle, name, 0.0)) for name in ("open", "high", "low", "close", "volume")]
        digest.update(_canonical_json(values).encode("ascii"))
        digest.update(b"\n")
        prefixes.append(digest.hexdigest())
    return tuple(prefixes)


def _candidate_null_indices(
    target_entry: int,
    eligible: Sequence[int],
    volatility: Sequence[float],
    entry_spreads: Sequence[float],
    forbidden_entries: Sequence[int],
    embargo: int,
    count: int,
    rng: random.Random,
    horizon: int = 1,
) -> list[int]:
    if horizon <= 0 or embargo < 0:
        raise ValueError("horizon must be positive and embargo non-negative")
    forbidden = set()
    # [N, N+H-1] must not intersect [P-embargo, P+H-1+embargo].
    for event_entry in (*forbidden_entries, target_entry):
        for index in range(max(0, event_entry - horizon + 1 - embargo), event_entry + horizon + embargo):
            forbidden.add(index)
    candidates = [index for index in eligible if index != target_entry and index not in forbidden]
    if not candidates:
        return []

    eps = 1e-12
    target_vol = volatility[target_entry]
    target_spread = entry_spreads[target_entry]

    def distance(index: int) -> tuple[float, int]:
        vol_distance = abs(math.log((volatility[index] + eps) / (target_vol + eps)))
        spread_distance = abs(math.log((entry_spreads[index] + eps) / (target_spread + eps)))
        return vol_distance + 0.25 * spread_distance, index

    ranked = sorted(candidates, key=distance)
    pool_size = min(len(ranked), max(count, count * 5))
    pool = ranked[:pool_size]
    if len(pool) <= count:
        return pool
    selected = rng.sample(pool, count)
    return sorted(selected)


def _evaluate_index_return(
    candles: Sequence[Any],
    entry_spreads: Sequence[float],
    exit_spreads: Sequence[float],
    entry_index: int,
    horizon: int,
    direction: int,
    costs: CostModel,
    *,
    entry_mids: Sequence[float] | None = None,
    exit_mids: Sequence[float] | None = None,
) -> tuple[float, float, float, float, float]:
    exit_index = entry_index + horizon - 1
    entry_mid = float(entry_mids[entry_index]) if entry_mids is not None else float(candles[entry_index].open)
    exit_mid = float(exit_mids[exit_index]) if exit_mids is not None else float(candles[exit_index].close)
    return _net_trade_return(
        entry_mid,
        exit_mid,
        direction,
        float(entry_spreads[entry_index]),
        float(exit_spreads[exit_index]),
        costs,
    )


# ---------------------------------------------------------------------------
# Event evaluation and overlap handling
# ---------------------------------------------------------------------------


def _prepare_path_evaluation_cache(
    *,
    result: DetectionResult,
    candles: Sequence[Any],
    entry_spreads: Sequence[float],
    exit_spreads: Sequence[float],
    entry_mids: Sequence[float] | None,
    exit_mids: Sequence[float] | None,
    procedure: ProcedureConfig,
) -> _PathEvaluationCache:
    closes = [float(candle.close) for candle in candles]
    trailing_vol = _trailing_volatility(closes, procedure.null_volatility_lookback)
    confirmed_events = result.confirmed_breakout_events
    minimum_entries = sorted({_event_minimum_entry(event, result) for event in confirmed_events})
    all_tradeable_entries = [
        int(event.earliest_execution_index)
        for event in confirmed_events
        if _trade_direction(event) is not None and event.earliest_execution_index is not None
    ]
    eligible_by_horizon: dict[tuple[int, int], tuple[int, ...]] = {}
    unconditional: dict[tuple[int, int, int], tuple[float, float]] = {}
    for horizon, minimum_entry in itertools.product(procedure.horizons, minimum_entries):
        eligible = tuple(_eligible_entry_indices(len(candles), horizon, minimum_entry))
        eligible_by_horizon[(horizon, minimum_entry)] = eligible
        for direction in (-1, 1):
            values: list[float] = []
            hits = 0
            for index in eligible:
                gross, _, _, _, net = _evaluate_index_return(
                    candles,
                    entry_spreads,
                    exit_spreads,
                    index,
                    horizon,
                    direction,
                    procedure.cost_model,
                    entry_mids=entry_mids,
                    exit_mids=exit_mids,
                )
                values.append(net)
                hits += int(gross > 0.0)
            unconditional[(horizon, minimum_entry, direction)] = (
                mean(values) if values else 0.0,
                hits / len(values) if values else 0.0,
            )
    return _PathEvaluationCache(
        trailing_volatility=trailing_vol,
        all_tradeable_entries=all_tradeable_entries,
        eligible_by_horizon=eligible_by_horizon,
        unconditional_by_horizon_direction=unconditional,
        economic_prefix_fingerprints=_economic_prefix_fingerprints(candles),
    )


def _filter_overlaps(
    events: Sequence[PatternDetection],
    horizon: int,
    policy: str,
    n_candles: int,
) -> tuple[list[PatternDetection], int, int]:
    """Apply a causal one-position overlap policy without comparing cross-pattern fit scores.

    Events available for the same entry bar are first resolved as a simultaneous signal set.
    Same-direction duplicates collapse to one deterministic representative.  If bullish and bearish
    signals conflict on the same bar, the whole simultaneous set is skipped as ambiguous.  With
    ``skip_new`` a later signal is ignored while the prior accepted trade is still open.
    """

    tradeable: list[tuple[PatternDetection, int, int, int]] = []
    untradeable = 0
    for event in events:
        direction = _trade_direction(event)
        entry = event.earliest_execution_index
        if direction is None or entry is None:
            untradeable += 1
            continue
        exit_index = int(entry) + horizon - 1
        if int(entry) < 0 or exit_index >= n_candles:
            untradeable += 1
            continue
        tradeable.append((event, int(entry), exit_index, direction))

    tradeable.sort(
        key=lambda item: (
            item[1],
            causal_signal_priority(
                item[0].pattern_name,
                item[0].available_at_index,
                item[0].start_index,
                item[0].end_index,
                item[0].metadata.get("observation_window_length", 0),
                item[0].metadata.get("core_window_length", 0),
                item[0].geometry_fit_score,
            ),
        )
    )

    simultaneous_resolved: list[tuple[PatternDetection, int, int]] = []
    skipped = 0
    for entry, group_iter in itertools.groupby(tradeable, key=lambda item: item[1]):
        group = list(group_iter)
        directions = {item[3] for item in group}
        if len(directions) > 1:
            skipped += len(group)
            continue
        representative = group[0]
        simultaneous_resolved.append((representative[0], entry, representative[2]))
        skipped += len(group) - 1

    if policy == "allow":
        return [item[0] for item in simultaneous_resolved], skipped, untradeable

    accepted: list[PatternDetection] = []
    open_until = -1
    for event, entry, exit_index in simultaneous_resolved:
        if entry <= open_until:
            skipped += 1
            continue
        accepted.append(event)
        open_until = exit_index
    return accepted, skipped, untradeable


def _all_pattern_events(result: DetectionResult) -> list[PatternDetection]:
    return list(result.confirmed_breakout_events)


def _events_for_pattern(result: DetectionResult, pattern_name: str) -> list[PatternDetection]:
    if pattern_name == ALL_PATTERNS_LABEL:
        return _all_pattern_events(result)
    return [
        event
        for event in result.confirmed_breakout_events
        if event.pattern_name == pattern_name
    ]


def _evaluate_hypothesis_on_path(
    *,
    result: DetectionResult,
    candles: Sequence[Any],
    entry_spreads: Sequence[float],
    exit_spreads: Sequence[float],
    entry_mids: Sequence[float] | None,
    exit_mids: Sequence[float] | None,
    pattern_name: str,
    horizon: int,
    run_identity: str,
    procedure: ProcedureConfig,
    cache: _PathEvaluationCache,
) -> tuple[list[TradeObservation], int, int]:
    events = _events_for_pattern(result, pattern_name)
    accepted, skipped_overlap, untradeable = _filter_overlaps(
        events, horizon, procedure.overlap_policy, len(candles)
    )
    trailing_vol = cache.trailing_volatility
    all_tradeable_entries = cache.all_tradeable_entries

    observations: list[TradeObservation] = []
    for event in accepted:
        direction = _trade_direction(event)
        entry = event.earliest_execution_index
        if direction is None or entry is None:
            continue
        entry = int(entry)
        exit_index = entry + horizon - 1
        if entry < 0 or exit_index >= len(candles):
            continue
        minimum_entry = _event_minimum_entry(event, result)
        if entry < minimum_entry:
            raise ValueError("pattern event precedes its own detector-scale warm-up")
        eligible = cache.eligible_by_horizon[(horizon, minimum_entry)]
        gross, spread_slippage, brokerage, total_cost, net = _evaluate_index_return(
            candles,
            entry_spreads,
            exit_spreads,
            entry,
            horizon,
            direction,
            procedure.cost_model,
            entry_mids=entry_mids,
            exit_mids=exit_mids,
        )

        rng = random.Random(_null_seed(
            cache.economic_prefix_fingerprints[event.available_at_index + 1],
            event.pattern_name, entry, horizon,
        ))
        null_indices = _candidate_null_indices(
            target_entry=entry,
            eligible=eligible,
            volatility=trailing_vol,
            entry_spreads=entry_spreads,
            forbidden_entries=all_tradeable_entries,
            embargo=procedure.null_embargo_bars,
            count=procedure.matched_nulls_per_event,
            rng=rng,
            horizon=horizon,
        )
        null_returns: list[float] = []
        null_hits = 0
        for null_entry in null_indices:
            null_gross, _, _, _, null_net = _evaluate_index_return(
                candles,
                entry_spreads,
                exit_spreads,
                null_entry,
                horizon,
                direction,
                procedure.cost_model,
                entry_mids=entry_mids,
                exit_mids=exit_mids,
            )
            null_returns.append(null_net)
            null_hits += int(null_gross > 0.0)
        # A state-matched comparison must not silently become an unconditional comparison when
        # the path contains too few eligible controls.  Those two estimands answer different
        # questions.  Insufficiently matched events therefore retain NaN matched-null fields and
        # are excluded from the matched-null inference while remaining valid strategy outcomes.
        if len(null_returns) >= procedure.minimum_matched_nulls_per_event:
            null_mean = mean(null_returns)
            null_accuracy = null_hits / len(null_returns)
        else:
            null_mean = float("nan")
            null_accuracy = float("nan")

        observations.append(
            TradeObservation(
                pattern_name=pattern_name,
                event_id=event.event_id or "",
                horizon=horizon,
                direction=direction,
                entry_index=entry,
                exit_index=exit_index,
                gross_return=gross,
                spread_slippage_cost=spread_slippage,
                brokerage_cost=brokerage,
                total_transaction_cost=total_cost,
                net_return=net,
                directional_correct=gross > 0.0,
                unconditional_mean_net_return=cache.unconditional_by_horizon_direction[
                    (horizon, minimum_entry, direction)
                ][0],
                matched_null_mean_net_return=null_mean,
                matched_null_direction_accuracy=null_accuracy,
                matched_null_count=len(null_indices),
                geometry_fit_score=float(event.geometry_fit_score),
            )
        )
    return observations, skipped_overlap, untradeable


def _metric_from_observations(
    *,
    partition: str,
    structure: ScenarioSpec,
    scenario_id: str,
    run_id: str,
    seed: int,
    pattern_name: str,
    horizon: int,
    observations: Sequence[TradeObservation],
    skipped_overlap: int,
    untradeable: int,
) -> RunMetric:
    if not observations:
        return RunMetric(
            partition=partition,
            structure_id=structure.structure_id,
            structure_label=structure.label,
            scenario_id=scenario_id,
            run_id=run_id,
            seed=seed,
            pattern_name=pattern_name,
            horizon=horizon,
            event_count=0,
            matched_null_event_count=0,
            skipped_overlap_count=skipped_overlap,
            untradeable_event_count=untradeable,
            total_gross_return=0.0,
            total_transaction_cost=0.0,
            total_net_return=0.0,
            mean_net_return=0.0,
            median_net_return=0.0,
            mean_effect_vs_matched=float("nan"),
            mean_effect_vs_unconditional=float("nan"),
            mean_matched_null_net_return=float("nan"),
            mean_unconditional_net_return=float("nan"),
            direction_correct_count=0,
            direction_accuracy=float("nan"),
            matched_null_direction_accuracy=float("nan"),
        )
    nets = [obs.net_return for obs in observations]
    matched = [
        obs
        for obs in observations
        if obs.matched_null_count >= 1
        and math.isfinite(obs.matched_null_mean_net_return)
        and math.isfinite(obs.matched_null_direction_accuracy)
    ]
    return RunMetric(
        partition=partition,
        structure_id=structure.structure_id,
        structure_label=structure.label,
        scenario_id=scenario_id,
        run_id=run_id,
        seed=seed,
        pattern_name=pattern_name,
        horizon=horizon,
        event_count=len(observations),
        matched_null_event_count=len(matched),
        skipped_overlap_count=skipped_overlap,
        untradeable_event_count=untradeable,
        total_gross_return=sum(obs.gross_return for obs in observations),
        total_transaction_cost=sum(obs.total_transaction_cost for obs in observations),
        total_net_return=sum(nets),
        mean_net_return=mean(nets),
        median_net_return=median(nets),
        mean_effect_vs_matched=(
            mean(obs.effect_vs_matched for obs in matched) if matched else float("nan")
        ),
        mean_effect_vs_unconditional=mean(obs.effect_vs_unconditional for obs in observations),
        mean_matched_null_net_return=(
            mean(obs.matched_null_mean_net_return for obs in matched)
            if matched
            else float("nan")
        ),
        mean_unconditional_net_return=mean(obs.unconditional_mean_net_return for obs in observations),
        direction_correct_count=sum(int(obs.directional_correct) for obs in observations),
        direction_accuracy=mean(float(obs.directional_correct) for obs in observations),
        matched_null_direction_accuracy=(
            mean(obs.matched_null_direction_accuracy for obs in matched)
            if matched
            else float("nan")
        ),
        matched_event_direction_accuracy=(
            mean(float(obs.directional_correct) for obs in matched) if matched else float("nan")
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic task execution
# ---------------------------------------------------------------------------


def _build_market_config(
    base_config: SyntheticMarketConfig,
    structure: ScenarioSpec,
    seed: int,
    steps: int,
) -> SyntheticMarketConfig:
    cfg = replace(
        base_config,
        steps=steps,
        seed=seed,
        world_weights=structure.weights_dict(),
    )
    cfg = _apply_overrides(cfg, structure.overrides_dict())
    if _structure_id(cfg) != structure.structure_id:
        raise AssertionError("scenario structure_id changed while applying seed/steps")
    return cfg


def _run_synthetic_task(
    structure: ScenarioSpec,
    seed: int,
    partition: str,
    steps: int,
    base_config: SyntheticMarketConfig,
    procedure: ProcedureConfig,
    git_commit: str | None,
) -> list[RunMetric]:
    # NumPy/BLAS is already parallel internally on some platforms.  Restrict each process to one
    # native thread so a many-process research sweep does not oversubscribe the machine by 10–50x.
    with threadpool_limits(limits=1):
        cfg = _build_market_config(base_config, structure, seed, steps)
        history = simulate_market(cfg, record_step_diagnostics=False)
        if any(snapshot.arrival_cap_hit for snapshot in history):
            raise RuntimeError(
                "synthetic arrival cap was hit; raise max_arrivals_per_world_per_step or reduce "
                "the pre-registered arrival-rate grid before using this run for inference"
            )
        candles = build_candles(
            history,
            procedure.steps_per_candle,
            include_partial=False,
            require_contiguous_steps=True,
        )
        dataset_id = candle_dataset_id(cfg, procedure.steps_per_candle, include_partial=False)
        rid = simulator_run_id(cfg)
        context = RunContext(
            run_id=rid,
            market_world=structure.label,
            seed=seed,
            asset=f"synthetic:{structure.structure_id}",
            timeframe=f"{procedure.steps_per_candle}_sim_steps",
            dataset_id=dataset_id,
            git_commit=git_commit,
        )
        result = detect_pattern_universe(candles, DEFAULT_PATTERN_CONFIG, context)
        entry_mids, exit_mids, entry_spreads, exit_spreads = _synthetic_execution_arrays(history, candles)
        cache = _prepare_path_evaluation_cache(
            result=result,
            candles=candles,
            entry_spreads=entry_spreads,
            exit_spreads=exit_spreads,
            entry_mids=entry_mids,
            exit_mids=exit_mids,
            procedure=procedure,
        )

        metrics: list[RunMetric] = []
        for pattern_name in (*PATTERN_NAMES, ALL_PATTERNS_LABEL):
            for horizon in procedure.horizons:
                observations, skipped, untradeable = _evaluate_hypothesis_on_path(
                    result=result,
                    candles=candles,
                    entry_spreads=entry_spreads,
                    exit_spreads=exit_spreads,
                    entry_mids=entry_mids,
                    exit_mids=exit_mids,
                    pattern_name=pattern_name,
                    horizon=horizon,
                    run_identity=rid,
                    procedure=procedure,
                    cache=cache,
                )
                metrics.append(
                    _metric_from_observations(
                        partition=partition,
                        structure=structure,
                        scenario_id=simulator_scenario_id(cfg),
                        run_id=rid,
                        seed=seed,
                        pattern_name=pattern_name,
                        horizon=horizon,
                        observations=observations,
                        skipped_overlap=skipped,
                        untradeable=untradeable,
                    )
                )
        return metrics


def run_synthetic_partition(
    scenarios: Sequence[ScenarioSpec],
    seeds: Sequence[int],
    partition: str,
    *,
    steps: int,
    base_config: SyntheticMarketConfig,
    procedure: ProcedureConfig,
    workers: int,
    git_commit: str | None = None,
) -> list[RunMetric]:
    """Evaluate a synthetic partition with bounded process scheduling.

    The public return type is retained for downstream compatibility.  Process futures are bounded
    to O(workers), however, so a large finite scenario grid does not allocate one Future object per
    scenario/seed combination before any work has completed.
    """

    if partition not in {"discovery", "validation"}:
        raise ValueError("partition must be discovery or validation")
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("each synthetic seed must occur only once per partition")
    if len({scenario.structure_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("each synthetic structure must occur only once per partition")
    total_tasks = len(scenarios) * len(seeds)
    if total_tasks == 0:
        return []
    task_iter = (
        (scenario, seed)
        for scenario in scenarios
        for seed in seeds
    )
    results: list[RunMetric] = []
    progress_every = max(1, min(25, total_tasks))

    if workers <= 1:
        for index, (scenario, seed) in enumerate(task_iter, start=1):
            results.extend(
                _run_synthetic_task(
                    scenario,
                    seed,
                    partition,
                    steps,
                    base_config,
                    procedure,
                    git_commit,
                )
            )
            if index % progress_every == 0 or index == total_tasks:
                print(
                    f"[{partition}] completed {index}/{total_tasks} scenario-seed tasks",
                    flush=True,
                )
    else:
        max_in_flight = max(workers, workers * 2)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map: dict[Any, tuple[str, int]] = {}

            def submit_until_full() -> None:
                while len(future_map) < max_in_flight:
                    try:
                        scenario, seed = next(task_iter)
                    except StopIteration:
                        return
                    future = pool.submit(
                        _run_synthetic_task,
                        scenario,
                        seed,
                        partition,
                        steps,
                        base_config,
                        procedure,
                        git_commit,
                    )
                    future_map[future] = (scenario.structure_id, seed)

            submit_until_full()
            completed = 0
            while future_map:
                done, _not_done = wait(tuple(future_map), return_when=FIRST_COMPLETED)
                for future in done:
                    structure_id, seed = future_map.pop(future)
                    try:
                        results.extend(future.result())
                    except Exception as exc:
                        raise RuntimeError(
                            f"{partition} scenario {structure_id} seed {seed} failed"
                        ) from exc
                    completed += 1
                submit_until_full()
                if completed % progress_every == 0 or completed == total_tasks:
                    print(
                        f"[{partition}] completed {completed}/{total_tasks} scenario-seed tasks",
                        flush=True,
                    )

    results.sort(key=lambda row: (row.structure_id, row.seed, row.pattern_name, row.horizon))
    return results


# ---------------------------------------------------------------------------
# Statistical aggregation / multiple testing
# ---------------------------------------------------------------------------


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _mean_ci(values: Sequence[float], confidence: float) -> tuple[float, float, float]:
    clean = _finite(values)
    if not clean:
        return float("nan"), float("nan"), float("nan")
    center = mean(clean)
    if len(clean) < 2:
        return center, float("nan"), float("nan")
    sd = float(np.std(np.asarray(clean, dtype=float), ddof=1))
    if sd == 0.0:
        return center, center, center
    sem = sd / math.sqrt(len(clean))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=len(clean) - 1))
    return center, center - critical * sem, center + critical * sem


def _one_sided_positive_t_pvalue(values: Sequence[float]) -> float:
    clean = _finite(values)
    if len(clean) < 2:
        return 1.0
    sd = float(np.std(np.asarray(clean, dtype=float), ddof=1))
    center = mean(clean)
    if sd == 0.0:
        return 0.0 if center > 0.0 else 1.0
    result = stats.ttest_1samp(np.asarray(clean, dtype=float), popmean=0.0, alternative="greater")
    p = float(result.pvalue)
    return p if math.isfinite(p) else 1.0


def _wilson_interval(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def aggregate_metrics(metrics: Sequence[RunMetric], procedure: ProcedureConfig) -> list[dict[str, Any]]:
    if len({metric.partition for metric in metrics}) > 1:
        raise ValueError("cannot pool discovery and validation partitions")
    grouped: dict[tuple[str, str, int], list[RunMetric]] = {}
    for metric in metrics:
        grouped.setdefault((metric.structure_id, metric.pattern_name, metric.horizon), []).append(metric)

    rows: list[dict[str, Any]] = []
    for (structure_id, pattern_name, horizon), group in sorted(grouped.items()):
        if len({item.run_id for item in group}) != len(group) or len({item.seed for item in group}) != len(group):
            raise ValueError("duplicate run/seed observations would inflate inferential sample size")
        label = group[0].structure_label
        total_events = sum(item.event_count for item in group)
        seed_with_events = [item for item in group if item.event_count > 0]
        matched_null_event_count = sum(item.matched_null_event_count for item in group)
        seed_with_matched_nulls = [
            item
            for item in group
            if item.matched_null_event_count > 0 and math.isfinite(item.mean_effect_vs_matched)
        ]
        seed_effects = _finite(item.mean_effect_vs_matched for item in seed_with_matched_nulls)
        seed_unconditional_effects = _finite(item.mean_effect_vs_unconditional for item in seed_with_events)
        seed_mean_nets = _finite(item.mean_net_return for item in seed_with_events)
        # Zero-trade seeds contribute zero to total strategy return, which is the economically
        # correct convention when asking what one market structure earns per independent path.
        total_return_by_seed = [item.total_net_return for item in group]

        event_weighted_net = (
            sum(item.mean_net_return * item.event_count for item in seed_with_events) / total_events
            if total_events
            else 0.0
        )
        event_weighted_gross = (
            sum(item.total_gross_return for item in group) / total_events if total_events else 0.0
        )
        event_weighted_cost = (
            sum(item.total_transaction_cost for item in group) / total_events if total_events else 0.0
        )
        direction_hits_exact = sum(item.direction_correct_count for item in seed_with_events)
        event_weighted_direction_accuracy = direction_hits_exact / total_events if total_events else float("nan")
        direction_accuracy, direction_lo, direction_hi = _mean_ci(
            [item.direction_accuracy for item in seed_with_events], procedure.confidence_level
        )
        direction_lo = max(0.0, direction_lo) if math.isfinite(direction_lo) else direction_lo
        direction_hi = min(1.0, direction_hi) if math.isfinite(direction_hi) else direction_hi
        mean_seed_net, seed_net_lo, seed_net_hi = _mean_ci(seed_mean_nets, procedure.confidence_level)
        mean_effect, effect_lo, effect_hi = _mean_ci(seed_effects, procedure.confidence_level)
        mean_total_seed, total_seed_lo, total_seed_hi = _mean_ci(
            total_return_by_seed, procedure.confidence_level
        )
        effect_sd = float(np.std(np.asarray(seed_effects), ddof=1)) if len(seed_effects) > 1 else float("nan")
        standardized = mean_effect / effect_sd if math.isfinite(effect_sd) and effect_sd > 0.0 else float("nan")

        p_effect = _one_sided_positive_t_pvalue(seed_effects)
        p_total_profit = _one_sided_positive_t_pvalue(total_return_by_seed)
        # The primary edge claim requires both (a) positive net strategy profit after costs and
        # (b) positive excess return versus the matched null.  max(p1, p2) is a conservative
        # intersection-union p-value for that joint claim.
        p_primary_edge = max(p_effect, p_total_profit)

        matched_direction_values = _finite(
            item.matched_null_direction_accuracy for item in seed_with_matched_nulls
        )
        matched_direction_accuracy_event_weighted = (
            sum(
                item.matched_null_direction_accuracy * item.matched_null_event_count
                for item in seed_with_matched_nulls
                if math.isfinite(item.matched_null_direction_accuracy)
            ) / matched_null_event_count
            if matched_null_event_count
            else float("nan")
        )
        direction_lifts = _finite(
            item.matched_event_direction_accuracy - item.matched_null_direction_accuracy
            for item in seed_with_matched_nulls
            if math.isfinite(item.matched_event_direction_accuracy) and math.isfinite(item.matched_null_direction_accuracy)
        )
        mean_direction_lift, direction_lift_lo, direction_lift_hi = _mean_ci(
            direction_lifts, procedure.confidence_level
        )
        rows.append(
            {
                "structure_id": structure_id,
                "structure_label": label,
                "pattern_name": pattern_name,
                "horizon": horizon,
                "seed_count_total": len(group),
                "seed_count_with_events": len(seed_with_events),
                "seed_count_with_matched_nulls": len(seed_with_matched_nulls),
                "event_count": total_events,
                "matched_null_event_count": matched_null_event_count,
                "skipped_overlap_count": sum(item.skipped_overlap_count for item in group),
                "untradeable_event_count": sum(item.untradeable_event_count for item in group),
                "net_return_points_sum_across_seeds": sum(item.total_net_return for item in group),
                "mean_net_return_points_per_seed": mean_total_seed,
                "mean_net_return_points_per_seed_ci_low": total_seed_lo,
                "mean_net_return_points_per_seed_ci_high": total_seed_hi,
                "return_accounting": "fixed_notional_additive_return_points",
                "mean_gross_return_per_trade": event_weighted_gross,
                "mean_transaction_cost_per_trade": event_weighted_cost,
                "mean_net_return_per_trade": event_weighted_net,
                "mean_seed_net_return": mean_seed_net,
                "mean_seed_net_return_ci_low": seed_net_lo,
                "mean_seed_net_return_ci_high": seed_net_hi,
                "mean_effect_vs_matched": mean_effect,
                "mean_effect_vs_matched_ci_low": effect_lo,
                "mean_effect_vs_matched_ci_high": effect_hi,
                "mean_effect_vs_unconditional": (
                    mean(seed_unconditional_effects)
                    if seed_unconditional_effects
                    else float("nan")
                ),
                "standardized_seed_effect": standardized,
                "direction_accuracy": direction_accuracy,
                "direction_accuracy_event_weighted_descriptive": event_weighted_direction_accuracy,
                "direction_accuracy_inference_unit": "independent_run_or_dependence_cluster",
                "direction_accuracy_ci_low": direction_lo,
                "direction_accuracy_ci_high": direction_hi,
                "matched_null_direction_accuracy": matched_direction_accuracy_event_weighted,
                "mean_seed_matched_null_direction_accuracy": (
                    mean(matched_direction_values)
                    if matched_direction_values
                    else float("nan")
                ),
                "direction_accuracy_lift_vs_matched": mean_direction_lift,
                "direction_accuracy_lift_ci_low": direction_lift_lo,
                "direction_accuracy_lift_ci_high": direction_lift_hi,
                "raw_p_value_effect_vs_matched": p_effect,
                "raw_p_value_positive_total_net_return": p_total_profit,
                "raw_p_value_primary_edge": p_primary_edge,
            }
        )
    return rows


def _benjamini_hochberg(rows: list[dict[str, Any]], p_key: str, output_key: str) -> None:
    indexed = [(index, min(1.0, max(0.0, float(row[p_key])))) for index, row in enumerate(rows)]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end in range(m - 1, -1, -1):
        original_index, p = indexed[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted[rank_from_end] = min(1.0, running)
    for position, (original_index, _) in enumerate(indexed):
        rows[original_index][output_key] = adjusted[position]


def _holm_adjust(rows: list[dict[str, Any]], p_key: str, output_key: str) -> None:
    indexed = [(index, min(1.0, max(0.0, float(row[p_key])))) for index, row in enumerate(rows)]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for position, (original_index, p) in enumerate(indexed):
        adjusted = min(1.0, (m - position) * p)
        running = max(running, adjusted)
        rows[original_index][output_key] = running


def select_discovery_hypotheses(
    rows: list[dict[str, Any]],
    procedure: ProcedureConfig,
) -> list[dict[str, Any]]:
    _benjamini_hochberg(rows, "raw_p_value_primary_edge", "discovery_fdr_q_value")
    selected = [
        row
        for row in rows
        if row["event_count"] >= procedure.minimum_events
        and row["matched_null_event_count"] >= procedure.minimum_events
        and row["seed_count_with_events"] >= procedure.minimum_seeds_with_events
        and row["seed_count_with_matched_nulls"] >= procedure.minimum_seeds_with_events
        and row["mean_net_return_per_trade"] > 0.0
        and math.isfinite(float(row["mean_effect_vs_matched"]))
        and row["mean_effect_vs_matched"] > 0.0
        and row["discovery_fdr_q_value"] <= procedure.alpha
    ]
    selected.sort(
        key=lambda row: (
            row["discovery_fdr_q_value"],
            -row["mean_effect_vs_matched"],
            -row["mean_net_return_per_trade"],
            -row["event_count"],
        )
    )
    return selected[: procedure.discovery_top_k]


def validate_selected_hypotheses(
    validation_rows: list[dict[str, Any]],
    selected_discovery: Sequence[dict[str, Any]],
    procedure: ProcedureConfig,
) -> list[dict[str, Any]]:
    selected_keys = {
        (row["structure_id"], row["pattern_name"], int(row["horizon"]))
        for row in selected_discovery
    }
    selected_rows = [
        row
        for row in validation_rows
        if (row["structure_id"], row["pattern_name"], int(row["horizon"])) in selected_keys
    ]
    observed_keys = [(row["structure_id"], row["pattern_name"], int(row["horizon"])) for row in selected_rows]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != selected_keys:
        raise ValueError("validation must contain exactly one result for every pre-selected hypothesis")
    if selected_rows:
        _holm_adjust(selected_rows, "raw_p_value_primary_edge", "validation_holm_p_value")
    for row in selected_rows:
        row["validated"] = bool(
            row["event_count"] >= procedure.minimum_events
            and row["matched_null_event_count"] >= procedure.minimum_events
            and row["seed_count_with_events"] >= procedure.minimum_seeds_with_events
            and row["seed_count_with_matched_nulls"] >= procedure.minimum_seeds_with_events
            and row["mean_net_return_per_trade"] > 0.0
            and math.isfinite(float(row["mean_effect_vs_matched"]))
            and row["mean_effect_vs_matched"] > 0.0
            and row["validation_holm_p_value"] <= procedure.alpha
        )
    selected_rows.sort(
        key=lambda row: (
            not row["validated"],
            row["validation_holm_p_value"],
            -row["mean_effect_vs_matched"],
            -row["mean_net_return_per_trade"],
        )
    )
    return selected_rows


def structure_rankings(validated_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_structure: dict[str, list[dict[str, Any]]] = {}
    for row in validated_rows:
        if row.get("validated"):
            by_structure.setdefault(str(row["structure_id"]), []).append(row)
    rankings: list[dict[str, Any]] = []
    for structure_id, rows in by_structure.items():
        best = max(
            rows,
            key=lambda row: (
                row["mean_net_return_per_trade"],
                row["mean_effect_vs_matched"],
                -row["validation_holm_p_value"],
            ),
        )
        portfolio_rows = [row for row in rows if row["pattern_name"] == ALL_PATTERNS_LABEL]
        best_portfolio = (
            max(portfolio_rows, key=lambda row: row["mean_net_return_points_per_seed"])
            if portfolio_rows
            else None
        )
        rankings.append(
            {
                "structure_id": structure_id,
                "structure_label": best["structure_label"],
                "validated_hypothesis_count": len(rows),
                "best_pattern": best["pattern_name"],
                "best_horizon": best["horizon"],
                "best_mean_net_return_per_trade": best["mean_net_return_per_trade"],
                "best_mean_effect_vs_matched": best["mean_effect_vs_matched"],
                "best_validation_holm_p_value": best["validation_holm_p_value"],
                "best_event_count": best["event_count"],
                "best_portfolio_horizon": best_portfolio["horizon"] if best_portfolio else None,
                "best_portfolio_mean_net_return_points_per_seed": (
                    best_portfolio["mean_net_return_points_per_seed"] if best_portfolio else None
                ),
            }
        )
    rankings.sort(
        key=lambda row: (
            -(
                row["best_portfolio_mean_net_return_points_per_seed"]
                if row["best_portfolio_mean_net_return_points_per_seed"] is not None
                else -math.inf
            ),
            -row["best_mean_net_return_per_trade"],
            row["best_validation_holm_p_value"],
        )
    )
    for index, row in enumerate(rankings, start=1):
        row["rank"] = index
    return rankings


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------


def _write_dict_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        os.replace(temporary, path)
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, float):
                    clean[key] = "" if not math.isfinite(value) else format(value, ".17g")
                else:
                    clean[key] = value
            writer.writerow(clean)
    os.replace(temporary, path)


def _write_run_metrics(metrics: Sequence[RunMetric], path: Path) -> None:
    rows = []
    for metric in metrics:
        row = asdict(metric)
        row["net_return_points"] = row.pop("total_net_return")
        row["gross_return_points"] = row.pop("total_gross_return")
        row["transaction_cost_return_points"] = row.pop("total_transaction_cost")
        row["return_accounting"] = "fixed_notional_additive_return_points"
        rows.append(row)
    _write_dict_csv(rows, path)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Real-market validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealDatasetSpec:
    """Frozen real-data conventions, supplied explicitly rather than inferred from a filename.

    Only continuous, regularly spaced, timezone-aware bars are supported. Raw OHLC requires a
    declaration that no corporate actions occur in the window; adjusted data must adjust all
    four OHLC columns and any execution quotes consistently. ``dependence_group`` can merge
    additional known dependent windows beyond the automatic temporal overlap clustering.
    """

    asset: str
    venue: str
    timeframe: str
    price_basis: str
    corporate_action_policy: str
    session_policy: str = "continuous"
    timestamp_semantics: str = "bar_open"
    dependence_group: str | None = None

    def __post_init__(self) -> None:
        for name in ("asset", "venue", "timeframe", "price_basis", "corporate_action_policy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"real dataset {name} must be explicitly specified")
            object.__setattr__(self, name, value.strip())
        if self.session_policy != "continuous":
            raise ValueError("only the frozen 'continuous' session policy is supported")
        if self.timestamp_semantics != "bar_open":
            raise ValueError("real timestamps must label bar opens (timestamp_semantics='bar_open')")
        if (self.price_basis, self.corporate_action_policy) not in {
            ("raw", "none_in_window"), ("adjusted", "fully_adjusted"),
        }:
            raise ValueError(
                "raw OHLC requires corporate_action_policy='none_in_window'; adjusted OHLC "
                "requires corporate_action_policy='fully_adjusted' (including execution quotes)"
            )
        if self.dependence_group is not None:
            if not isinstance(self.dependence_group, str) or not self.dependence_group.strip():
                raise ValueError("dependence_group must be non-empty when provided")
            object.__setattr__(self, "dependence_group", self.dependence_group.strip())
        _timeframe_seconds(self.timeframe)


@dataclass(frozen=True, slots=True)
class RealExecutionData:
    candles: list[RealCandle]
    entry_spreads: list[float]
    exit_spreads: list[float]
    entry_mids: list[float]
    exit_mids: list[float]
    bar_duration_seconds: float
    execution_source: str


def _timeframe_seconds(timeframe: str) -> float:
    """Parse fixed-duration labels; calendar months and arbitrary display labels are unsafe."""
    if not isinstance(timeframe, str):
        raise ValueError("timeframe must be an explicit fixed duration such as '1m', '1h', or '1d'")
    text = timeframe.strip()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if len(text) < 2 or text[-1] not in units or not text[:-1].isdigit() or int(text[:-1]) <= 0:
        raise ValueError("timeframe must be an explicit fixed duration such as '1m', '1h', or '1d'")
    try:
        seconds = float(int(text[:-1]) * units[text[-1]])
    except OverflowError as exc:
        raise ValueError("timeframe duration is too large") from exc
    if not math.isfinite(seconds):
        raise ValueError("timeframe duration must be finite")
    return seconds


def _parse_timestamp(value: Any) -> datetime:
    text = str(value).strip()
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("real validation timestamps must be timezone-aware ISO-8601 values") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("real validation timestamps must include a timezone; integer bar labels are unsupported")
    return timestamp.astimezone(timezone.utc)


def _resolve_column(fieldnames: Sequence[str], candidates: Sequence[str], required: bool = True) -> str | None:
    lookup = {name.strip().lower(): name for name in fieldnames}
    if len(lookup) != len(fieldnames):
        raise ValueError("CSV header contains duplicate case-insensitive column names")
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"missing required CSV column; expected one of {candidates}")
    return None


def load_real_csv(
    path: Path,
    fallback_spread_bps: float,
    *,
    timeframe: str | None = None,
    price_basis: str = "raw",
) -> RealExecutionData:
    """Load regular UTC bars and exact opening/closing execution quote arrays.

    Requires timestamp and consistently raw or fully adjusted OHLC. Optional quotes must provide
    all of open_bid/open_ask/close_bid/close_ask; unsuffixed bid/ask is rejected because its timing
    is unknown. Without quotes, candle open/close and the frozen spread form an explicit modeled
    execution assumption. Adjusted OHLC uses adj_open/adj_high/adj_low/adj_close (spaces accepted),
    and quotes, if provided, must be on that same declared adjusted basis.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"real-market CSV does not exist: {path}")
    if isinstance(fallback_spread_bps, bool) or not isinstance(fallback_spread_bps, (int, float)):
        raise ValueError("fallback_spread_bps must be a finite non-negative real number")
    fallback_spread_bps = float(fallback_spread_bps)
    if not math.isfinite(fallback_spread_bps) or not 0.0 <= fallback_spread_bps < 20_000.0:
        raise ValueError("fallback_spread_bps must be finite and between 0 (inclusive) and 20000")
    if price_basis not in {"raw", "adjusted"}:
        raise ValueError("price_basis must be 'raw' or 'adjusted'")
    expected_duration = _timeframe_seconds(timeframe) if timeframe is not None else None
    candles: list[RealCandle] = []
    entry_spreads: list[float] = []
    exit_spreads: list[float] = []
    entry_mids: list[float] = []
    exit_mids: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        columns = {
            name: _resolve_column(
                reader.fieldnames, (name,) if price_basis == "raw" else (f"adj_{name}", f"adj {name}")
            )
            for name in ("open", "high", "low", "close")
        }
        volume_col = _resolve_column(reader.fieldnames, ("volume",), required=False)
        time_col = _resolve_column(reader.fieldnames, ("timestamp", "datetime", "date", "time"))
        if any(_resolve_column(reader.fieldnames, (name,), required=False) for name in ("bid", "ask")):
            raise ValueError("ambiguous bid/ask timing: use open_bid, open_ask, close_bid, close_ask")
        quotes = {
            name: _resolve_column(reader.fieldnames, (name,), required=False)
            for name in ("open_bid", "open_ask", "close_bid", "close_ask")
        }
        if any(quotes.values()) and not all(quotes.values()):
            raise ValueError("execution quotes require all four open_bid/open_ask/close_bid/close_ask columns")
        have_quotes = all(quotes.values())
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV row {index + 2}")
            def number(column: str, field_name: str) -> float:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid {field_name} at row {index + 2}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}: non-finite {field_name} at row {index + 2}")
                return value

            o, h, l, c = (number(columns[name], name) for name in ("open", "high", "low", "close"))
            if min(o, h, l, c) <= 0.0 or h < max(o, c, l) or l > min(o, c, h):
                raise ValueError(f"{path}: invalid OHLC relationship at row {index + 2}")
            volume = number(volume_col, "volume") if volume_col else 0.0
            if volume < 0.0:
                raise ValueError(f"{path}: negative volume at row {index + 2}")
            candles.append(RealCandle(index + 1, _parse_timestamp(row[time_col]), o, h, l, c, volume))
            for boundary, fallback_mid, mids, spreads in (
                ("open", o, entry_mids, entry_spreads), ("close", c, exit_mids, exit_spreads),
            ):
                if have_quotes:
                    bid = number(quotes[f"{boundary}_bid"], f"{boundary}_bid")
                    ask = number(quotes[f"{boundary}_ask"], f"{boundary}_ask")
                    if bid <= 0.0 or ask < bid:
                        raise ValueError(f"{path}: invalid {boundary} bid/ask at row {index + 2}")
                    mid = bid / 2.0 + ask / 2.0
                    mids.append(mid)
                    spreads.append((ask - bid) / mid)
                else:
                    mids.append(fallback_mid)
                    spreads.append(fallback_spread_bps / 10_000.0)
    if len(candles) < 2:
        raise ValueError(f"{path} must contain at least two market bars")
    durations = [(b.timestamp - a.timestamp).total_seconds() for a, b in zip(candles, candles[1:])]
    duration = durations[0]
    if duration <= 0.0 or any(not math.isclose(step, duration, rel_tol=0.0, abs_tol=1e-6) for step in durations):
        raise ValueError(f"{path}: timestamps must be strictly increasing and regularly spaced; missing/session bars are unsupported")
    if expected_duration is not None and not math.isclose(duration, expected_duration, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{path}: observed bar duration {duration:g}s does not match timeframe {timeframe!r}")
    return RealExecutionData(
        candles, entry_spreads, exit_spreads, entry_mids, exit_mids, duration,
        "explicit_open_close_quotes" if have_quotes else "modeled_ohlc_mid_with_frozen_spread",
    )


def _real_dataset_id(path: Path) -> str:
    return "REAL-" + _file_sha256(path)[:20]


def _real_metric_structure(path: Path) -> ScenarioSpec:
    return ScenarioSpec(structure_id=_real_dataset_id(path), label=path.stem, world_weights=(), overrides=())


def _real_dataset_clusters(
    datasets: Sequence[RealExecutionData], specs: Sequence[RealDatasetSpec], procedure: ProcedureConfig,
) -> list[int]:
    """Cluster overlapping market periods across assets; never count files as independent trials."""
    if len(datasets) != len(specs) or not datasets:
        raise ValueError("every real dataset requires exactly one explicit metadata specification")
    durations = {data.bar_duration_seconds for data in datasets}
    if len(durations) != 1:
        raise ValueError("different bar durations cannot be pooled in a real-market hypothesis test")
    parents = list(range(len(datasets)))
    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    duration = datasets[0].bar_duration_seconds
    embargo = (max(procedure.horizons) + procedure.null_embargo_bars) * duration
    intervals = [(data.candles[0].timestamp.timestamp(), data.candles[-1].timestamp.timestamp() + duration) for data in datasets]
    for i, (start, end) in enumerate(intervals):
        for j in range(i):
            other_start, other_end = intervals[j]
            overlap = start < other_end and other_start < end
            if overlap and specs[i].asset.casefold() == specs[j].asset.casefold():
                raise ValueError("overlapping windows of the same asset duplicate exposure and must not be supplied together")
            same_group = specs[i].dependence_group is not None and specs[i].dependence_group == specs[j].dependence_group
            if same_group or (start <= other_end + embargo and other_start <= end + embargo):
                parents[root(i)] = root(j)
    identities: dict[int, int] = {}
    return [identities.setdefault(root(i), len(identities)) for i in range(len(datasets))]


def _load_real_dataset_manifest(path: Path, inputs: Sequence[Path]) -> list[RealDatasetSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("datasets"), list):
        raise ValueError("dataset manifest must contain a 'datasets' array")
    by_path: dict[str, RealDatasetSpec] = {}
    for item in payload["datasets"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("each dataset manifest item must contain a CSV path")
        csv_path = Path(item["path"])
        if not csv_path.is_absolute():
            csv_path = Path(path).parent / csv_path
        key = str(csv_path.resolve()).casefold()
        if key in by_path:
            raise ValueError("dataset manifest contains a duplicate path")
        try:
            by_path[key] = RealDatasetSpec(**{name: value for name, value in item.items() if name != "path"})
        except TypeError as exc:
            raise ValueError(f"invalid or missing metadata for {csv_path}: {exc}") from exc
    keys = [str(Path(item).resolve()).casefold() for item in inputs]
    if set(keys) != set(by_path) or len(keys) != len(by_path):
        raise ValueError("dataset manifest must describe exactly the supplied input CSV files")
    return [by_path[key] for key in keys]


def _pooled_real_cluster_metrics(metrics: Sequence[RunMetric]) -> list[RunMetric]:
    """Collapse all files in a dependence cluster before independent-unit inference."""
    grouped: dict[tuple[int, str, int], list[RunMetric]] = {}
    for item in metrics:
        grouped.setdefault((item.seed, item.pattern_name, item.horizon), []).append(item)
    output: list[RunMetric] = []
    for (cluster, _, _), group in sorted(grouped.items()):
        events = sum(item.event_count for item in group)
        matched = sum(item.matched_null_event_count for item in group)
        def weighted(field: str, count_field: str) -> float:
            supported = [item for item in group if getattr(item, count_field) > 0]
            total = sum(getattr(item, count_field) for item in supported)
            return sum(getattr(item, field) * getattr(item, count_field) for item in supported) / total if total else float("nan")
        hits = sum(item.direction_correct_count for item in group)
        output.append(replace(
            group[0], structure_id="REAL-POOLED", structure_label="untouched_real_time_clusters",
            scenario_id=f"REAL-CLUSTER-{cluster}", run_id=f"REAL-CLUSTER-{cluster}",
            event_count=events, matched_null_event_count=matched,
            skipped_overlap_count=sum(item.skipped_overlap_count for item in group),
            untradeable_event_count=sum(item.untradeable_event_count for item in group),
            total_gross_return=sum(item.total_gross_return for item in group),
            total_transaction_cost=sum(item.total_transaction_cost for item in group),
            total_net_return=sum(item.total_net_return for item in group),
            mean_net_return=weighted("mean_net_return", "event_count") if events else 0.0,
            median_net_return=float("nan"),
            mean_effect_vs_matched=weighted("mean_effect_vs_matched", "matched_null_event_count"),
            mean_effect_vs_unconditional=weighted("mean_effect_vs_unconditional", "event_count"),
            mean_matched_null_net_return=weighted("mean_matched_null_net_return", "matched_null_event_count"),
            mean_unconditional_net_return=weighted("mean_unconditional_net_return", "event_count"),
            direction_correct_count=hits, direction_accuracy=hits / events if events else float("nan"),
            matched_null_direction_accuracy=weighted("matched_null_direction_accuracy", "matched_null_event_count"),
            matched_event_direction_accuracy=weighted("matched_event_direction_accuracy", "matched_null_event_count"),
        ))
    return output


def run_real_validation(
    inputs: Sequence[Path],
    selected_pattern_horizons: Sequence[tuple[str, int]],
    procedure: ProcedureConfig,
    *,
    timeframe: str | None = None,
    dataset_specs: Sequence[RealDatasetSpec] | None = None,
    git_commit: str | None = None,
    dataset_records: list[dict[str, Any]] | None = None,
) -> list[RunMetric]:
    if not inputs:
        raise ValueError("real validation requires at least one untouched dataset")
    paths = [Path(path) for path in inputs]
    resolved_paths = [str(path.resolve()).casefold() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("the same real-market file was supplied more than once")
    content_hashes = [_file_sha256(path) for path in paths]
    if len(set(content_hashes)) != len(content_hashes):
        raise ValueError("real validation inputs contain duplicate file contents")
    if dataset_specs is None or len(dataset_specs) != len(paths) or any(not isinstance(spec, RealDatasetSpec) for spec in dataset_specs):
        raise ValueError("real validation requires explicit RealDatasetSpec metadata for every CSV")
    hypotheses = list(selected_pattern_horizons)
    if not hypotheses or len(set(hypotheses)) != len(hypotheses):
        raise ValueError("real validation requires unique, non-empty frozen hypotheses")
    for pattern, horizon in hypotheses:
        if (
            pattern not in (*PATTERN_NAMES, ALL_PATTERNS_LABEL)
            or isinstance(horizon, bool) or not isinstance(horizon, int)
            or horizon not in procedure.horizons
        ):
            raise ValueError("real validation hypothesis is outside the frozen pattern/horizon procedure")
    if timeframe is not None and any(_timeframe_seconds(spec.timeframe) != _timeframe_seconds(timeframe) for spec in dataset_specs):
        raise ValueError("timeframe override conflicts with the dataset manifest")
    datasets = [
        load_real_csv(path, procedure.cost_model.real_spread_bps_fallback, timeframe=spec.timeframe, price_basis=spec.price_basis)
        for path, spec in zip(paths, dataset_specs)
    ]
    clusters = _real_dataset_clusters(datasets, dataset_specs, procedure)
    metrics: list[RunMetric] = []
    for path, data, spec, cluster, content_hash in zip(paths, datasets, dataset_specs, clusters, content_hashes):
        if _file_sha256(path) != content_hash:
            raise ValueError(f"real dataset changed while being loaded: {path}")
        dataset_id = "REAL-" + content_hash[:20]
        # Economic identity is independent of filename, asset display text, git and procedure provenance.
        economic_id = "REAL-ECON-" + _sha256_text([
            (c.timestamp.isoformat(), c.open, c.high, c.low, c.close, c.volume, em, xm, es, xs)
            for c, em, xm, es, xs in zip(data.candles, data.entry_mids, data.exit_mids, data.entry_spreads, data.exit_spreads)
        ])[:20]
        rid = "RR-" + hashlib.sha256((dataset_id + procedure_hash(procedure)).encode("utf-8")).hexdigest()[:16]
        context = RunContext(
            run_id=rid, market_world="real_market", seed=None, asset=spec.asset,
            timeframe=spec.timeframe, dataset_id=dataset_id, git_commit=git_commit,
        )
        result = detect_pattern_universe(data.candles, DEFAULT_PATTERN_CONFIG, context)
        structure = ScenarioSpec(dataset_id, path.stem, (), ())
        cache = _prepare_path_evaluation_cache(
            result=result, candles=data.candles, entry_spreads=data.entry_spreads,
            exit_spreads=data.exit_spreads, entry_mids=data.entry_mids, exit_mids=data.exit_mids,
            procedure=procedure,
        )
        for pattern_name, horizon in hypotheses:
            observations, skipped, untradeable = _evaluate_hypothesis_on_path(
                result=result, candles=data.candles, entry_spreads=data.entry_spreads,
                exit_spreads=data.exit_spreads, entry_mids=data.entry_mids, exit_mids=data.exit_mids,
                pattern_name=pattern_name, horizon=horizon, run_identity=economic_id,
                procedure=procedure, cache=cache,
            )
            metrics.append(_metric_from_observations(
                partition="real_validation", structure=structure, scenario_id=dataset_id,
                run_id=rid, seed=cluster, pattern_name=pattern_name, horizon=horizon,
                observations=observations, skipped_overlap=skipped, untradeable=untradeable,
            ))
        if dataset_records is not None:
            dataset_records.append({
                "path": str(path), "sha256": content_hash, "dataset_id": dataset_id,
                "economic_id": economic_id, **_jsonable(spec),
                "start": data.candles[0].timestamp.isoformat(),
                "last_bar_open": data.candles[-1].timestamp.isoformat(),
                "end_exclusive_unix_seconds": data.candles[-1].timestamp.timestamp() + data.bar_duration_seconds,
                "bar_duration_seconds": data.bar_duration_seconds, "bar_count": len(data.candles),
                "execution_source": data.execution_source, "dependence_cluster": cluster,
            })
    return metrics

# ---------------------------------------------------------------------------
# Manifest / selected hypothesis persistence
# ---------------------------------------------------------------------------


def _result_jsonable(value: Any) -> Any:
    """JSON-safe representation for statistical results; undefined statistics become null."""

    if isinstance(value, Mapping):
        return {str(k): _result_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_result_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return _jsonable(value)


def _selected_payload(
    selected: Sequence[Mapping[str, Any]],
    procedure: ProcedureConfig,
    *,
    synthetic_design_hash: str,
    status: str,
) -> dict[str, Any]:
    grouped_real: dict[tuple[str, int], set[str]] = {}
    for row in selected:
        key = (str(row["pattern_name"]), int(row["horizon"]))
        grouped_real.setdefault(key, set()).add(str(row["structure_id"]))
    transfer_hypotheses = [
        {
            "pattern_name": pattern,
            "horizon": horizon,
            "source_synthetic_structure_ids": sorted(grouped_real[(pattern, horizon)]),
        }
        for pattern, horizon in sorted(grouped_real)
    ]
    return {
        "status": status,
        "procedure_hash": procedure_hash(procedure),
        "synthetic_design_hash": synthetic_design_hash,
        "detector_contract": _detector_contract(),
        "procedure": _jsonable(procedure),
        "selected_synthetic_hypotheses": [_result_jsonable(dict(row)) for row in selected],
        "real_validation_estimand": (
            "unconditional real-market transfer by pattern and horizon; synthetic structure ids "
            "identify the discovery conditions but are not claimed to be observable real regimes"
        ),
        "real_transfer_hypotheses": transfer_hypotheses,
    }


def _procedure_from_manifest(manifest: Mapping[str, Any]) -> ProcedureConfig:
    payload = manifest.get("procedure")
    if not isinstance(payload, Mapping):
        raise ValueError("validated manifest does not contain the frozen procedure settings")
    cost_payload = payload.get("cost_model")
    if not isinstance(cost_payload, Mapping):
        raise ValueError("validated manifest is missing the frozen cost model")
    if set(payload) != {field.name for field in fields(ProcedureConfig)}:
        raise ValueError("frozen procedure must contain exactly the complete procedure fields")
    if set(cost_payload) != {field.name for field in fields(CostModel)}:
        raise ValueError("frozen cost model must contain exactly the complete cost fields")
    settings = dict(payload)
    settings["cost_model"] = CostModel(**cost_payload)
    return ProcedureConfig(**settings)


def _verify_frozen_manifest(manifest: Mapping[str, Any], procedure: ProcedureConfig) -> None:
    if manifest.get("status") != "synthetic_validation_complete":
        raise ValueError("validated manifest is incomplete or is not a completed validation artifact")
    design_hash = manifest.get("synthetic_design_hash")
    if not isinstance(design_hash, str) or len(design_hash) != 64:
        raise ValueError("validated manifest is missing its complete synthetic-design hash")
    expected_hash = procedure_hash(procedure)
    observed = manifest.get("procedure_hash")
    if observed != expected_hash:
        raise ValueError(
            "frozen procedure hash mismatch: real validation must use exactly the pre-registered "
            f"procedure (expected {observed!r}, current {expected_hash!r})"
        )
    current_contract = _detector_contract()
    stored_contract = manifest.get("detector_contract")
    if stored_contract != current_contract:
        raise ValueError("pattern detector version/config/file checksum differs from the frozen discovery detector")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_horizons(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("horizons must be comma-separated positive integers")
    return tuple(sorted(set(values)))


def _auto_workers(value: int) -> int:
    if value < 0:
        raise ValueError("workers cannot be negative")
    if value == 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return value


def _procedure_from_args(args: argparse.Namespace) -> ProcedureConfig:
    return ProcedureConfig(
        horizons=args.horizons,
        steps_per_candle=args.candle_steps,
        overlap_policy=args.overlap_policy,
        matched_nulls_per_event=args.matched_nulls,
        minimum_matched_nulls_per_event=args.min_matched_nulls_per_event,
        null_volatility_lookback=args.null_vol_lookback,
        null_embargo_bars=args.null_embargo,
        alpha=args.alpha,
        confidence_level=args.confidence,
        minimum_events=args.min_events,
        minimum_seeds_with_events=args.min_seeds,
        discovery_top_k=args.top_k,
        cost_model=CostModel(
            brokerage_bps_per_side=args.brokerage_bps,
            slippage_bps_per_side=args.slippage_bps,
            real_spread_bps_fallback=args.real_spread_bps,
        ),
    )


def _add_procedure_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 3, 5, 10, 20))
    parser.add_argument("--candle-steps", type=int, default=5)
    parser.add_argument("--overlap-policy", choices=("skip_new", "allow"), default="skip_new")
    parser.add_argument("--matched-nulls", type=int, default=20)
    parser.add_argument(
        "--min-matched-nulls-per-event",
        type=int,
        default=5,
        help=(
            "Minimum state-matched controls required before an event contributes to matched-null "
            "inference. Events below this remain valid strategy trades but are not mislabeled as "
            "matched comparisons."
        ),
    )
    parser.add_argument("--null-vol-lookback", type=int, default=20)
    parser.add_argument("--null-embargo", type=int, default=5)
    parser.add_argument("--brokerage-bps", type=float, default=1.0, help="Brokerage fee in bps per side.")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Additional adverse slippage in bps per side.")
    parser.add_argument(
        "--real-spread-bps",
        type=float,
        default=5.0,
        help="Modeled full spread for real CSVs without all four opening/closing quote columns.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-events", type=int, default=30)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and validate which synthetic market structures make causal chart "
            "patterns profitable after costs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Run discovery + held-out synthetic validation.")
    _add_procedure_args(synthetic)
    synthetic.add_argument("--output-dir", type=Path, default=Path("results/market_structure_experiment"))
    synthetic.add_argument("--steps", type=int, default=3_000)
    synthetic.add_argument("--discovery-seeds", type=int, default=20)
    synthetic.add_argument("--validation-seeds", type=int, default=20)
    synthetic.add_argument("--seed-start", type=int, default=10_000)
    synthetic.add_argument("--min-worlds", type=int, default=1)
    synthetic.add_argument("--max-worlds", type=int, default=len(WORLD_NAMES))
    synthetic.add_argument(
        "--weight-denominator",
        type=int,
        default=0,
        help=(
            "0 = equal-weight screening of every world subset.  Positive D enumerates every positive "
            "integer simplex composition with denominator D when D >= subset size; e.g. D=9 gives "
            "a much larger exhaustive weight grid."
        ),
    )
    synthetic.add_argument(
        "--grid-json",
        type=Path,
        default=None,
        help="JSON mapping dotted SyntheticMarketConfig paths to arrays of pre-registered values.",
    )
    synthetic.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="0 means no cap; useful only for smoke tests when >0.",
    )
    synthetic.add_argument("--workers", type=int, default=0, help="0 = all but one logical CPU; 1 = fully serial.")
    synthetic.add_argument("--git-commit", default=None)
    synthetic.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate and write the manifest without simulating.",
    )

    real = subparsers.add_parser("real", help="Apply the frozen selected procedure to untouched real OHLCV CSVs.")
    real.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Untouched OHLCV CSV; repeat for datasets described in --dataset-manifest.",
    )
    real.add_argument(
        "--validated-manifest",
        type=Path,
        required=True,
        help="validated_hypotheses.json written only after held-out synthetic validation.",
    )
    real.add_argument("--output-dir", type=Path, default=Path("results/real_validation"))
    real.add_argument(
        "--dataset-manifest", type=Path, required=True,
        help=("Frozen JSON datasets array: path, asset, venue, timeframe (e.g. 1h), price_basis "
              "(raw/adjusted), corporate_action_policy (none_in_window/fully_adjusted); "
              "optional session_policy=continuous and dependence_group."),
    )
    real.add_argument("--timeframe", default=None, help="Optional duration cross-check against every dataset specification.")
    real.add_argument("--git-commit", default=None)
    return parser.parse_args()


def _synthetic_main(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.discovery_seeds <= 0 or args.validation_seeds <= 0:
        raise ValueError("discovery-seeds and validation-seeds must be > 0")
    procedure = _procedure_from_args(args)
    base_config = SyntheticMarketConfig(steps=1, seed=0)
    parameter_grid = _load_grid_json(args.grid_json)
    scenarios = generate_scenarios(
        base_config,
        min_worlds=args.min_worlds,
        max_worlds=args.max_worlds,
        weight_denominator=args.weight_denominator,
        parameter_grid=parameter_grid,
        max_scenarios=args.max_scenarios,
    )
    if not scenarios:
        raise ValueError("scenario grid is empty")
    discovery_seeds = tuple(range(args.seed_start, args.seed_start + args.discovery_seeds))
    validation_start = args.seed_start + args.discovery_seeds
    validation_seeds = tuple(range(validation_start, validation_start + args.validation_seeds))
    if set(discovery_seeds).intersection(validation_seeds):
        raise AssertionError("discovery and validation seed sets must be disjoint")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_design = {
        "steps": args.steps,
        "scenario_count": len(scenarios),
        "min_worlds": args.min_worlds,
        "max_worlds": args.max_worlds,
        "weight_denominator": args.weight_denominator,
        "parameter_grid": _jsonable(parameter_grid),
        "discovery_seeds": list(discovery_seeds),
        "validation_seeds": list(validation_seeds),
        "world_names": list(WORLD_NAMES),
        "all_nonempty_equal_weight_subset_count": (2 ** len(WORLD_NAMES)) - 1,
    }
    scenario_payloads = [_jsonable(scenario) for scenario in scenarios]
    synthetic_design_hash = _sha256_text(
        {"synthetic_design": synthetic_design, "scenarios": scenario_payloads}
    )
    manifest = {
        **_procedure_payload(procedure),
        "procedure_hash": procedure_hash(procedure),
        "synthetic_design_hash": synthetic_design_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_design": synthetic_design,
        "scenarios": scenario_payloads,
    }
    _write_json(manifest, output_dir / "procedure_manifest.json")
    print(f"Procedure hash: {manifest['procedure_hash']}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Discovery seeds: {len(discovery_seeds)} | Validation seeds: {len(validation_seeds)}")
    print(f"Scenario-seed simulations: {len(scenarios) * (len(discovery_seeds) + len(validation_seeds))}")
    if args.dry_run:
        print("Dry run complete; no simulations executed.")
        return

    # Replace any successful manifest left in a reused directory before doing new work.  A failed
    # rerun can therefore never masquerade as the current design's completed validation.
    _write_json(
        {
            "status": "incomplete",
            "procedure_hash": manifest["procedure_hash"],
            "synthetic_design_hash": synthetic_design_hash,
        },
        output_dir / "validated_hypotheses.json",
    )

    workers = _auto_workers(args.workers)
    discovery_metrics = run_synthetic_partition(
        scenarios,
        discovery_seeds,
        "discovery",
        steps=args.steps,
        base_config=base_config,
        procedure=procedure,
        workers=workers,
        git_commit=args.git_commit,
    )
    _write_run_metrics(discovery_metrics, output_dir / "discovery_run_metrics.csv")
    discovery_rows = aggregate_metrics(discovery_metrics, procedure)
    selected = select_discovery_hypotheses(discovery_rows, procedure)
    _write_dict_csv(discovery_rows, output_dir / "discovery_hypotheses.csv")
    _write_json(
        _selected_payload(
            selected,
            procedure,
            synthetic_design_hash=synthetic_design_hash,
            status="discovery_complete",
        ),
        output_dir / "discovery_selected_hypotheses.json",
    )

    validation_metrics = run_synthetic_partition(
        scenarios,
        validation_seeds,
        "validation",
        steps=args.steps,
        base_config=base_config,
        procedure=procedure,
        workers=workers,
        git_commit=args.git_commit,
    )
    _write_run_metrics(validation_metrics, output_dir / "validation_run_metrics.csv")
    validation_rows_all = aggregate_metrics(validation_metrics, procedure)
    validation_rows = validate_selected_hypotheses(validation_rows_all, selected, procedure)
    _write_dict_csv(validation_rows_all, output_dir / "validation_all_hypotheses.csv")
    _write_dict_csv(validation_rows, output_dir / "validation_selected_hypotheses.csv")
    rankings = structure_rankings(validation_rows)
    _write_dict_csv(rankings, output_dir / "validated_structure_rankings.csv")

    validated = [row for row in validation_rows if row.get("validated")]
    validated_payload = _selected_payload(
        validated,
        procedure,
        synthetic_design_hash=synthetic_design_hash,
        status="synthetic_validation_complete",
    )
    validated_payload["source_discovery_hypothesis_count"] = len(selected)
    validated_payload["synthetic_validation_correction"] = "Holm family-wise error rate"
    _write_json(validated_payload, output_dir / "validated_hypotheses.json")
    completion_artifacts = (
        "procedure_manifest.json",
        "discovery_run_metrics.csv",
        "discovery_hypotheses.csv",
        "discovery_selected_hypotheses.json",
        "validation_run_metrics.csv",
        "validation_all_hypotheses.csv",
        "validation_selected_hypotheses.csv",
        "validated_structure_rankings.csv",
        "validated_hypotheses.json",
    )
    _write_json(
        {
            "status": "complete",
            "procedure_hash": manifest["procedure_hash"],
            "synthetic_design_hash": synthetic_design_hash,
            "artifact_sha256": {
                name: _file_sha256(output_dir / name) for name in completion_artifacts
            },
        },
        output_dir / "run_completion_manifest.json",
    )
    print(f"Discovery hypotheses passing FDR + minimum sample filters: {len(selected)}")
    print(f"Held-out hypotheses passing Holm validation: {len(validated)}")
    if rankings:
        best = rankings[0]
        print("Best held-out market structure:")
        print(f"  rank={best['rank']} structure={best['structure_label']}")
        print(f"  pattern={best['best_pattern']} horizon={best['best_horizon']}")
        print(f"  mean net return/trade={best['best_mean_net_return_per_trade']:.8g}")
        print(f"  mean effect vs matched null={best['best_mean_effect_vs_matched']:.8g}")
        print(f"  Holm p={best['best_validation_holm_p_value']:.6g}")
    else:
        print("No market structure passed the complete held-out validation criteria.")
    print(f"Results written to {output_dir}")


def _real_main(args: argparse.Namespace) -> None:
    validated_bytes = args.validated_manifest.read_bytes()
    validated_manifest_sha256 = hashlib.sha256(validated_bytes).hexdigest()
    validated_manifest = json.loads(validated_bytes)
    if not isinstance(validated_manifest, Mapping):
        raise ValueError("validated manifest must be a JSON object")
    procedure = _procedure_from_manifest(validated_manifest)
    _verify_frozen_manifest(validated_manifest, procedure)
    selected_items = validated_manifest.get("real_transfer_hypotheses", [])
    if not isinstance(selected_items, list) or any(not isinstance(item, Mapping) for item in selected_items):
        raise ValueError("frozen real_transfer_hypotheses must be an array of hypothesis objects")
    pattern_horizons = [(item.get("pattern_name"), item.get("horizon")) for item in selected_items]
    if not pattern_horizons:
        raise ValueError("validated manifest contains no held-out hypotheses for real validation")
    for pattern, horizon in pattern_horizons:
        if pattern not in (*PATTERN_NAMES, ALL_PATTERNS_LABEL):
            raise ValueError(f"unknown frozen pattern {pattern!r}")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in procedure.horizons:
            raise ValueError(f"frozen real hypothesis horizon {horizon} is not in the procedure")

    dataset_manifest_sha256 = _file_sha256(args.dataset_manifest)
    dataset_specs = _load_real_dataset_manifest(args.dataset_manifest, args.input)
    dataset_records: list[dict[str, Any]] = []
    metrics = run_real_validation(
        args.input,
        pattern_horizons,
        procedure,
        timeframe=args.timeframe,
        dataset_specs=dataset_specs,
        dataset_records=dataset_records,
        git_commit=args.git_commit,
    )
    if _file_sha256(args.dataset_manifest) != dataset_manifest_sha256:
        raise ValueError("dataset manifest changed during real validation")
    if _file_sha256(args.validated_manifest) != validated_manifest_sha256:
        raise ValueError("validated hypothesis manifest changed during real validation")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metrics(metrics, output_dir / "real_run_metrics.csv")

    # Per-file results are descriptive. Pool temporal dependence clusters as single observations,
    # including zero-trade files; merely adding correlated asset CSVs never increases sample size.
    rows = aggregate_metrics(metrics, procedure)
    pooled_metrics = _pooled_real_cluster_metrics(metrics)
    pooled_rows = aggregate_metrics(pooled_metrics, procedure)
    if pooled_rows:
        _holm_adjust(pooled_rows, "raw_p_value_primary_edge", "real_holm_p_value")
        for row in pooled_rows:
            row["real_validated"] = bool(
                row["event_count"] >= procedure.minimum_events
                and row["matched_null_event_count"] >= procedure.minimum_events
                and row["seed_count_with_events"] >= procedure.minimum_seeds_with_events
                and row["seed_count_with_matched_nulls"] >= procedure.minimum_seeds_with_events
                and row["mean_net_return_per_trade"] > 0.0
                and math.isfinite(float(row["mean_effect_vs_matched"]))
                and row["mean_effect_vs_matched"] > 0.0
                and row["real_holm_p_value"] <= procedure.alpha
            )
    _write_dict_csv(rows, output_dir / "real_per_dataset_hypotheses.csv")
    _write_dict_csv(pooled_rows, output_dir / "real_pooled_validation.csv")
    _write_json(
        {
            "procedure_hash": procedure_hash(procedure),
            "detector_contract": _detector_contract(),
            "status": "complete",
            "real_validation_estimand": validated_manifest["real_validation_estimand"],
            "source_transfer_hypotheses": selected_items,
            "inference_unit": "temporal_dependence_cluster",
            "dependence_cluster_count": len({record["dependence_cluster"] for record in dataset_records}),
            "cluster_policy": "merge overlapping periods across all assets plus horizon/embargo buffer and declared dependence groups",
            "cluster_embargo_bars": max(procedure.horizons) + procedure.null_embargo_bars,
            "input_files": dataset_records,
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "validated_manifest": str(args.validated_manifest),
            "validated_manifest_sha256": validated_manifest_sha256,
        },
        output_dir / "real_validation_manifest.json",
    )
    print(f"Real validation complete across {len(args.input)} untouched dataset(s).")
    print(f"Results written to {output_dir}")


def main() -> None:
    args = parse_args()
    # A failed rerun must not leave a prior successful result looking current.
    status_path = args.output_dir / "experiment_run_status.json"
    status = {"status": "RUNNING", "command": args.command, "confirmatory_results_valid": False}
    _write_json(status, status_path)
    try:
        if args.command == "synthetic":
            _synthetic_main(args)
        elif args.command == "real":
            _real_main(args)
        else:  # pragma: no cover
            raise RuntimeError(f"unknown command {args.command!r}")
    except Exception as exc:
        _write_json({**status, "status": "FAILED", "error": str(exc)}, status_path)
        raise
    dry_run = bool(getattr(args, "dry_run", False))
    _write_json({**status, "status": "DRY_RUN_COMPLETE" if dry_run else "COMPLETE",
                 "confirmatory_results_valid": not dry_run}, status_path)


__all__ = [
    "EXPERIMENT_RUNNER_VERSION",
    "CostModel",
    "ProcedureConfig",
    "ScenarioSpec",
    "TradeObservation",
    "RunMetric",
    "RealDatasetSpec",
    "RealExecutionData",
    "net_trade_return",
    "synthetic_execution_arrays",
    "trade_direction",
    "procedure_hash",
    "generate_scenarios",
    "run_synthetic_partition",
    "aggregate_metrics",
    "select_discovery_hypotheses",
    "validate_selected_hypotheses",
    "structure_rankings",
    "load_real_csv",
    "run_real_validation",
]


if __name__ == "__main__":
    main()
