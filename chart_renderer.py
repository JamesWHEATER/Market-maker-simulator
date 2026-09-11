"""Validated candlestick construction and a dependency-free interactive HTML renderer.

The module deliberately keeps data construction separate from visualisation.  ``PriceCandle``
objects satisfy the OHLCV interface consumed by ``pattern_detector.py`` and carry monotonically
increasing synthetic step labels for reproducible research runs.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Protocol, Sequence


CANDLE_BUILDER_VERSION = "2.1.0"
RENDERER_VERSION = "2.1.0"


class SnapshotLike(Protocol):
    """Minimum simulator observation required by :func:`build_candles`.

    Rich simulator snapshots may additionally expose ``open``, ``high``, ``low``, ``close`` and
    ``volume``.  When present, those intrastep OHLCV fields are preserved rather than collapsing
    a multi-trade step to its last transaction.
    """

    step: int
    trade_price: float
    order_size: int


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric value, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {numeric!r}")
    return numeric


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    """Return a canonical Python ``int`` without silently accepting floats.

    ``numbers.Integral`` deliberately accepts NumPy integer scalars while rejecting values such as
    ``1.0`` that merely compare equal to an integer.  This keeps candle chronology/volume metadata
    exact and avoids platform-dependent coercions in provenance-sensitive research runs.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    numeric = int(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {numeric}")
    return numeric


def _nonnegative_int(value: object, name: str) -> int:
    return _integer(value, name, minimum=0)


@dataclass(frozen=True, slots=True)
class PriceCandle:
    """One validated positive-price OHLCV candle.

    ``start_step`` is intentionally exposed because the pattern detector uses it as a chronology
    label.  ``is_complete`` is provenance only; it is ignored by the detector.
    """

    candle_number: int
    start_step: int
    end_step: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    is_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "candle_number", _integer(self.candle_number, "candle_number", minimum=1))
        for name in ("start_step", "end_step"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("candle step interval is invalid")

        values = {
            name: _finite_float(getattr(self, name), name)
            for name in ("open", "high", "low", "close")
        }
        if any(value <= 0.0 for value in values.values()):
            raise ValueError("OHLC prices must be strictly positive")
        scale = max(values.values())
        tolerance = max(scale * 1e-12, math.ulp(scale))
        if values["high"] + tolerance < values["low"]:
            raise ValueError("high cannot be below low")
        if values["open"] > values["high"] + tolerance or values["close"] > values["high"] + tolerance:
            raise ValueError("open/close cannot exceed high")
        if values["open"] < values["low"] - tolerance or values["close"] < values["low"] - tolerance:
            raise ValueError("open/close cannot be below low")
        for name, value in values.items():
            object.__setattr__(self, name, value)

        object.__setattr__(self, "volume", _nonnegative_int(self.volume, "volume"))
        if not isinstance(self.is_complete, bool):
            raise ValueError("is_complete must be boolean")

    @property
    def is_up(self) -> bool:
        return self.close >= self.open


@dataclass(frozen=True, slots=True)
class _SnapshotBar:
    step: int
    open: float
    high: float
    low: float
    close: float
    volume: int


def _snapshot_to_bar(snapshot: SnapshotLike, index: int) -> _SnapshotBar:
    step = getattr(snapshot, "step", None)
    step = _integer(step, f"history[{index}].step", minimum=0)

    fallback = _finite_float(getattr(snapshot, "trade_price", None), f"history[{index}].trade_price")
    raw_open = getattr(snapshot, "open", fallback)
    raw_high = getattr(snapshot, "high", fallback)
    raw_low = getattr(snapshot, "low", fallback)
    raw_close = getattr(snapshot, "close", fallback)
    open_price = _finite_float(raw_open, f"history[{index}].open")
    high = _finite_float(raw_high, f"history[{index}].high")
    low = _finite_float(raw_low, f"history[{index}].low")
    close = _finite_float(raw_close, f"history[{index}].close")
    if min(open_price, high, low, close) <= 0.0:
        raise ValueError(f"history[{index}] contains a non-positive price")

    scale = max(open_price, high, low, close)
    tolerance = max(scale * 1e-12, math.ulp(scale))
    if high + tolerance < max(open_price, close, low):
        raise ValueError(f"history[{index}] high is inconsistent with OHLC")
    if low - tolerance > min(open_price, close, high):
        raise ValueError(f"history[{index}] low is inconsistent with OHLC")

    raw_volume = getattr(snapshot, "volume", getattr(snapshot, "order_size", None))
    volume = _nonnegative_int(raw_volume, f"history[{index}].volume")
    return _SnapshotBar(step, open_price, high, low, close, volume)


def build_candles(
    history: Sequence[SnapshotLike],
    steps_per_candle: int,
    *,
    include_partial: bool = False,
    require_contiguous_steps: bool = True,
) -> list[PriceCandle]:
    """Aggregate simulator observations into fixed-width OHLCV candles.

    By default an incomplete final candle is dropped.  This is the safer backtest convention:
    every detector observation then spans the same number of simulator steps.  Set
    ``include_partial=True`` for exploratory visualisation when retaining the tail matters more
    than fixed-duration comparability.
    """

    steps_per_candle = _integer(steps_per_candle, "steps_per_candle", minimum=1)
    if not isinstance(include_partial, bool) or not isinstance(require_contiguous_steps, bool):
        raise ValueError("include_partial and require_contiguous_steps must be boolean")

    bars = [_snapshot_to_bar(snapshot, index) for index, snapshot in enumerate(history)]
    for index in range(1, len(bars)):
        previous = bars[index - 1].step
        current = bars[index].step
        if current <= previous:
            raise ValueError("history steps must be strictly increasing")
        if require_contiguous_steps and current != previous + 1:
            raise ValueError(
                f"history steps must be contiguous; observed {previous} followed by {current}"
            )

    candles: list[PriceCandle] = []
    for start in range(0, len(bars), steps_per_candle):
        chunk = bars[start : start + steps_per_candle]
        if not chunk:
            continue
        complete = len(chunk) == steps_per_candle
        if not complete and not include_partial:
            break
        candles.append(
            PriceCandle(
                candle_number=len(candles) + 1,
                start_step=chunk[0].step,
                end_step=chunk[-1].step,
                open=chunk[0].open,
                high=max(bar.high for bar in chunk),
                low=min(bar.low for bar in chunk),
                close=chunk[-1].close,
                volume=sum(bar.volume for bar in chunk),
                is_complete=complete,
            )
        )
    return candles


def _validated_candle(candle: object, index: int) -> PriceCandle:
    if isinstance(candle, PriceCandle):
        return candle
    try:
        complete = getattr(candle, "is_complete", True)
        if not isinstance(complete, bool):
            raise ValueError("is_complete must be boolean")
        return PriceCandle(
            candle_number=getattr(candle, "candle_number"),
            start_step=getattr(candle, "start_step"),
            end_step=getattr(candle, "end_step"),
            open=getattr(candle, "open"),
            high=getattr(candle, "high"),
            low=getattr(candle, "low"),
            close=getattr(candle, "close"),
            volume=getattr(candle, "volume"),
            is_complete=complete,
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid candle at index {index}") from exc


def _validate_candle_sequence(candles: Sequence[PriceCandle]) -> list[PriceCandle]:
    validated: list[PriceCandle] = []
    previous_number = 0
    previous_end: int | None = None
    for index, candle in enumerate(candles):
        current = _validated_candle(candle, index)
        if current.candle_number <= previous_number:
            raise ValueError("candle_number values must be strictly increasing")
        if previous_end is not None and current.start_step <= previous_end:
            raise ValueError("candle step intervals must be strictly increasing and non-overlapping")
        previous_number = current.candle_number
        previous_end = current.end_step
        validated.append(current)
    return validated


def write_interactive_candlestick_html(
    candles: Sequence[PriceCandle],
    output_path: Path,
    visible_candles: int = 160,
    title: str = "Market Candlestick Chart",
) -> None:
    """Write a self-contained interactive candlestick + volume chart.

    The renderer uses no CDN or JavaScript dependency, so a stored experiment remains viewable
    offline.  Price-scale zoom is anchored to the current data rather than to a hard-coded quote
    level, and users can toggle linear/log display without altering the underlying candle data.
    """

    validated_candles = _validate_candle_sequence(candles)
    visible_candles = _integer(visible_candles, "visible_candles", minimum=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candle_data = [
        {
            "number": candle.candle_number,
            "startStep": candle.start_step,
            "endStep": candle.end_step,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "complete": candle.is_complete,
        }
        for candle in validated_candles
    ]
    candles_json = json.dumps(candle_data, separators=(",", ":"), allow_nan=False)
    initial_visible = min(visible_candles, len(validated_candles)) if validated_candles else 0

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__CHART_TITLE__</title>
<style>
:root {
  color-scheme: light;
  --bg: #f8fafc;
  --panel: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --line: #cbd5e1;
  --grid: #e2e8f0;
  --green: #16a34a;
  --red: #dc2626;
  --blue: #2563eb;
  --control: #334155;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, sans-serif;
}
.app {
  display: grid;
  grid-template-rows: auto auto minmax(420px, 1fr);
  gap: 12px;
  min-height: 100vh;
  padding: 18px;
}
.topbar, .controls, .chart-shell {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
}
.topbar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 18px;
  padding: 12px 14px;
}
h1 { margin: 0; font-size: 18px; font-weight: 700; }
.meta { color: var(--muted); font-size: 13px; white-space: nowrap; }
.controls {
  display: grid;
  grid-template-columns: auto auto minmax(140px, 1fr) auto auto auto auto auto auto auto;
  gap: 8px;
  align-items: center;
  padding: 10px 12px;
}
button, input { font: inherit; }
button {
  min-width: 42px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: #fff;
  color: var(--control);
  cursor: pointer;
}
button:hover:not(:disabled) { background: #f1f5f9; }
button:disabled { opacity: .45; cursor: default; }
button.active { border-color: var(--blue); background: #eff6ff; color: #1d4ed8; }
input[type="range"] { width: 100%; }
.visible-control, .scale-controls, .tool-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
}
.visible-control input {
  width: 74px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 0 8px;
}
.scale-controls button { min-width: 34px; }
.scale-controls .auto-scale { min-width: 52px; }
.tool-controls button { min-width: 50px; }
#scaleStatus { color: var(--muted); font-size: 13px; min-width: 78px; }
#windowStatus { color: var(--muted); font-size: 13px; text-align: right; white-space: nowrap; }
.chart-shell { position: relative; min-height: 520px; overflow: hidden; }
canvas {
  display: block;
  width: 100%;
  height: min(74vh, 760px);
  min-height: 520px;
  cursor: move;
  touch-action: none;
}
.tooltip {
  position: absolute;
  display: none;
  pointer-events: none;
  min-width: 205px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255,255,255,.97);
  color: var(--text);
  box-shadow: 0 10px 30px rgba(15,23,42,.14);
  font-size: 12px;
  line-height: 1.45;
  z-index: 2;
}
.tooltip strong { display: block; margin-bottom: 3px; }
@media (max-width: 800px) {
  .app { padding: 10px; }
  .topbar { align-items: flex-start; flex-direction: column; }
  .controls { grid-template-columns: auto auto 1fr auto; }
  .visible-control, .scale-controls, .tool-controls, #jumpStart, #jumpEnd { display: none; }
  #windowStatus { grid-column: 1 / -1; text-align: left; }
}
</style>
</head>
<body>
<main class="app">
  <header class="topbar">
    <h1>__CHART_TITLE__</h1>
    <div class="meta" id="chartMeta"></div>
  </header>
  <section class="controls" aria-label="Chart controls">
    <button id="jumpStart" type="button" title="First candles">|&lt;</button>
    <button id="stepBack" type="button" title="Previous window">&lt;</button>
    <input id="positionRange" type="range" min="0" value="0" step="1" aria-label="Visible window position">
    <button id="stepForward" type="button" title="Next window">&gt;</button>
    <button id="jumpEnd" type="button" title="Latest candles">&gt;|</button>
    <label class="visible-control">Visible
      <input id="visibleInput" type="number" min="1" step="10">
    </label>
    <div class="scale-controls" aria-label="Price scale controls">
      <button id="scaleOut" type="button" title="Zoom price scale out">−</button>
      <button id="scaleAuto" class="auto-scale" type="button" title="Reset price scale">Auto</button>
      <button id="scaleIn" type="button" title="Zoom price scale in">+</button>
      <button id="logToggle" type="button" title="Toggle logarithmic price scale">Log</button>
      <output id="scaleStatus"></output>
    </div>
    <div class="tool-controls" aria-label="Drawing tools">
      <button id="drawLineToggle" type="button" title="Draw trend lines">Draw</button>
      <button id="clearLines" type="button" title="Clear drawn lines">Clear</button>
    </div>
    <output id="windowStatus"></output>
  </section>
  <section class="chart-shell" id="chartShell">
    <canvas id="chartCanvas"></canvas>
    <div class="tooltip" id="tooltip"></div>
  </section>
</main>
<script>
"use strict";
const candles = __CANDLES_JSON__;
const initialVisibleCandles = __VISIBLE_CANDLES__;

const canvas = document.getElementById("chartCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const chartShell = document.getElementById("chartShell");
const range = document.getElementById("positionRange");
const visibleInput = document.getElementById("visibleInput");
const windowStatus = document.getElementById("windowStatus");
const scaleStatus = document.getElementById("scaleStatus");
const chartMeta = document.getElementById("chartMeta");
const drawLineToggle = document.getElementById("drawLineToggle");
const clearLines = document.getElementById("clearLines");
const logToggle = document.getElementById("logToggle");

const colors = {
  text: "#0f172a", muted: "#64748b", grid: "#e2e8f0", axis: "#94a3b8",
  green: "#16a34a", red: "#dc2626", hover: "#2563eb", draw: "#2563eb",
  volume: "#94a3b8"
};

let visibleCount = candles.length ? Math.max(1, Math.min(initialVisibleCandles, candles.length)) : 0;
let startIndex = candles.length ? Math.max(0, candles.length - visibleCount) : 0;
let hoverIndex = null;
let isDragging = false;
let dragMode = null;
let lastDragX = 0;
let lastDragY = 0;
let lastLayout = null;
let manualBounds = null;
let logScale = false;
let isDrawMode = false;
let drawnLines = [];
let draftLine = null;

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function transformPrice(price) { return logScale ? Math.log(price) : price; }
function inversePrice(value) { return logScale ? Math.exp(value) : value; }

function formatPrice(value) {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute === 0) return "0";
  if (absolute >= 1000000 || absolute < 0.0001) return value.toExponential(4);
  if (absolute >= 1000) return value.toFixed(2);
  if (absolute >= 1) return value.toFixed(4);
  return value.toPrecision(5);
}

function formatVolume(value) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

function maxStartIndex() { return Math.max(0, candles.length - visibleCount); }

function setStartIndex(value) {
  if (!candles.length) return;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return;
  startIndex = clamp(Math.round(numeric), 0, maxStartIndex());
  updateControls();
  draw();
}

function setVisibleCount(value) {
  if (!candles.length) return;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) { updateControls(); return; }
  const minimum = Math.min(20, candles.length);
  visibleCount = clamp(Math.round(numeric), minimum, candles.length);
  startIndex = clamp(startIndex, 0, maxStartIndex());
  manualBounds = null;
  updateControls();
  draw();
}

function resetPriceScale() {
  manualBounds = null;
  updateControls();
  draw();
}

function zoomPriceScale(factor, anchorY = null) {
  if (!lastLayout || !Number.isFinite(factor) || factor <= 0) return;
  const oldMin = lastLayout.minT;
  const oldMax = lastLayout.maxT;
  const oldRange = oldMax - oldMin;
  if (!(oldRange > 0)) return;
  const anchorT = anchorY === null
    ? (oldMin + oldMax) / 2
    : lastLayout.maxT - (anchorY - lastLayout.margin.top) / lastLayout.priceHeight * oldRange;
  const anchorRatio = clamp((anchorT - oldMin) / oldRange, 0, 1);
  const newRange = clamp(oldRange / factor, oldRange * 1e-6, oldRange * 1e6);
  const minT = anchorT - anchorRatio * newRange;
  manualBounds = { minT, maxT: minT + newRange };
  updateControls();
  draw();
}

function panChartVertically(deltaPixels) {
  if (!lastLayout || !Number.isFinite(deltaPixels)) return;
  const span = lastLayout.maxT - lastLayout.minT;
  const shift = deltaPixels / lastLayout.priceHeight * span;
  manualBounds = {
    minT: lastLayout.minT + shift,
    maxT: lastLayout.maxT + shift
  };
  updateControls();
  draw();
}

function updateControls() {
  const controls = document.querySelectorAll("button, input");
  controls.forEach((element) => { element.disabled = candles.length === 0; });
  if (!candles.length) {
    chartMeta.textContent = "No candle data";
    windowStatus.value = "";
    scaleStatus.value = logScale ? "Log" : "Linear";
    return;
  }
  range.max = String(maxStartIndex());
  range.value = String(startIndex);
  visibleInput.min = String(Math.min(20, candles.length));
  visibleInput.max = String(candles.length);
  visibleInput.value = String(visibleCount);
  const first = candles[startIndex];
  const last = candles[Math.min(candles.length - 1, startIndex + visibleCount - 1)];
  chartMeta.textContent = `${candles.length} candles · steps ${first.startStep}–${last.endStep}`;
  windowStatus.value = `Candles ${first.number}–${last.number} | steps ${first.startStep}–${last.endStep}`;
  scaleStatus.value = `${logScale ? "Log" : "Linear"}${manualBounds ? " · manual" : " · auto"}`;
  drawLineToggle.classList.toggle("active", isDrawMode);
  logToggle.classList.toggle("active", logScale);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function yForTransformed(value, minT, maxT, top, height) {
  return top + (maxT - value) / (maxT - minT) * height;
}

function yForPrice(price) {
  return yForTransformed(
    transformPrice(price), lastLayout.minT, lastLayout.maxT,
    lastLayout.margin.top, lastLayout.priceHeight
  );
}

function priceForY(y) {
  const transformed = lastLayout.maxT
    - (y - lastLayout.margin.top) / lastLayout.priceHeight
    * (lastLayout.maxT - lastLayout.minT);
  return inversePrice(transformed);
}

function screenXForPosition(position) {
  return lastLayout.margin.left + (position - startIndex) * lastLayout.slotWidth + lastLayout.slotWidth / 2;
}

function pointerPoint(event) {
  if (!lastLayout) return null;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const right = rect.width - lastLayout.margin.right;
  const bottom = lastLayout.margin.top + lastLayout.priceHeight;
  if (x < lastLayout.margin.left || x > right || y < lastLayout.margin.top || y > bottom) return null;
  return {
    candlePosition: startIndex + (x - lastLayout.margin.left) / lastLayout.slotWidth - 0.5,
    price: priceForY(y)
  };
}

function drawAnnotationLine(line, isDraft = false) {
  if (!lastLayout) return;
  const x1 = screenXForPosition(line.start.candlePosition);
  const y1 = yForPrice(line.start.price);
  const x2 = screenXForPosition(line.end.candlePosition);
  const y2 = yForPrice(line.end.price);
  ctx.save();
  ctx.beginPath();
  ctx.rect(lastLayout.margin.left, lastLayout.margin.top, lastLayout.plotWidth, lastLayout.priceHeight);
  ctx.clip();
  ctx.strokeStyle = colors.draw;
  ctx.lineWidth = isDraft ? 1.4 : 1.1;
  ctx.globalAlpha = isDraft ? 0.72 : 0.95;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function drawEmpty(width, height) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = colors.muted;
  ctx.font = "14px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("No candle data", width / 2, height / 2);
  lastLayout = null;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  if (!candles.length || visibleCount <= 0) { drawEmpty(width, height); return; }

  const margin = { left: 78, right: 24, top: 24, bottom: 44 };
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const totalHeight = Math.max(1, height - margin.top - margin.bottom);
  const priceHeight = Math.max(120, totalHeight * 0.76);
  const volumeGap = 14;
  const volumeTop = margin.top + priceHeight + volumeGap;
  const volumeHeight = Math.max(30, height - margin.bottom - volumeTop);
  const endIndex = Math.min(candles.length, startIndex + visibleCount);
  const visible = candles.slice(startIndex, endIndex);
  if (!visible.length) { drawEmpty(width, height); return; }

  let dataMin = Infinity;
  let dataMax = -Infinity;
  let maxVolume = 1;
  visible.forEach((candle) => {
    dataMin = Math.min(dataMin, transformPrice(candle.low));
    dataMax = Math.max(dataMax, transformPrice(candle.high));
    maxVolume = Math.max(maxVolume, candle.volume);
  });
  const linearMagnitude = Math.max(Math.abs(dataMax), Math.abs(dataMin), Number.MIN_VALUE);
  const scaleFloor = logScale ? 1e-8 : linearMagnitude * 1e-8;
  const rawRange = Math.max(dataMax - dataMin, scaleFloor);
  const padding = rawRange * 0.08;
  const autoMinT = dataMin - padding;
  const autoMaxT = dataMax + padding;
  const minT = manualBounds ? manualBounds.minT : autoMinT;
  const maxT = manualBounds ? manualBounds.maxT : autoMaxT;
  const slotWidth = plotWidth / visible.length;
  const bodyWidth = clamp(slotWidth * 0.58, 2, 12);
  const wickWidth = clamp(slotWidth * 0.08, 0.5, 1.2);

  lastLayout = {
    margin, plotWidth, priceHeight, volumeTop, volumeHeight,
    slotWidth, minT, maxT, visible
  };

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.font = "12px Arial";
  ctx.textBaseline = "middle";

  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const transformed = minT + (maxT - minT) * i / 5;
    const y = yForTransformed(transformed, minT, maxT, margin.top, priceHeight);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "right";
    ctx.fillText(formatPrice(inversePrice(transformed)), margin.left - 10, y);
  }

  ctx.strokeStyle = colors.axis;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + priceHeight);
  ctx.lineTo(width - margin.right, margin.top + priceHeight);
  ctx.moveTo(margin.left, volumeTop);
  ctx.lineTo(margin.left, volumeTop + volumeHeight);
  ctx.lineTo(width - margin.right, volumeTop + volumeHeight);
  ctx.stroke();

  ctx.fillStyle = colors.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(formatVolume(maxVolume), margin.left - 10, volumeTop);
  ctx.textBaseline = "bottom";
  ctx.fillText("0", margin.left - 10, volumeTop + volumeHeight);

  const labelCount = Math.min(6, visible.length);
  if (labelCount > 0) {
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < labelCount; i += 1) {
      const localIndex = labelCount === 1 ? 0 : Math.round(i * (visible.length - 1) / (labelCount - 1));
      const candle = visible[localIndex];
      const x = margin.left + slotWidth * localIndex + slotWidth / 2;
      ctx.fillText(String(candle.startStep), x, height - margin.bottom + 12);
    }
  }

  visible.forEach((candle, localIndex) => {
    const globalIndex = startIndex + localIndex;
    const x = margin.left + slotWidth * localIndex + slotWidth / 2;
    const openY = yForPrice(candle.open);
    const closeY = yForPrice(candle.close);
    const highY = yForPrice(candle.high);
    const lowY = yForPrice(candle.low);
    const bodyTop = Math.min(openY, closeY);
    const bodyHeight = Math.max(1.5, Math.abs(closeY - openY));
    const color = candle.close >= candle.open ? colors.green : colors.red;

    ctx.strokeStyle = color;
    ctx.lineWidth = wickWidth;
    ctx.beginPath();
    ctx.moveTo(x, highY);
    ctx.lineTo(x, lowY);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);

    const volHeight = candle.volume / maxVolume * volumeHeight;
    ctx.globalAlpha = 0.45;
    ctx.fillStyle = color;
    ctx.fillRect(x - bodyWidth / 2, volumeTop + volumeHeight - volHeight, bodyWidth, volHeight);
    ctx.globalAlpha = 1;

    if (globalIndex === hoverIndex) {
      ctx.strokeStyle = colors.hover;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - bodyWidth / 2 - 2, bodyTop - 2, bodyWidth + 4, bodyHeight + 4);
    }
  });

  drawnLines.forEach((line) => drawAnnotationLine(line));
  if (draftLine) drawAnnotationLine(draftLine, true);
}

function candleIndexAt(clientX) {
  if (!lastLayout) return null;
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const localIndex = Math.floor((x - lastLayout.margin.left) / lastLayout.slotWidth);
  if (localIndex < 0 || localIndex >= lastLayout.visible.length) return null;
  return startIndex + localIndex;
}

function moveTooltip(event, candle) {
  tooltip.style.display = "block";
  tooltip.innerHTML = `
    <strong>Candle ${candle.number}${candle.complete ? "" : " · partial"}</strong>
    steps ${candle.startStep}–${candle.endStep}<br>
    O ${formatPrice(candle.open)} &nbsp; H ${formatPrice(candle.high)}<br>
    L ${formatPrice(candle.low)} &nbsp; C ${formatPrice(candle.close)}<br>
    Vol ${formatVolume(candle.volume)}
  `;
  const canvasRect = canvas.getBoundingClientRect();
  const shellRect = chartShell.getBoundingClientRect();
  const desiredLeft = event.clientX - canvasRect.left + 14;
  const desiredTop = event.clientY - canvasRect.top + 14;
  const tooltipWidth = tooltip.offsetWidth || 210;
  const tooltipHeight = tooltip.offsetHeight || 90;
  tooltip.style.left = `${clamp(desiredLeft, 4, shellRect.width - tooltipWidth - 4)}px`;
  tooltip.style.top = `${clamp(desiredTop, 4, shellRect.height - tooltipHeight - 4)}px`;
}

function panBy(delta) { setStartIndex(startIndex + delta); }

range.addEventListener("input", () => setStartIndex(range.value));
visibleInput.addEventListener("change", () => setVisibleCount(visibleInput.value));
drawLineToggle.addEventListener("click", () => {
  isDrawMode = !isDrawMode;
  draftLine = null;
  tooltip.style.display = "none";
  updateControls();
  draw();
});
clearLines.addEventListener("click", () => { drawnLines = []; draftLine = null; draw(); });
document.getElementById("scaleOut").addEventListener("click", () => zoomPriceScale(1 / 1.25));
document.getElementById("scaleAuto").addEventListener("click", resetPriceScale);
document.getElementById("scaleIn").addEventListener("click", () => zoomPriceScale(1.25));
logToggle.addEventListener("click", () => { logScale = !logScale; manualBounds = null; updateControls(); draw(); });
document.getElementById("stepBack").addEventListener("click", () => panBy(-Math.max(1, Math.floor(visibleCount / 2))));
document.getElementById("stepForward").addEventListener("click", () => panBy(Math.max(1, Math.floor(visibleCount / 2))));
document.getElementById("jumpStart").addEventListener("click", () => setStartIndex(0));
document.getElementById("jumpEnd").addEventListener("click", () => setStartIndex(maxStartIndex()));

canvas.addEventListener("wheel", (event) => {
  if (!lastLayout) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const priceScaleGesture = event.shiftKey || x <= lastLayout.margin.left;
  if (priceScaleGesture) {
    const anchorY = clamp(y, lastLayout.margin.top, lastLayout.margin.top + lastLayout.priceHeight);
    zoomPriceScale(event.deltaY < 0 ? 1.15 : 1 / 1.15, anchorY);
    return;
  }
  const direction = Math.sign(event.deltaY || event.deltaX);
  if (direction !== 0) panBy(direction * Math.max(1, Math.floor(visibleCount * 0.08)));
}, { passive: false });

canvas.addEventListener("mousedown", (event) => {
  if (!lastLayout) return;
  isDragging = true;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  if (x <= lastLayout.margin.left) {
    dragMode = "price-scale";
  } else if (isDrawMode) {
    const point = pointerPoint(event);
    if (point) {
      dragMode = "draw-line";
      draftLine = { start: point, end: point };
      tooltip.style.display = "none";
    } else {
      dragMode = "chart";
    }
  } else {
    dragMode = "chart";
  }
  lastDragX = event.clientX;
  lastDragY = event.clientY;
});

window.addEventListener("mouseup", () => {
  if (dragMode === "draw-line" && draftLine) {
    const candleDistance = Math.abs(draftLine.end.candlePosition - draftLine.start.candlePosition);
    const priceDistance = Math.abs(draftLine.end.price - draftLine.start.price);
    if (candleDistance > 0.1 || priceDistance > Math.max(1e-12, Math.abs(draftLine.start.price) * 1e-8)) {
      drawnLines.push(draftLine);
    }
    draftLine = null;
    draw();
  }
  isDragging = false;
  dragMode = null;
});

canvas.addEventListener("mousemove", (event) => {
  if (isDragging && lastLayout) {
    if (dragMode === "price-scale") {
      const delta = event.clientY - lastDragY;
      if (Math.abs(delta) >= 1) {
        zoomPriceScale(Math.exp(-delta * 0.006), event.clientY - canvas.getBoundingClientRect().top);
        lastDragY = event.clientY;
      }
    } else if (dragMode === "draw-line") {
      const point = pointerPoint(event);
      if (point && draftLine) {
        draftLine = { ...draftLine, end: point };
        hoverIndex = null;
        tooltip.style.display = "none";
        draw();
      }
    } else {
      const deltaPixels = lastDragX - event.clientX;
      const deltaCandles = Math.trunc(deltaPixels / lastLayout.slotWidth);
      if (deltaCandles !== 0) {
        panBy(deltaCandles);
        lastDragX = event.clientX;
      }
      const verticalDelta = event.clientY - lastDragY;
      if (Math.abs(verticalDelta) >= 1) {
        panChartVertically(verticalDelta);
        lastDragY = event.clientY;
      }
    }
  }

  if (lastLayout && !isDragging) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    canvas.style.cursor = x <= lastLayout.margin.left ? "ns-resize" : (isDrawMode ? "crosshair" : "move");
  }
  if (isDrawMode) {
    hoverIndex = null;
    tooltip.style.display = "none";
    return;
  }
  const index = candleIndexAt(event.clientX);
  if (index === null) {
    if (hoverIndex !== null) { hoverIndex = null; tooltip.style.display = "none"; draw(); }
    return;
  }
  if (hoverIndex !== index) { hoverIndex = index; draw(); }
  moveTooltip(event, candles[index]);
});

canvas.addEventListener("mouseleave", () => {
  hoverIndex = null;
  tooltip.style.display = "none";
  canvas.style.cursor = "move";
  draw();
});
canvas.addEventListener("dblclick", (event) => {
  if (!lastLayout) return;
  const x = event.clientX - canvas.getBoundingClientRect().left;
  if (x <= lastLayout.margin.left) resetPriceScale();
});

window.addEventListener("resize", resizeCanvas);
if (window.ResizeObserver) new ResizeObserver(resizeCanvas).observe(chartShell);
updateControls();
resizeCanvas();
</script>
</body>
</html>
'''

    safe_title = html.escape(str(title), quote=True)
    rendered_html = (
        template.replace("__CANDLES_JSON__", candles_json)
        .replace("__VISIBLE_CANDLES__", str(initial_visible))
        .replace("__CHART_TITLE__", safe_title)
    )
    output_path.write_text(rendered_html, encoding="utf-8")


__all__ = ["CANDLE_BUILDER_VERSION", "RENDERER_VERSION", "SnapshotLike", "PriceCandle", "build_candles", "write_interactive_candlestick_html"]
