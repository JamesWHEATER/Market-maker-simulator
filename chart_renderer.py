from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SnapshotLike(Protocol):
    step: int
    trade_price: float
    order_size: int


@dataclass(frozen=True)
class PriceCandle:
    candle_number: int
    start_step: int
    end_step: int
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def is_up(self) -> bool:
        return self.close >= self.open


def build_candles(
    history: list[SnapshotLike],
    steps_per_candle: int,
) -> list[PriceCandle]:
    if steps_per_candle <= 0:
        raise ValueError("steps_per_candle must be greater than zero")

    candles: list[PriceCandle] = []
    for start_index in range(0, len(history), steps_per_candle):
        chunk = history[start_index : start_index + steps_per_candle]
        if not chunk:
            continue

        trade_prices = [snapshot.trade_price for snapshot in chunk]
        candles.append(
            PriceCandle(
                candle_number=len(candles) + 1,
                start_step=chunk[0].step,
                end_step=chunk[-1].step,
                open=trade_prices[0],
                high=max(trade_prices),
                low=min(trade_prices),
                close=trade_prices[-1],
                volume=sum(snapshot.order_size for snapshot in chunk),
            )
        )

    return candles


def write_interactive_candlestick_html(
    candles: list[PriceCandle],
    output_path: Path,
    visible_candles: int = 160,
    title: str = "Market Candlestick Chart",
) -> None:
    if not candles:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visible_candles = max(1, min(visible_candles, len(candles)))
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
        }
        for candle in candles
    ]
    candles_json = json.dumps(candle_data, separators=(",", ":"))
    template = r"""<!doctype html>
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
  --control: #334155;
}

* {
  box-sizing: border-box;
}

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

.topbar,
.controls,
.chart-shell {
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

h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0;
}

.meta {
  color: var(--muted);
  font-size: 13px;
  white-space: nowrap;
}

.controls {
  display: grid;
  grid-template-columns: auto auto minmax(160px, 1fr) auto auto auto auto auto auto;
  gap: 10px;
  align-items: center;
  padding: 10px 12px;
}

button,
input {
  font: inherit;
}

button {
  min-width: 44px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: #ffffff;
  color: var(--control);
  cursor: pointer;
}

button:hover {
  background: #f1f5f9;
}

input[type="range"] {
  width: 100%;
}

.visible-control {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
}

.visible-control input {
  width: 76px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 0 8px;
}

.scale-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}

.scale-controls button {
  min-width: 34px;
}

.scale-controls .auto-scale {
  min-width: 54px;
}

.tool-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tool-controls button {
  min-width: 52px;
}

button.active {
  border-color: #2563eb;
  background: #eff6ff;
  color: #1d4ed8;
}

#scaleStatus {
  color: var(--muted);
  font-size: 13px;
  min-width: 92px;
  text-align: left;
}

#windowStatus {
  color: var(--muted);
  font-size: 13px;
  text-align: right;
  white-space: nowrap;
}

.chart-shell {
  position: relative;
  min-height: 520px;
  overflow: hidden;
}

canvas {
  display: block;
  width: 100%;
  height: min(72vh, 720px);
  min-height: 520px;
  cursor: move;
}

canvas.dragging {
  cursor: move;
}

.tooltip {
  position: absolute;
  display: none;
  pointer-events: none;
  min-width: 190px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.96);
  color: var(--text);
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.14);
  font-size: 12px;
  line-height: 1.45;
}

.tooltip strong {
  display: block;
  margin-bottom: 3px;
}

@media (max-width: 760px) {
  .app {
    padding: 10px;
  }

  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .controls {
    grid-template-columns: auto auto 1fr;
  }

  .visible-control,
  .scale-controls,
  .tool-controls,
  #jumpStart,
  #jumpEnd {
    display: none;
  }

  #windowStatus {
    grid-column: 1 / -1;
    text-align: left;
  }
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
    <input id="positionRange" type="range" min="0" value="0" step="1">
    <button id="stepForward" type="button" title="Next window">&gt;</button>
    <button id="jumpEnd" type="button" title="Latest candles">&gt;|</button>
    <label class="visible-control">Visible
      <input id="visibleInput" type="number" min="20" step="10">
    </label>
    <div class="scale-controls" aria-label="Price scale controls">
      <button id="scaleOut" type="button" title="Zoom price scale out">-</button>
      <button id="scaleAuto" class="auto-scale" type="button" title="Reset price scale">Auto</button>
      <button id="scaleIn" type="button" title="Zoom price scale in">+</button>
      <output id="scaleStatus"></output>
    </div>
    <div class="tool-controls" aria-label="Drawing tools">
      <button id="drawLineToggle" type="button" title="Draw blue trend lines">Draw</button>
      <button id="clearLines" type="button" title="Clear drawn lines">Clear</button>
    </div>
    <output id="windowStatus"></output>
  </section>
  <section class="chart-shell">
    <canvas id="chartCanvas"></canvas>
    <div class="tooltip" id="tooltip"></div>
  </section>
</main>
<script>
const candles = __CANDLES_JSON__;
const initialVisibleCandles = __VISIBLE_CANDLES__;

const canvas = document.getElementById("chartCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const range = document.getElementById("positionRange");
const visibleInput = document.getElementById("visibleInput");
const windowStatus = document.getElementById("windowStatus");
const scaleStatus = document.getElementById("scaleStatus");
const chartMeta = document.getElementById("chartMeta");
const drawLineToggle = document.getElementById("drawLineToggle");
const clearLines = document.getElementById("clearLines");

const colors = {
  text: "#0f172a",
  muted: "#64748b",
  grid: "#e2e8f0",
  axis: "#94a3b8",
  green: "#16a34a",
  red: "#dc2626",
  anchor: "#475569",
  hover: "#2563eb",
  draw: "#2563eb"
};

const scaleAnchorPrice = 100;
let visibleCount = Math.min(initialVisibleCandles, candles.length);
let startIndex = Math.max(0, candles.length - visibleCount);
let hoverIndex = null;
let isDragging = false;
let dragMode = null;
let lastDragX = 0;
let lastDragY = 0;
let lastLayout = null;
let manualPriceScale = false;
let priceZoom = 1;
let priceOffset = 0;
let isDrawMode = false;
let drawnLines = [];
let draftLine = null;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function priceText(value) {
  return value.toFixed(4);
}

function maxStartIndex() {
  return Math.max(0, candles.length - visibleCount);
}

function resetPriceScale() {
  manualPriceScale = false;
  priceZoom = 1;
  priceOffset = 0;
  updateControls();
  draw();
}

function zoomPriceScale(factor) {
  manualPriceScale = true;
  priceZoom = Math.max(1e-9, Math.min(priceZoom * factor, 1e9));
  updateControls();
  draw();
}

function compressPriceScaleFromDrag(deltaPixels) {
  const factor = Math.exp(-deltaPixels * 0.006);
  zoomPriceScale(factor);
}

function panChartVertically(deltaPixels) {
  if (!lastLayout) {
    return;
  }
  manualPriceScale = true;
  priceOffset += deltaPixels / lastLayout.plotHeight * (lastLayout.maxPrice - lastLayout.minPrice);
  updateControls();
  draw();
}

function setStartIndex(value) {
  startIndex = clamp(Math.round(value), 0, maxStartIndex());
  updateControls();
  draw();
}

function setVisibleCount(value) {
  visibleCount = clamp(Math.round(value), 20, candles.length);
  startIndex = clamp(startIndex, 0, maxStartIndex());
  updateControls();
  draw();
}

function updateControls() {
  range.max = String(maxStartIndex());
  range.value = String(startIndex);
  visibleInput.max = String(candles.length);
  visibleInput.value = String(visibleCount);

  const first = candles[startIndex];
  const last = candles[Math.min(candles.length - 1, startIndex + visibleCount - 1)];
  chartMeta.textContent = `${candles.length} candles, ${first.startStep}-${last.endStep} visible`;
  windowStatus.value = `Candles ${first.number}-${last.number} | steps ${first.startStep}-${last.endStep}`;
  scaleStatus.value = manualPriceScale ? `${priceZoom.toPrecision(3)}x` : "Auto";
  drawLineToggle.classList.toggle("active", isDrawMode);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function yFor(price, minPrice, maxPrice, top, plotHeight) {
  return top + (maxPrice - price) / (maxPrice - minPrice) * plotHeight;
}

function priceForY(y) {
  return lastLayout.maxPrice
    - (y - lastLayout.margin.top) / lastLayout.plotHeight
    * (lastLayout.maxPrice - lastLayout.minPrice);
}

function screenXForPosition(position) {
  return lastLayout.margin.left + (position - startIndex) * lastLayout.slotWidth;
}

function screenYForPrice(price) {
  return yFor(
    price,
    lastLayout.minPrice,
    lastLayout.maxPrice,
    lastLayout.margin.top,
    lastLayout.plotHeight
  );
}

function pointerPoint(event) {
  if (!lastLayout) {
    return null;
  }
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const plotRight = rect.width - lastLayout.margin.right;
  const plotBottom = rect.height - lastLayout.margin.bottom;

  if (
    x < lastLayout.margin.left
    || x > plotRight
    || y < lastLayout.margin.top
    || y > plotBottom
  ) {
    return null;
  }

  return {
    candlePosition: startIndex + (x - lastLayout.margin.left) / lastLayout.slotWidth,
    price: priceForY(y),
  };
}

function drawAnnotationLine(line, isDraft = false) {
  const x1 = screenXForPosition(line.start.candlePosition);
  const y1 = screenYForPrice(line.start.price);
  const x2 = screenXForPosition(line.end.candlePosition);
  const y2 = screenYForPrice(line.end.price);

  ctx.save();
  ctx.beginPath();
  ctx.rect(
    lastLayout.margin.left,
    lastLayout.margin.top,
    lastLayout.plotWidth,
    lastLayout.plotHeight
  );
  ctx.clip();
  ctx.strokeStyle = colors.draw;
  ctx.lineWidth = isDraft ? 1.25 : 1;
  ctx.globalAlpha = isDraft ? 0.72 : 0.95;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const margin = { left: 72, right: 24, top: 24, bottom: 44 };
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const plotHeight = Math.max(1, height - margin.top - margin.bottom);
  const endIndex = Math.min(candles.length, startIndex + visibleCount);
  const visible = candles.slice(startIndex, endIndex);
  const lows = visible.map((candle) => candle.low);
  const highs = visible.map((candle) => candle.high);
  const dataMinPrice = Math.min(...lows);
  const dataMaxPrice = Math.max(...highs);
  const rawRange = Math.max(dataMaxPrice - dataMinPrice, 0.01);
  const padding = rawRange * 0.08;
  const autoMinPrice = dataMinPrice - padding;
  const autoMaxPrice = dataMaxPrice + padding;
  const autoRange = autoMaxPrice - autoMinPrice;
  let minPrice = autoMinPrice;
  let maxPrice = autoMaxPrice;

  if (manualPriceScale) {
    const scaledRange = autoRange / priceZoom;
    if (priceZoom === 1) {
      minPrice = autoMinPrice;
      maxPrice = autoMaxPrice;
    } else {
      const anchorRatio = clamp(
        (autoMaxPrice - scaleAnchorPrice) / autoRange,
        0.08,
        0.92
      );
      maxPrice = scaleAnchorPrice + anchorRatio * scaledRange;
      minPrice = maxPrice - scaledRange;
    }
    minPrice += priceOffset;
    maxPrice += priceOffset;
  }

  const slotWidth = plotWidth / visible.length;
  const bodyWidth = clamp(slotWidth * 0.56, 3, 12);
  const wickWidth = clamp(slotWidth * 0.08, 0.45, 1.1);
  lastLayout = { margin, plotWidth, plotHeight, slotWidth, minPrice, maxPrice, visible };

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.font = "12px Arial";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 5; i += 1) {
    const price = minPrice + (maxPrice - minPrice) * i / 5;
    const y = yFor(price, minPrice, maxPrice, margin.top, plotHeight);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "right";
    ctx.fillText(price.toFixed(2), margin.left - 10, y);
  }

  ctx.strokeStyle = colors.axis;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, height - margin.bottom);
  ctx.lineTo(width - margin.right, height - margin.bottom);
  ctx.stroke();

  const labelCount = Math.min(6, visible.length);
  if (labelCount > 1) {
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < labelCount; i += 1) {
      const localIndex = Math.round(i * (visible.length - 1) / (labelCount - 1));
      const candle = visible[localIndex];
      const x = margin.left + slotWidth * localIndex + slotWidth / 2;
      ctx.fillText(String(candle.startStep), x, height - margin.bottom + 12);
    }
  }

  visible.forEach((candle, localIndex) => {
    const globalIndex = startIndex + localIndex;
    const x = margin.left + slotWidth * localIndex + slotWidth / 2;
    const openY = yFor(candle.open, minPrice, maxPrice, margin.top, plotHeight);
    const closeY = yFor(candle.close, minPrice, maxPrice, margin.top, plotHeight);
    const highY = yFor(candle.high, minPrice, maxPrice, margin.top, plotHeight);
    const lowY = yFor(candle.low, minPrice, maxPrice, margin.top, plotHeight);
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

    if (globalIndex === hoverIndex) {
      ctx.strokeStyle = colors.hover;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - bodyWidth / 2 - 2, bodyTop - 2, bodyWidth + 4, bodyHeight + 4);
    }
  });

  drawnLines.forEach((line) => drawAnnotationLine(line));
  if (draftLine) {
    drawAnnotationLine(draftLine, true);
  }
}

function candleIndexAt(clientX) {
  if (!lastLayout) {
    return null;
  }
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const localIndex = Math.floor((x - lastLayout.margin.left) / lastLayout.slotWidth);
  if (localIndex < 0 || localIndex >= lastLayout.visible.length) {
    return null;
  }
  return startIndex + localIndex;
}

function moveTooltip(event, candle) {
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX - canvas.getBoundingClientRect().left + 14}px`;
  tooltip.style.top = `${event.clientY - canvas.getBoundingClientRect().top + 14}px`;
  tooltip.innerHTML = `
    <strong>Candle ${candle.number}</strong>
    steps ${candle.startStep}-${candle.endStep}<br>
    O ${priceText(candle.open)} &nbsp; H ${priceText(candle.high)}<br>
    L ${priceText(candle.low)} &nbsp; C ${priceText(candle.close)}<br>
    Vol ${candle.volume}
  `;
}

function panBy(delta) {
  setStartIndex(startIndex + delta);
}

range.addEventListener("input", () => setStartIndex(Number(range.value)));
visibleInput.addEventListener("change", () => setVisibleCount(Number(visibleInput.value)));
drawLineToggle.addEventListener("click", () => {
  isDrawMode = !isDrawMode;
  draftLine = null;
  tooltip.style.display = "none";
  updateControls();
  draw();
});
clearLines.addEventListener("click", () => {
  drawnLines = [];
  draftLine = null;
  draw();
});
document.getElementById("scaleOut").addEventListener("click", () => zoomPriceScale(1 / 1.25));
document.getElementById("scaleAuto").addEventListener("click", resetPriceScale);
document.getElementById("scaleIn").addEventListener("click", () => zoomPriceScale(1.25));
document.getElementById("stepBack").addEventListener("click", () => panBy(-Math.max(1, Math.floor(visibleCount / 2))));
document.getElementById("stepForward").addEventListener("click", () => panBy(Math.max(1, Math.floor(visibleCount / 2))));
document.getElementById("jumpStart").addEventListener("click", () => setStartIndex(0));
document.getElementById("jumpEnd").addEventListener("click", () => setStartIndex(maxStartIndex()));

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const isPriceScaleGesture = event.shiftKey || (lastLayout && x <= lastLayout.margin.left);
  if (isPriceScaleGesture) {
    zoomPriceScale(event.deltaY < 0 ? 1.15 : 1 / 1.15);
    return;
  }

  const direction = Math.sign(event.deltaY || event.deltaX);
  panBy(direction * Math.max(1, Math.floor(visibleCount * 0.08)));
}, { passive: false });

canvas.addEventListener("mousedown", (event) => {
  isDragging = true;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  if (lastLayout && x <= lastLayout.margin.left) {
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
  canvas.classList.add("dragging");
});

window.addEventListener("mouseup", () => {
  if (dragMode === "draw-line" && draftLine) {
    const candleDistance = Math.abs(
      draftLine.end.candlePosition - draftLine.start.candlePosition
    );
    const priceDistance = Math.abs(draftLine.end.price - draftLine.start.price);
    if (candleDistance > 0.1 || priceDistance > 0.0001) {
      drawnLines.push(draftLine);
    }
    draftLine = null;
    draw();
  }
  isDragging = false;
  dragMode = null;
  canvas.classList.remove("dragging");
});

canvas.addEventListener("mousemove", (event) => {
  if (isDragging && lastLayout) {
    if (dragMode === "price-scale") {
      const deltaPixels = event.clientY - lastDragY;
      if (Math.abs(deltaPixels) >= 1) {
        compressPriceScaleFromDrag(deltaPixels);
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

      const verticalDeltaPixels = event.clientY - lastDragY;
      if (Math.abs(verticalDeltaPixels) >= 1) {
        panChartVertically(verticalDeltaPixels);
        lastDragY = event.clientY;
      }
    }
  }

  if (lastLayout && !isDragging) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    canvas.style.cursor = x <= lastLayout.margin.left
      ? "ns-resize"
      : isDrawMode ? "crosshair" : "move";
  }

  if (isDrawMode) {
    hoverIndex = null;
    tooltip.style.display = "none";
    draw();
    return;
  }

  const index = candleIndexAt(event.clientX);
  if (index === null) {
    hoverIndex = null;
    tooltip.style.display = "none";
    draw();
    return;
  }

  hoverIndex = index;
  moveTooltip(event, candles[index]);
  draw();
});

canvas.addEventListener("mouseleave", () => {
  hoverIndex = null;
  tooltip.style.display = "none";
  canvas.style.cursor = "move";
  draw();
});

canvas.addEventListener("dblclick", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  if (lastLayout && x <= lastLayout.margin.left) {
    resetPriceScale();
  }
});

window.addEventListener("resize", resizeCanvas);
updateControls();
resizeCanvas();
</script>
</body>
</html>
"""
    safe_title = html.escape(str(title), quote=True)
    rendered_html = (
        template.replace("__CANDLES_JSON__", candles_json)
        .replace("__VISIBLE_CANDLES__", str(visible_candles))
        .replace("__CHART_TITLE__", safe_title)
    )
    output_path.write_text(rendered_html, encoding="utf-8")
