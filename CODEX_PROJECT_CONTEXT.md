# Market Maker Simulator — Codex Project Context

## Project goal

This project is a quantitative-finance research prototype designed to test whether chart patterns contain predictive or profitable information.

The research plan is:

1. Create controlled synthetic market data.
2. Build an automatic chart-pattern recognition algorithm.
3. Test detected patterns in the synthetic market first, where the underlying process is known.
4. Measure whether any apparent profitability is greater than what we would expect from chance.
5. Later, apply the same pattern-recognition and backtesting pipeline to real financial-market data.
6. Add an AI / machine-learning component after the classical baseline is working, with the exact AI direction to be finalized after the project meeting.

The important principle is to keep the simulator, chart generation, pattern detection, backtesting, and future AI model modular so that the same detector can be evaluated across synthetic and real markets.

---

## What has already been built

### 1. `market_maker_simulator.py`

This file contains both the synthetic random-market simulation and the market-maker model.

Current components:

- `RandomMarketConfig`
  - number of simulation steps
  - random seed
  - initial fair value
  - fair-value step volatility
  - base bid/ask spread
  - inventory skew
  - minimum and maximum order size

- `MarketMaker`
  - maintains cash and inventory
  - creates bid and ask quotes around a reservation price
  - shifts the reservation price according to inventory
  - processes market taker buys and sells
  - calculates mark-to-market P&L

- `simulate_random_market(...)`
  - evolves fair value using a Gaussian random walk
  - randomly generates buy or sell market orders
  - randomly generates order sizes
  - executes the order against the market maker's bid or ask
  - records a `MarketSnapshot` after every step

- Each `MarketSnapshot` contains:
  - step
  - fair value
  - bid
  - ask
  - taker side
  - order size
  - trade price
  - maker inventory
  - maker cash
  - maker P&L

- `summarize(...)`
  - calculates basic simulation statistics such as:
    - final fair value
    - final trade price
    - final inventory
    - final P&L
    - average spread
    - average absolute inventory
    - realized log-return volatility
    - maximum P&L drawdown
    - buy volume
    - sell volume
    - total volume

- `write_history_csv(...)`
  - exports the simulated market history to CSV.

- The simulator then sends its history to the chart renderer to create candlestick data and an interactive chart.

This random market is intended to be the first null / baseline world. Because the fair value follows a random walk and order direction is random, persistent profitable chart patterns should not systematically exist. This makes it useful for checking whether a pattern detector or trading strategy is simply finding patterns by chance.

---

### 2. `chart_renderer.py`

This file converts simulation output into candlestick data and creates an interactive HTML chart.

Current components:

- `PriceCandle`
  - candle number
  - start step
  - end step
  - open
  - high
  - low
  - close
  - volume

- `build_candles(...)`
  - groups a chosen number of simulation steps into each candle
  - calculates OHLC values from trade prices
  - calculates candle volume from order sizes

- `write_interactive_candlestick_html(...)`
  - generates an interactive candlestick chart in HTML
  - supports:
    - horizontal navigation through candles
    - changing the number of visible candles
    - price-axis zooming
    - vertical panning
    - tooltips with OHLC and volume information
    - manually drawing and clearing trend lines

The chart renderer is separate from the simulator so future synthetic worlds and real-market data can reuse the same visualization layer.

---

## Research papers currently being used

The project is informed by several papers already included in the project materials.

### Lo, Mamaysky & Wang (2000) — *Foundations of Technical Analysis*

The paper develops a systematic, automatic approach to technical pattern recognition using nonparametric kernel regression. Its core motivation is directly relevant to this project: turn visually subjective chart patterns into reproducible quantitative rules and then test whether returns conditional on those patterns differ from unconditional returns.

### Tsinaslanidis & Guijarro (2021) — *What makes trading strategies based on chart pattern recognition profitable?*

This paper uses generic pattern recognition rather than restricting itself only to named patterns. It searches historical price sequences for similar patterns using Dynamic Time Warping and the UCR Suite, labels historical matches according to subsequent price behavior, and generates long / short / neutral recommendations based on consensus among similar historical patterns.

Important ideas from this paper for the future architecture:

- separate training and testing data
- define a query pattern
- find similar historical reference patterns
- examine what happened after those references
- turn the historical outcomes into a trading decision
- evaluate the resulting trades
- control for data snooping and transaction costs

### Park & Irwin (2007) — *What Do We Know About the Profitability of Technical Analysis?*

This review is important for the experimental design. It highlights major problems in technical-analysis research, especially:

- data snooping
- ex-post selection of trading rules
- insufficient out-of-sample testing
- transaction costs
- risk adjustment

The project should explicitly guard against these issues.

### Deep Learning and Technical Analysis in Cryptocurrency Market

This paper is useful for the later AI stage. It compares machine-learning and deep-learning approaches using candlestick/OHLC information and technical indicators, including models such as logistic regression, gradient boosting, MLP, GRU, LSTM, and CNN.

The AI portion of this project should be added only after there is a clear non-AI baseline so its incremental value can be measured.

---

# Immediate next step: build the pattern-detection algorithm

The next file should be a separate module, for example:

`pattern_detector.py`

Do **not** tightly couple pattern detection to the simulator or chart renderer.

The detector should consume candle data and return structured pattern detections that can later be passed to a backtester or AI model.

## First objective

Build a classical, deterministic baseline pattern detector before adding AI.

The first version does not need to recognize every chart pattern. It should establish the full pipeline with one or a small number of patterns.

A good first candidate is **Head and Shoulders / Inverse Head and Shoulders**, because it is a well-known chart pattern and is also represented in the academic literature.

The detector should be designed so additional patterns can be added later.

---

## Suggested architecture

Create something similar to:

```python
@dataclass(frozen=True)
class PatternDetection:
    pattern_name: str
    start_index: int
    end_index: int
    signal: str
    confidence: float
    metadata: dict
```

Possible public API:

```python
def detect_patterns(candles: list[PriceCandle]) -> list[PatternDetection]:
    ...
```

Internally, keep individual pattern detectors separate, for example:

```python
def detect_head_and_shoulders(...):
    ...

def detect_inverse_head_and_shoulders(...):
    ...
```

Later we should be able to add:

- double top
- double bottom
- triangles
- flags
- support/resistance breakouts
- generic similarity-based patterns
- AI-based pattern classification

without rewriting the simulator.

---

## Important design requirements

### 1. Avoid look-ahead bias

A pattern may only use information that would have been available at the time of detection.

For example, if a Head and Shoulders pattern is considered complete only after a neckline break, the detector must not use future candles after the breakout to decide that the pattern existed.

Future data may be used only by the backtester to measure the outcome after detection.

### 2. Keep detection separate from profitability

The pattern detector should answer:

> "Was a pattern detected here?"

The backtester should separately answer:

> "What happened after the pattern was detected?"

Do not make a pattern count as detected only because the future trade happened to be profitable.

### 3. Store enough information for later analysis

Each detection should record useful information such as:

- pattern type
- detection time
- start/end candle
- key peaks/troughs
- neckline or breakout level
- pattern height
- direction
- confidence / fit score
- any thresholds used

This will later allow us to analyze which versions of a pattern are more profitable.

### 4. Make thresholds configurable

Do not hide tolerances as magic numbers.

Examples:

- allowed shoulder-height difference
- required head prominence
- minimum spacing between extrema
- neckline tolerance
- minimum pattern length
- smoothing parameters

Put these into a configuration dataclass so they can later be varied systematically.

### 5. Build for reproducible experiments

The same detector and same parameters must be usable on:

- the current random-walk synthetic market
- future synthetic market worlds
- real historical market data

Do not tune the detector specifically to one generated price path.

---

# What comes immediately after the detector

Once `pattern_detector.py` works:

1. Run it over many random-walk simulations using different seeds.
2. Record how often each pattern appears.
3. Build a backtester that enters a trade only after pattern completion.
4. Measure post-pattern returns and strategy P&L.
5. Compare results with the null expectation from random markets.
6. Only after the baseline is working, introduce more structured synthetic worlds such as:
   - drift / trending markets
   - mean-reverting markets
   - volatility-regime markets
   - momentum regimes
7. Run the exact same detection code in each world.
8. Later run it on real-market data.

This lets the research question evolve from simply:

> "Are chart patterns profitable?"

into the stronger question:

> "Under which market-generating conditions do chart patterns become informative, and do those effects survive in real financial data?"

---

# Future AI layer

Do not implement this until the baseline detector and backtester are operational or until the exposition requirements specify the AI method.

Likely AI extensions include:

### Option A — AI validates classical detections

The rule-based detector identifies candidate patterns.

A machine-learning classifier then predicts whether each detected pattern is likely to succeed using features such as:

- pattern geometry
- pattern length
- volatility
- volume
- trend before the pattern
- market regime
- neckline slope
- breakout strength

This is probably the cleanest extension because the performance of the AI can be directly compared with the classical detector.

### Option B — AI learns patterns directly

Give a model windows of OHLC/candlestick data and train it to predict future direction or profitability.

Possible models:

- logistic regression as a baseline
- gradient boosting
- MLP
- CNN
- LSTM
- GRU

### Option C — Generic pattern similarity

Implement a DTW/UCR-style approach inspired by Tsinaslanidis & Guijarro:

1. take the current price window as a query
2. search historical training data for similar sequences
3. label those references according to what happened next
4. infer a long/short/neutral signal from their outcomes

This does not necessarily require deep learning, but it provides a sophisticated automatic pattern-recognition benchmark.

---

# Current development priority

For now, prioritize this sequence:

**Simulator → Candles → Classical pattern detector → Backtester → Statistical evaluation → AI extension → Real-market data**

The simulator and candles stages already exist.

The next coding task is therefore:

> **Create `pattern_detector.py` as a modular, look-ahead-safe classical chart-pattern recognition system, starting with one pattern and producing structured detection records that can later feed both a backtester and an AI model.**

Do not modify `market_maker_simulator.py` or `chart_renderer.py` unnecessarily while building the detector. Import and reuse their existing data structures where appropriate.
