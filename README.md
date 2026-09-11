# Market-maker-simulator
This project simulates different market environments to explore if chart patterns predict price movements. It includes random, mathematical, and emotional markets. The goal is to test, analyze, and document findings, while building quant skills for future finance roles.

## Atlas: one million structures with two seeds

The default pipeline now generates compressed shards on **D:**, constructs
specifications in batches, trains from disk, and streams evaluation. All seeds
of a structure remain in the same split. A sampled process-tree memory guard
adapts to available RAM on a 16 GB laptop.

```powershell
.\.venv\Scripts\python.exe atlas_pipeline.py all --structures 1000000 --seeds-per-structure 2 --jobs 1 --output-dir D:/Atlas/atlas_1000000x2
```

Use `build-dataset` for generation only and `--resume` with the original command
to recover committed generation shards. Models use incremental logistic
regression, CPU external-memory XGBoost and a disk-fed PyTorch MLP. Some training
and metric procedures intentionally differ from the historical implementation.

Read [ATLAS_RUNBOOK.md](ATLAS_RUNBOOK.md) for the independent capacity pilot,
disk/runtime estimates, memory limits, changed statistical procedures and exact
launch commands. A successful small pilot does not certify the complete run.

## Historical small CSV export

The following describes the retained `--legacy-in-memory` export, not the default
Atlas storage format. Default data-only invocations now use `data/shards/*.csv.gz`
and the SQLite catalog described in the runbook.

`ai_pattern_dataset_generator.py` generates data only. It reuses the existing simulator,
candle builder, deterministic detector and execution-cost calculation. It fits no models
and computes no outcome summaries on the held-out test partition.

With the dependencies in `requirements.txt` installed, start with a small run:

```powershell
.\.venv\Scripts\python.exe ai_pattern_dataset_generator.py --legacy-in-memory --structures 10 --seeds-per-structure 2 --steps 3000 --jobs 2 --output-dir D:/Atlas/legacy_small_csv
```

The default design uses 1,000 sampled structures, one seed per structure, five simulator
steps per candle, a 20-candle holding period and one basis point of brokerage per side.
`--horizon`, `--brokerage-bps` and `--slippage-bps` configure labels; `--market-lookback`
configures the microstructure lookback. Candle return/volatility features have explicitly
named fixed lookbacks. Use `--help` for all options and a fresh output directory for each run.

The output contains:

- `synthetic_pattern_dataset.csv`: one row per unique confirmed breakout with a resolved
  trade direction and enough future candles for a complete label.
- `train.csv`, `validation.csv`, `test.csv`: the same rows in separate partitions.
- `feature_schema.json`: explicit input allowlists for Model A (observable) and Model B
  (observable plus Oracle), the classification target, units and timing conventions.
- `structure_split_manifest.csv` and `run_split_manifest.csv`: assignments frozen from
  the sampled design **before simulation**, including runs that detect no patterns.
- `simulation_design.jsonl` and `ai_research_manifest.json`: full sampled configurations,
  execution assumptions, detector settings and source fingerprints.
- `synthetic_runs.csv`: detection counts and exclusions for incomplete horizons or
  unresolved directions. These excluded signals are never given fabricated labels.
- `dataset_manifest.json` and `dataset_status.json`: file hashes, row counts and completion
  status. Use outputs only when the status is `DATASET_COMPLETE`.

The 60%/20%/20% allocation applies to **whole market structures**, stratified by null versus
structured design. All repeated seeds and overlapping events from a structure stay together.
Integer rounding and different pattern frequencies mean row counts need not be exactly
60%/20%/20%. Small datasets may have an empty or single-class partition; generation preserves
the frozen assignment rather than searching for a favorable split.

Features stop at the close of `available_at_index` (zero-based), also recorded as
`feature_end_step`. A pattern ending earlier is measured when its breakout becomes known,
not retroactively at its geometric endpoint. Entry uses the next executable bar's opening
quote; exit uses the closing quote of `entry_index + 19` for the default horizon. Thus a
signal confirmed at bar 100 enters at bar 101 and exits at bar 120.

| Concept | Dataset field |
| --- | --- |
| Pattern confidence / length | `geometry_fit_score` / `pattern_length_bars` |
| Volatility / momentum | `recent_volatility_20` / `recent_return_20` |
| Relative volume / trend strength | `recent_volume_ratio` / `recent_trend_strength_20` |
| Spread / order flow | `observed_spread_bps_current` / `observed_order_flow_imbalance` |
| Hidden mechanisms and current latent state | `oracle_*` columns |
| Future asset return before direction and costs | `future_20_bar_return` |
| Return adjusted for long/short direction and costs | `net_directional_return` |
| Binary classification target | `profitable_after_costs` (`profitable` compatibility alias) |

Returns are fractions: `0.012` is 1.2%. Cost-adjusted P&L uses fixed notional normalized by
the entry quote midpoint, actual entry/exit spread and fees/slippage on both sides. A flat
or negative net return is class 0. Spread features use public quotes; internal simulator
reference prices, trader weights, latent liquidity and regime state are excluded from
Model A. The detector's geometry score is not a calibrated profitability probability.

Select model inputs using `feature_schema.json` or `feature_columns("observable")` /
`feature_columns("oracle")` from `ai_pattern_research_pipeline.py`. Never train on every
numeric CSV column: IDs, seeds, world-family metadata, split membership and future outcomes
are excluded. Fit imputation/scaling on train only, choose the trade threshold on validation,
and reserve test evaluation for the final frozen experiment.

Generation streams rows and bounds worker buffering; it does not load the complete pattern
table into memory. The master and partition CSVs together store each row twice. Runtime and
disk use depend on path length and signal density; increase `--structures` after measuring
a small run. No fixed number of simulations guarantees a particular pattern count.

The existing research trainer can later read this generator's frozen raw dataset via its
`train --legacy-in-memory --output-dir ...` command; that command includes final test evaluation and should only
be invoked when the experiment is ready. Earlier generated datasets must be rebuilt to use
the corrected observable quote features in pipeline version 2.5.0.

Run regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
