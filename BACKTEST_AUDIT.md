**Backtest integration audit — 2026-09-08**

Reviewed `market_maker_simulator.py`, `chart_renderer.py`, `pattern_detector.py`,
`market_structure_experiment.py`, and `ai_pattern_research_pipeline.py` together.
The supplied findings were treated as a defect report. Existing workspace changes and
previous research results were preserved.

**Corrections to the 13 reported issues**

1. Matched-null randomness now uses an OHLCV prefix available at the signal time,
   pattern name, entry index and horizon. Public event IDs, filenames, display labels,
   git labels and future bars do not seed it.
2. Controls must avoid the entire signal return interval plus the embargo. For horizon H,
   control entry N is excluded when `[N, N+H-1]` intersects
   `[P-embargo, P+H-1+embargo]` for any confirmed signal entry P.
3. Both matched controls and unconditional comparisons use each event's observation-scale
   warm-up, including its execution delay. Valid early short-scale events remain eligible.
4. Real validation requires explicit asset/venue metadata. Overlapping windows of the same
   asset are rejected. Overlapping periods across assets, periods within the horizon/embargo
   buffer, and declared dependence groups are merged transitively into inferential clusters.
   Adding correlated files does not increase the independent-observation count.
5. Real timestamps must be timezone-aware ISO-8601 bar-open timestamps with regular spacing.
   Explicit fixed-duration labels are checked against that spacing. Mixed durations, missing
   bars and irregular session calendars are rejected by the supported continuous-bar policy.
6. Real execution preserves distinct opening and closing quote mids and spreads from
   `open_bid`, `open_ask`, `close_bid`, `close_ask`. Ambiguous `bid`/`ask` and incomplete
   boundary quotes are rejected. Without quotes, the output explicitly identifies modeled
   execution using OHLC mids and the frozen fallback spread.
7. Raw OHLC must be complete; adjusted close alone is never substituted. Fully adjusted
   data requires all four `adj_open`, `adj_high`, `adj_low`, `adj_close` columns and an
   explicit consistent corporate-action policy, including execution quotes.
8. AI persists and checks its own procedure contract. Optional
   `--reference-procedure-manifest` verifies the statistical procedure hash and shared
   candle duration, horizon membership, costs, overlap and execution implementation.
9. AI explicitly declares its sampled mechanism-discovery universe and paired-filter
   estimand separate from the statistical scenario grid and matched-null estimand.
   Matching shared settings does not claim identical universes or comparable headline rates.
10. If no threshold satisfies validation support requirements, AI raises
    `INSUFFICIENT_VALIDATION_SUPPORT`; the run cannot complete confirmatory evaluation.
11. Every detector candidate, including a promoted confirmed member, receives explicit
    competition metadata as of its own availability time. Earlier event metadata stays immutable.
12. Direction accuracy and its uncertainty are calculated across run/cluster accuracy
    observations. The event-weighted hit rate is retained only as a separately named
    descriptive statistic. Direction lift compares the same subset of matched signal events.
13. Exported additive P&L and drawdown aggregates use `return_points` names and explicit
    fixed-notional accounting metadata. They are not compounded portfolio returns.
    Internal statistical `RunMetric.total_*` fields remain documented compatibility fields.

**Further defects found and fixed during repeat reviews**

- Provenance-dependent simultaneous-signal tie breaking could select a different detector scale.
  Both evaluators now share causal pattern/geometry/scale ordering. Equal-score AI baseline
  family ordering also agrees with the statistical baseline.
- Malformed explicit synthetic closing quotes could silently invoke a legacy execution fallback.
  They now fail, and quote midpoint arithmetic avoids unnecessary intermediate overflow.
- The AI threshold grid could exceed its frozen maximum through floating-point endpoint handling.
- Duplicate DataFrame index labels could duplicate executed fills.
- An incomplete evaluation run universe could silently omit returns or zero-pattern runs.
- Cross-structure RNG seed collisions were not prevented, and mismatched row/run structure
  identities could enter the split. Both are checked.
- AI uncertainty now clusters repeated seeds within sampled structures, including structures
  with zero eligible patterns. The observable/oracle primary tests have a two-family correction;
  other model comparisons are explicitly exploratory.
- Statistical aggregation now rejects duplicate run/seed observations and mixed partitions.
  Validation must supply every selected hypothesis, preventing an incomplete result set from
  silently reducing the Holm correction family.
- Frozen statistical procedure loading rejects missing/unknown fields and invalid numeric types
  instead of truncating fractional integers or silently applying defaults.
- Real CSV validation rejects malformed rows and duplicate headers; arbitrary time labels and
  incomplete or out-of-procedure hypothesis lists cannot reach inferential validation.
- AI and statistical CLI run-status files invalidate stale completion claims when reruns fail.
- Future excursion fields explicitly describe transaction-candle extremes, not executable quote
  paths, and remain excluded from AI model features.

**Verification**

The full regression suite passes 54 tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Coverage includes actual identical real CSV bytes under different filenames/labels,
synthetic git-label invariance, full-window exclusions and their exact boundaries,
multi-scale warm-up, insufficient controls, detector prefix causality, exact quote fills,
identical AI/statistical trade returns, clustered inference, threshold support, manifests,
real-validation CLI execution, and stale-run rejection.

End-to-end smoke runs exercise statistical discovery/validation over three structures and
three seeds per partition (18 paths), including multiprocessing, and AI dataset generation
over 36 structures followed by Logistic Regression, XGBoost and PyTorch training with both
observable and oracle feature sets. Reloading the generated frozen AI dataset and verifying
the matching statistical procedure manifest also succeeded.
These small runs check integration; they are not research evidence and deliberately use small
sample sizes/training budgets. Smoke artifacts are under `tmp/audit_*`.

Repeated reviews, including an independent final cross-file review, found no further concrete
defects in the reviewed contracts. This is bounded verification, not a proof of bug-free software
or of statistical assumptions for arbitrary future datasets. Temporal clustering cannot establish
independence by itself; use frozen `dependence_group` values for additional known dependence.

**Using the stricter interfaces**

Real validation now requires `--dataset-manifest datasets.json`. Example content, with paths
resolved relative to that manifest:

```json
{
  "datasets": [
    {
      "path": "btc.csv",
      "asset": "BTC-USD",
      "venue": "EXCHANGE",
      "timeframe": "1h",
      "price_basis": "raw",
      "corporate_action_policy": "none_in_window"
    }
  ]
}
```

Optional conventions default to `session_policy: continuous` and
`timestamp_semantics: bar_open`. A `dependence_group` can join additional dependent windows.
The real CSV loader now returns `RealExecutionData` with candles and all four execution arrays.
Session-gap data must be prepared under a separately justified policy; the current loader does
not silently fill it. Merely changing the manifest cannot make incompatible data valid.

To check shared AI/statistical settings, provide the freshly generated statistical
`procedure_manifest.json` via `--reference-procedure-manifest`. AI remains a separate sampled
study; use the statistical all-pattern portfolio when inspecting its corresponding baseline.

Rebuild datasets and rerun discovery/validation before relying on new conclusions. Old manifests
and result files describe the prior implementation and fail current code/procedure fingerprint
checks. Consumers of result CSVs must adopt the explicit return-point column names. Check
`ai_run_status.json` or `experiment_run_status.json` for successful completion of the current run.
