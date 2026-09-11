# Atlas on a 16 GB laptop

The default pipeline now constructs specifications in batches, saves compressed
shards once, trains from disk and streams evaluation. All production data, models,
predictions, SQLite databases and temporary caches must resolve to **D:**. The CLI
will not fall back to C:. The simulator, causal detector and execution costs are
shared with the existing research code.

## Requested experiment

**1,000,000 structures x 2 seeds = 2,000,000 runs.** This full experiment has not
been launched automatically. Start it after reviewing independent capacity data:

```powershell
.\.venv\Scripts\python.exe atlas_pipeline.py all --structures 1000000 --seeds-per-structure 2 --steps 3000 --seed 42 --jobs 1 --runs-per-shard 32 --batch-rows 4096 --output-dir D:/Atlas/atlas_1000000x2
```

The command trains both observable and oracle models: up to 120 MLP epochs with
early stopping, 10 incremental logistic epochs and up to 300 boosting rounds.
It retains the original validation threshold/support defaults.

Replace `all` with `build-dataset` to generate without model training or test
outcome summaries. Train/evaluate the frozen dataset afterward with:

```powershell
.\.venv\Scripts\python.exe atlas_pipeline.py train --jobs 1 --batch-rows 4096 --output-dir D:/Atlas/atlas_1000000x2
```

Train-only mode reads experiment, labels, costs and MLP settings from the dataset
manifest. Batch size, logistic epochs, boosting rounds and feature sets are frozen
in `streaming_training_protocol.json` before training/test evaluation. Reusing a
training directory with a changed protocol/software is rejected.

`ai_pattern_research_pipeline.py` and `ai_pattern_dataset_generator.py` also use
streaming by default. The latter generates only. Historical `--legacy-in-memory`
mode and Python APIs remain for small-data compatibility checks; they are not the
Atlas path. Legacy CSV/checkpoint data cannot resume as a new streaming dataset.

## Independent capacity pilot

This separate pilot deliberately shortens training and relaxes validation support
to exercise all stages. Its outputs measure capacity, not final research evidence.
Do not copy these reduced settings into the million-structure experiment.

```powershell
.\.venv\Scripts\python.exe atlas_pipeline.py all --structures 96 --seeds-per-structure 2 --steps 3000 --seed 918273 --jobs 2 --runs-per-shard 32 --batch-rows 4096 --mlp-epochs 2 --mlp-batch-size 128 --mlp-patience 2 --logistic-epochs 3 --xgb-rounds 5 --threshold-min 0.1 --threshold-max 0.9 --threshold-step 0.2 --min-validation-trades 5 --min-validation-runs 2 --explanation-rows 64 --output-dir D:/Atlas/capacity_verified_20260911
```

Project actual compressed data, catalog, prediction and observed XGBoost cache
sizes without opening observations or inspecting test performance:

```powershell
.\.venv\Scripts\python.exe atlas_capacity.py --pilot-dir D:/Atlas/capacity_verified_20260911 --structures 1000000 --seeds-per-structure 2 --storage-dir D:/
```

The disk allowance adds 50% headroom plus a 5 GiB reserve. Generation runtime uses
a linear pilot-rate estimate assuming the same path length, sampling settings,
worker count and machine conditions. Small pilots can miss dense/rare structures.
Measure increasing sizes with intended training settings: repeated disk reads can
dominate model runtime. Full-scale certification remains false in the report.

### Measured on 2026-09-11

The verified pilot at `D:/Atlas/capacity_verified_20260911` completed 192 runs,
produced 2,198 observations, and exercised both feature sets, all three models,
validation, test evaluation and SHAP. Total time was 151.5 seconds; generation
took 124.0 seconds. All memory-monitor stages were observed.

| Measurement | Result |
| --- | --- |
| Adaptive process-tree budget | 3.29 GiB |
| Peak summed process-tree private memory | 1.76 GiB |
| Peak summed process-tree RSS | 0.47 GiB |
| Minimum available physical RAM | 3.08 GiB |
| Maximum process count | 3 (parent and two workers) |
| Projected compressed data and catalog for 1,000,000 x 2 | 13.18 GiB |
| Projected larger live XGBoost disk cache | 21.07 GiB |
| Projected disk allowance including exports, 50% headroom and reserve | 58.03 GiB |
| D: free at measurement | 76.83 GiB |
| Generation-only extrapolation with two workers | 14.95 days |

The default launch uses one worker and will need a different runtime estimate.
Training runtime at the full model settings is not measured. Pilot estimates
passed the capacity screen, but full-scale RAM, disk and runtime remain unproven.
Details are saved in `resource_report.json` and
`capacity_projection_1000000x2.json` beside the D: pilot.

The bounded optimizer comparison measured mean absolute probability differences
of 0.040 (observable) and 0.112 (oracle) after three SGD epochs. These are material
procedure differences, not a claim that incremental and batch logistic regression
are equivalent. Full validation metrics and calibration differences are saved.

The first 256-row-batch pilot remains at `D:/Atlas/capacity_pilot_20260911` as
separate evidence. Its larger projected cache shows why batch settings matter.
Use fresh output directories when repeating a pilot; source fingerprints bind
each completed dataset to the code used to generate it.

## Memory and storage

- SQLite stores the growing design and structure/seed uniqueness indexes. There
  is no million-item Python specification list, seed set or split map.
- A shard has at most `--runs-per-shard` runs, rounded down to complete structures.
  Workers have bounded outstanding results. The laptop CLI allows one or two
  workers; each simulation still holds its own path/detections, so `--steps`
  also affects memory.
- Observations are stored once as gzip CSV. Readers yield at most `--batch-rows`
  observations per training batch, with no master/partition CSV duplication.
- The MLP and logistic models reread disk batches. CPU XGBoost 3.x uses actual
  `ExtMemQuantileDMatrix` files in `xgboost_cache` on D:, with no in-memory fallback.
  Cache size is recorded before handles/files are released.
- XGBoost still retains labels, gradients and metadata in RAM. A planning check
  allows 32 bytes per train/validation row and rejects an obviously insufficient
  remaining budget. This is a heuristic, not a bound on native allocations.
- Evaluation retains one run for overlap resolution, predicts in smaller batches,
  writes compressed predictions and aggregates metrics. Seeds are averaged within
  structure before uncertainty estimates; zero-pattern runs contribute zero.

The default budget is the minimum of 6 GiB, half physical RAM, and current process
usage plus 65% of currently available physical RAM. `--memory-budget-gib` may lower
it. Defaults also reserve 1 GiB available RAM and 5 GiB free disk. Parent and worker
RSS/private memory are sampled every 0.5 seconds and at cooperative stage checks.

Generation prints one progress line every 1,000 completed simulations and a final
100% line after dataset finalization, including percentage, elapsed time and ETA.
For 2,000,000 runs that is 2,000 generation progress lines. Elapsed time includes
saved active generation time on resume; ETA uses the current invocation's run
rate and excludes training/evaluation. Progress counts processed simulations;
recovery restarts from the last committed shard. Resource checks and periodic
disk snapshots continue silently between generation progress messages.

`resource_report.json` records this invocation's peaks, per-stage samples, minimum
available RAM and breaches. This is a sampled cooperative guard, not an OS memory
allocation cap. Native allocations can fail between samples, swap is not counted
as RAM, and a small successful run does not prove full-scale fit.

## Resume and integrity

Repeat the original generation/all command with `--resume`, using the same output
directory, source snapshot and scientific settings. Generation may safely change
worker count or shard size. `--checkpoint` is also accepted on initial generation
and recovery. Model training restarts from the beginning.

A committed structure-aligned shard, run summaries, uniqueness indexes and sampler
state form one checkpoint. Incomplete shards are regenerated. An OS writer lock
protects generation; use only one pipeline invocation per result directory.
Catalog and shard hashes are verified on reads. Require dataset status `COMPLETE`
and overall status `EVALUATION_COMPLETE` before using complete model results.

## Changed research procedure

This is a new frozen experiment, not numerical equivalence to the historical pilot.

- Exact null counts use sequential sampling without a full flag vector. SQLite
  ensures unique structures and integer seeds across all partitions.
- Splits use seeded SHA-256 of structure identity before outcomes exist, targeting
  60/20/20. The first three structures reserve one partition each. Both seeds stay
  together. Fractions are approximate and not stratified by family; actual counts
  are recorded without searching for favorable outcomes.
- Train-only running means replace numeric medians. Every numeric input receives
  a fixed missing indicator. Categorical vocabularies have at most 64 entries per
  feature, including separate missing/unknown encodings.
- Averaged `SGDClassifier(loss="log_loss").partial_fit` replaces batch logistic
  regression; validation log loss selects the epoch. Each feature set saves a
  deterministic solver regression check and a capped real train/validation prefix
  comparison against batch logistic regression: probability difference, log loss,
  Brier score and calibration error. The data comparison is diagnostic, does not
  assert equivalence, and never reads test data.
- ROC-AUC and average precision use fixed 4096-bin probability histograms and are
  explicitly approximate, including primary-model ranking. Brier score, confusion
  counts, fixed-bin calibration and structure-level mean/variance calculations
  are streamed exactly up to floating-point arithmetic.
- Medians are marked unavailable instead of retaining every return. Native tree
  SHAP and explanation exports use a uniform bounded event sample (512 by default).
  Explanations describe associations rather than establish causality.

Insufficient validation trade/run support stops before test evaluation. Thresholds
and the primary model are saved before the test stream opens. The two-feature
Bonferroni factor and complete structure denominators are retained.

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/atlas_dataset_manifest.json` | Frozen settings, fingerprints, counts and compression measurements |
| `data/atlas_catalog.sqlite3` | Compressed configurations/summaries, splits, seeds and shard catalog |
| `data/shards/part-*.csv.gz` | Single compressed copy of observations |
| `streaming_training_protocol.json` | Dataset binding and frozen training/software settings |
| `streaming_training_{feature_set}.json` | Preprocessing, model/epoch details and measured cache bytes |
| `logistic_solver_validation_*.json`, `logistic_data_validation_*.json` | Incremental solver checks and comparison diagnostics |
| `primary_model_{feature_set}.json` | Validation-only choices |
| `test_predictions_*.csv.gz`, `test_run_returns_*.csv.gz` | Streamed observations and all-run return denominators |
| `model_comparison_all.csv` | At most six model summary rows |
| `resource_report.json`, `atlas_status.json` | Memory evidence and completion/failure state |

Run regressions with the project interpreter:
`python -m unittest discover -s tests -v`.
API references: [XGBoost external memory](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html)
and [incremental logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
