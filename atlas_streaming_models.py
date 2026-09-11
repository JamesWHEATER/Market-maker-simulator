"""Bounded training for Atlas' sharded on-disk datasets.

The streaming procedure deliberately differs from the legacy in-memory procedure:
training-only running-mean imputation replaces median imputation, categorical
vocabularies are capped, and averaged SGD replaces the batch logistic solver.
All differences and the deterministic solver comparison are saved with each run.
XGBoost >=3.0 uses actual CPU ExtMemQuantileDMatrix disk pages; an iterator passed
to QuantileDMatrix would still concatenate the features in memory and is forbidden.
External-memory boosting still requires per-row metadata and gradients in RAM.

API references:
https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss

from ai_pattern_research_pipeline import FUTURE_OR_OUTCOME_COLUMNS, feature_columns


STREAMING_TRAINING_VERSION = "atlas-disk-training-v1"
LABEL_COLUMN = "profitable"


def _check(monitor: Any, stage: str) -> None:
    if monitor is not None:
        monitor.check(stage)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="raise").to_numpy(dtype=np.float64)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("profitable labels must be nonmissing binary 0/1 values")
    return labels.astype(np.int32)


def _category_token(value: Any) -> str:
    return "<MISSING>" if pd.isna(value) else "value:" + str(value)


class StreamingPreprocessor:
    """Train-only stable moments plus a fixed, bounded categorical vocabulary.

    Missing numeric values become the training mean (zero if wholly missing).
    Numeric scale is the population standard deviation *after* mean imputation.
    Every numeric feature receives an unscaled binary missing indicator. Unknown
    and missing categories have distinct columns, even if absent during fitting.
    """

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str], *, max_categories: int = 64):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.max_categories = _positive_int(max_categories, "max_categories")
        if self.max_categories < 2:
            raise ValueError("max_categories must leave room for missing and unknown categories")
        leakage = FUTURE_OR_OUTCOME_COLUMNS.intersection(self.numeric + self.categorical)
        if leakage:
            raise ValueError(f"Future/outcome feature leakage: {sorted(leakage)}")
        if len(set(self.numeric + self.categorical)) != len(self.numeric + self.categorical):
            raise ValueError("feature columns must be unique")
        self.fitted_ = False

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    def _numeric_array(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing feature columns: {sorted(missing)}")
        # pandas 3 may expose a read-only view; transformation imputes this bounded buffer.
        return frame[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)

    def fit(self, dataset: Any, *, batch_rows: int = 4096, monitor: Any = None) -> "StreamingPreprocessor":
        batch_rows = _positive_int(batch_rows, "batch_rows")
        n_features = len(self.numeric)
        counts = np.zeros(n_features, dtype=np.int64)
        means = np.zeros(n_features, dtype=np.float64)
        m2 = np.zeros(n_features, dtype=np.float64)
        categories = {name: set() for name in self.categorical}
        self.rows_seen_ = 0
        self.class_counts_ = np.zeros(2, dtype=np.int64)
        for frame in dataset.iter_batches(split="train", batch_rows=batch_rows,
                                          columns=self.columns + [LABEL_COLUMN]):
            _check(monitor, "preprocessor batch")
            values = self._numeric_array(frame)
            finite = np.isfinite(values)
            batch_counts = finite.sum(axis=0)
            clean = np.where(finite, values, 0.0)
            batch_means = np.divide(clean.sum(axis=0), batch_counts,
                                    out=np.zeros(n_features), where=batch_counts != 0)
            centered = np.where(finite, values - batch_means, 0.0)
            batch_m2 = (centered * centered).sum(axis=0)
            combined = counts + batch_counts
            delta = batch_means - means
            means += np.divide(delta * batch_counts, combined,
                               out=np.zeros(n_features), where=combined != 0)
            m2 += batch_m2 + np.divide(delta * delta * counts * batch_counts, combined,
                                       out=np.zeros(n_features), where=combined != 0)
            counts = combined
            self.rows_seen_ += len(frame)
            self.class_counts_ += np.bincount(_labels(frame), minlength=2)
            for name in self.categorical:
                # Values beyond this fixed cap are encoded as unknown. No unbounded set.
                known = categories[name]
                for value in frame[name]:
                    token = _category_token(value)
                    if token != "<MISSING>" and len(known) < self.max_categories - 2:
                        known.add(token)
            del frame, values, finite, clean, centered
        if self.rows_seen_ == 0:
            raise ValueError("Training split has no observations")
        self.mean_ = means
        self.observed_counts_ = counts
        self.scale_ = np.sqrt(np.maximum(m2 / self.rows_seen_, 0.0))
        self.scale_[self.scale_ <= np.finfo(np.float64).eps] = 1.0
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("Numeric training moments overflowed; inspect dataset feature magnitudes")
        self.categories_ = {name: ["<MISSING>", "<UNKNOWN>", *sorted(values)]
                            for name, values in categories.items()}
        self.category_maps_ = {name: {value: index for index, value in enumerate(values)}
                              for name, values in self.categories_.items()}
        self.fitted_ = True
        _check(monitor, "preprocessor complete")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("StreamingPreprocessor must be fitted before transforming")
        values = self._numeric_array(frame)
        missing = ~np.isfinite(values)
        values[missing] = np.broadcast_to(self.mean_, values.shape)[missing]
        n_numeric = len(self.numeric)
        result = np.zeros((len(frame), len(self.get_feature_names_out())), dtype=np.float32)
        result[:, :n_numeric] = (values - self.mean_) / self.scale_
        result[:, n_numeric:2 * n_numeric] = missing
        offset = 2 * n_numeric
        row_indices = np.arange(len(frame))
        for name in self.categorical:
            mapping = self.category_maps_[name]
            encoded = np.fromiter((mapping.get(_category_token(value), 1) for value in frame[name]),
                                  dtype=np.int32, count=len(frame))
            result[row_indices, offset + encoded] = 1.0
            offset += len(mapping)
        if not np.isfinite(result).all():
            raise ValueError("Transformed features exceed float32 range")
        return result

    def get_feature_names_out(self) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("StreamingPreprocessor is not fitted")
        return np.asarray(self.numeric + [f"missing_{name}" for name in self.numeric] +
                          [f"{name}={value}" for name in self.categorical
                           for value in self.categories_[name]], dtype=object)

    def metadata(self) -> dict[str, Any]:
        return {"imputation": "training-only running mean; wholly missing numeric columns -> 0",
                "scaling": "training population variance after mean imputation",
                "missing_indicators": "fixed binary indicator for every numeric feature, unscaled",
                "categorical": "first observed capped training vocabulary; explicit missing/unknown",
                "max_categories_per_feature": self.max_categories,
                "rows_seen": self.rows_seen_, "numeric_features": self.numeric,
                "categorical_features": self.categorical,
                "categories": self.categories_, "class_counts": self.class_counts_.tolist(),
                "feature_count": len(self.get_feature_names_out())}


def _xy_batches(dataset: Any, preprocessor: StreamingPreprocessor, split: str,
                batch_rows: int, monitor: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for frame in dataset.iter_batches(split=split, batch_rows=batch_rows,
                                      columns=preprocessor.columns + [LABEL_COLUMN]):
        _check(monitor, f"{split} transform")
        if len(frame):
            yield preprocessor.transform(frame), _labels(frame)


def _validation_loss(dataset: Any, preprocessor: StreamingPreprocessor, predict: Any,
                     *, batch_rows: int, monitor: Any = None) -> float:
    total_loss, count = 0.0, 0
    for x, y in _xy_batches(dataset, preprocessor, "validation", batch_rows, monitor):
        p = np.asarray(predict(x), dtype=np.float64)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        total_loss += float((-y * np.log(p) - (1 - y) * np.log1p(-p)).sum())
        count += len(y)
    if count == 0:
        raise ValueError("Validation split has no observations")
    return total_loss / count


def train_incremental_logistic(dataset: Any, preprocessor: StreamingPreprocessor, *,
                               seed: int = 42, epochs: int = 10, batch_rows: int = 4096,
                               monitor: Any = None) -> SGDClassifier:
    """One bounded partial_fit call per disk batch; select epoch on validation loss."""
    epochs = _positive_int(epochs, "logistic_epochs")
    batch_rows = _positive_int(batch_rows, "batch_rows")
    if np.any(preprocessor.class_counts_ == 0):
        raise ValueError("Training requires both profitable classes; generate more structures")
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / preprocessor.rows_seen_,
                          learning_rate="invscaling", eta0=0.05, power_t=0.25,
                          average=True, shuffle=False, random_state=int(seed) % (2**32),
                          class_weight=None, n_jobs=1)
    rng = np.random.default_rng(int(seed) % (2**32))
    best, best_loss, best_epoch = None, math.inf, 0
    history = []
    for epoch in range(1, epochs + 1):
        for x, y in _xy_batches(dataset, preprocessor, "train", batch_rows, monitor):
            permutation = rng.permutation(len(y))
            model.partial_fit(x[permutation], y[permutation], classes=np.asarray([0, 1]))
            _check(monitor, f"logistic epoch {epoch} batch")
        loss = _validation_loss(dataset, preprocessor, lambda x: model.predict_proba(x)[:, 1],
                                batch_rows=batch_rows, monitor=monitor)
        history.append({"epoch": epoch, "validation_log_loss": loss})
        if loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model), loss, epoch
        _check(monitor, f"logistic epoch {epoch} complete")
    assert best is not None
    best.streaming_history_ = history
    best.streaming_epochs_trained_ = epochs
    best.streaming_best_epoch_ = best_epoch
    return best


class StreamingMLP:
    """CPU MLP with bounded disk batches and validation-loss accumulation."""

    def __init__(self, input_dim: int, experiment: Any):
        import torch
        from torch import nn
        self.input_dim = input_dim
        self.seed = int(experiment.base_seed) % (2**32)
        self.batch_size = int(experiment.mlp_batch_size)
        self.epochs = int(experiment.mlp_epochs)
        self.patience = int(experiment.mlp_patience)
        self.learning_rate = float(experiment.mlp_learning_rate)
        self.threads = max(1, min(2, os.cpu_count() or 1))
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.model = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.20),
                                   nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                   nn.Linear(16, 1)).cpu()
        self.history_ = []
        self.best_validation_loss = math.inf
        self.epochs_trained = 0
        self.best_epoch = 0

    def fit(self, dataset: Any, preprocessor: StreamingPreprocessor, *, batch_rows: int,
            monitor: Any = None) -> "StreamingMLP":
        import torch
        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_function = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        best_state, stale = None, 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for x, y in _xy_batches(dataset, preprocessor, "train", batch_rows, monitor):
                permutation = rng.permutation(len(y))
                for start in range(0, len(y), self.batch_size):
                    indices = permutation[start:start + self.batch_size]
                    tensor_x = torch.from_numpy(x[indices])
                    tensor_y = torch.from_numpy(y[indices].astype(np.float32))
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_function(self.model(tensor_x).squeeze(1), tensor_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()
                _check(monitor, f"mlp epoch {epoch} batch")
            validation_loss = _validation_loss(dataset, preprocessor, self.predict_proba,
                                                batch_rows=batch_rows, monitor=monitor)
            self.epochs_trained = epoch
            self.history_.append({"epoch": epoch, "validation_log_loss": validation_loss})
            if validation_loss < self.best_validation_loss - 1e-6:
                self.best_validation_loss = validation_loss
                self.best_epoch = epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            _check(monitor, f"mlp epoch {epoch} complete")
            if stale >= self.patience:
                break
        if best_state is None:
            raise ValueError("MLP did not obtain a finite validation loss")
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        result = np.empty(len(values), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                x = torch.from_numpy(np.asarray(values[start:start + self.batch_size], dtype=np.float32))
                result[start:start + len(x)] = torch.sigmoid(self.model(x).squeeze(1)).cpu().numpy()
        return result

    def save(self, path: Path) -> None:
        import torch
        torch.save({"state_dict": self.model.state_dict(), "input_dim": self.input_dim,
                    "seed": self.seed, "batch_size": self.batch_size,
                    "epochs_trained": self.epochs_trained, "best_epoch": self.best_epoch,
                    "best_validation_loss": self.best_validation_loss,
                    "training_version": STREAMING_TRAINING_VERSION}, path)


def require_external_memory_xgboost() -> Any:
    import xgboost as xgb
    if int(xgb.__version__.split(".")[0]) < 3 or not hasattr(xgb, "ExtMemQuantileDMatrix"):
        raise RuntimeError("Streaming Atlas requires xgboost>=3.0 with ExtMemQuantileDMatrix; "
                           "upgrade XGBoost. There is no in-memory fallback.")
    return xgb


def train_external_xgboost(dataset: Any, preprocessor: StreamingPreprocessor, *, output_dir: Path,
                           seed: int = 42, rounds: int = 300, batch_rows: int = 4096,
                           monitor: Any = None) -> tuple[Any, dict[str, Any]]:
    rounds = _positive_int(rounds, "xgb_rounds")
    batch_rows = _positive_int(batch_rows, "batch_rows")
    xgb = require_external_memory_xgboost()
    row_counts = getattr(dataset, "manifest", {}).get("rows_by_split", {})
    resident_row_allowance = 32 * (int(row_counts.get("train", 0)) + int(row_counts.get("validation", 0)))
    if monitor is not None and hasattr(monitor, "budget_bytes"):
        from atlas_resources import memory_snapshot
        current = memory_snapshot()
        if max(current["rss"], current["private"]) + resident_row_allowance > monitor.budget_bytes:
            raise MemoryError("XGBoost's 32-byte-per-row planning allowance for resident metadata/gradients "
                              "already exceeds the remaining RAM budget. External disk pages do not remove "
                              "this requirement; use a larger-memory machine or a smaller experiment.")
    cache_dir = Path(output_dir).resolve() / "xgboost_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_token = uuid.uuid4().hex

    class DiskIterator(xgb.DataIter):
        def __init__(self, split: str):
            self.split = split
            self.iterator = None
            self.current = None
            self.prefix = str(cache_dir / f"{run_token}-{split}")
            super().__init__(cache_prefix=self.prefix, release_data=True, on_host=False)

        def reset(self) -> None:
            self.current = None
            self.iterator = iter(_xy_batches(dataset, preprocessor, self.split, batch_rows, monitor))

        def next(self, input_data: Any) -> bool:
            if self.iterator is None:
                self.reset()
            try:
                self.current = next(self.iterator)
            except StopIteration:
                self.current = None
                return False
            x, y = self.current
            input_data(data=x, label=y)
            _check(monitor, f"xgboost {self.split} external page")
            return True

    class MemoryCallback(xgb.callback.TrainingCallback):
        def before_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            _check(monitor, f"xgboost round {epoch + 1} before")
            return False

        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            _check(monitor, f"xgboost round {epoch + 1} after")
            return False

    nthread = max(1, min(2, os.cpu_count() or 1))
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
              "device": "cpu", "max_bin": 256, "max_depth": 4, "eta": 0.03,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
              "lambda": 1.0, "alpha": 0.0, "seed": int(seed) % (2**32), "nthread": nthread}
    train_iterator, validation_iterator = DiskIterator("train"), DiskIterator("validation")
    train_matrix = validation_matrix = None
    try:
        _check(monitor, "xgboost external matrix construction")
        train_matrix = xgb.ExtMemQuantileDMatrix(train_iterator, max_bin=256, nthread=nthread)
        validation_matrix = xgb.ExtMemQuantileDMatrix(validation_iterator, max_bin=256,
                                                      ref=train_matrix, nthread=nthread)
        if train_matrix.num_row() == 0 or validation_matrix.num_row() == 0:
            raise ValueError("XGBoost requires nonempty train and validation splits")
        _check(monitor, "xgboost external matrices complete")
        history: dict[str, Any] = {}
        booster = xgb.train(params, train_matrix, num_boost_round=rounds,
                             evals=[(validation_matrix, "validation")],
                             early_stopping_rounds=40, evals_result=history,
                             verbose_eval=False, callbacks=[MemoryCallback()])
        cache_bytes, cache_files = 0, 0
        for page in cache_dir.glob(f"{run_token}-*"):
            if page.is_file():
                cache_bytes += page.stat().st_size
                cache_files += 1
        if cache_files == 0 or cache_bytes == 0:
            raise RuntimeError("XGBoost did not create nonempty disk cache pages; refusing an unverified run")
        metadata = {"version": xgb.__version__, "matrix": "ExtMemQuantileDMatrix",
                    "device": "cpu", "cache_directory": str(cache_dir), "on_host": False,
                    "cache_files_observed": cache_files, "cache_bytes_observed": cache_bytes,
                    "cache_lifetime": "temporary; released after training",
                    "batch_rows": batch_rows, "parameters": params,
                    "requested_rounds": rounds, "early_stopping_rounds": 40,
                    "best_iteration": int(booster.best_iteration),
                    "validation_history": history,
                    "resident_row_planning_allowance_bytes": resident_row_allowance,
                    "memory_caveat": "per-row labels/gradients/metadata still require RAM; resource guard applies"}
        return booster, metadata
    finally:
        # Release matrix handles before their iterator references, including Windows mappings.
        validation_matrix = None
        train_matrix = None
        train_iterator.current = validation_iterator.current = None
        train_iterator.iterator = validation_iterator.iterator = None
        gc.collect()


@dataclass
class StreamingPredictor:
    kind: str
    name: str
    preprocessor: StreamingPreprocessor
    model: Any

    @property
    def iteration_range(self) -> tuple[int, int]:
        return (0, int(self.model.best_iteration) + 1) if self.kind == "xgboost" else (0, 0)

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.preprocessor.transform(frame)
        if self.kind == "logistic":
            return self.model.predict_proba(x)[:, 1]
        if self.kind == "xgboost":
            return self.model.inplace_predict(x, iteration_range=self.iteration_range)
        if self.kind == "mlp":
            return self.model.predict_proba(x)
        raise ValueError(f"Unknown streaming model: {self.kind}")


def validate_incremental_logistic_reference() -> dict[str, Any]:
    """Small deterministic solver check, never a materialized production sample.

    This checks numerical behavior on known logistic data; it does not establish
    equivalence on market data. Actual training/validation losses are saved too.
    """
    rng = np.random.default_rng(937)
    x = rng.normal(size=(4096, 8))
    probability = 1.0 / (1.0 + np.exp(-(x @ np.asarray([1.2, -0.9, 0.6, 0.0, -0.4, 0.2, 0.0, 0.0]) - 0.3)))
    y = (rng.random(len(x)) < probability).astype(np.int32)
    x_train, x_test = x[:3072], x[3072:]
    y_train, y_test = y[:3072], y[3072:]
    baseline = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(x_train, y_train)
    incremental = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / len(x_train),
                                learning_rate="invscaling", eta0=0.05, power_t=0.25,
                                average=True, shuffle=False, random_state=937, n_jobs=1)
    for _ in range(40):
        # The full fixture has a fixed small bound, independent of Atlas size.
        for start in range(0, len(x_train), 128):
            indices = np.arange(start, min(start + 128, len(x_train)))
            rng.shuffle(indices)
            incremental.partial_fit(x_train[indices], y_train[indices], classes=np.asarray([0, 1]))
    p_reference = baseline.predict_proba(x_test)[:, 1]
    p_incremental = incremental.predict_proba(x_test)[:, 1]

    def ece(p: np.ndarray) -> float:
        bins = np.minimum((p * 10).astype(int), 9)
        return float(sum(abs(float(y_test[bins == i].sum()) - float(p[bins == i].sum()))
                         for i in range(10)) / len(y_test))

    baseline_metrics = {"log_loss": float(log_loss(y_test, p_reference)),
                        "brier": float(brier_score_loss(y_test, p_reference)), "ece_10_bins": ece(p_reference)}
    incremental_metrics = {"log_loss": float(log_loss(y_test, p_incremental)),
                           "brier": float(brier_score_loss(y_test, p_incremental)), "ece_10_bins": ece(p_incremental)}
    differences = {name: incremental_metrics[name] - baseline_metrics[name] for name in baseline_metrics}
    probability_mae = float(np.abs(p_reference - p_incremental).mean())
    passed = (probability_mae < 0.05 and differences["log_loss"] < 0.02
              and differences["brier"] < 0.01 and differences["ece_10_bins"] < 0.03)
    return {"purpose": "deterministic bounded synthetic solver check; not market-data equivalence",
            "seed": 937, "training_rows": 3072, "evaluation_rows": 1024,
            "features": 8, "epochs": 40, "batch_rows": 128,
            "reference": baseline_metrics, "incremental": incremental_metrics,
            "incremental_minus_reference": differences, "probability_mean_absolute_difference": probability_mae,
            "acceptance": {"probability_mae_lt": 0.05, "log_loss_increase_lt": 0.02,
                           "brier_increase_lt": 0.01, "ece_increase_lt": 0.03},
            "passed": bool(passed)}


def validate_logistic_on_bounded_sample(dataset: Any, preprocessor: StreamingPreprocessor, *,
                                         epochs: int, batch_rows: int, sample_rows: int = 4096,
                                         seed: int = 42, monitor: Any = None) -> dict[str, Any]:
    """Compare solvers on the same capped real train/validation samples.

    Samples are deterministic prefixes, not representative random samples. Both
    use the actual training-only transform; this isolates the solver change from
    imputation changes. Results are diagnostic, never used to tune on test data.
    """
    sample_rows = _positive_int(sample_rows, "sample_rows")

    def sample(split: str) -> tuple[np.ndarray, np.ndarray]:
        xs, ys, remaining = [], [], sample_rows
        for x, y in _xy_batches(dataset, preprocessor, split, min(batch_rows, sample_rows), monitor):
            take = min(len(y), remaining)
            xs.append(x[:take].copy())
            ys.append(y[:take].copy())
            remaining -= take
            if remaining == 0:
                break
        if not xs:
            return np.empty((0, len(preprocessor.get_feature_names_out())), dtype=np.float32), np.empty(0, dtype=int)
        return np.concatenate(xs), np.concatenate(ys)

    x_train, y_train = sample("train")
    x_val, y_val = sample("validation")
    result: dict[str, Any] = {"sample_cap_per_split": sample_rows,
                              "training_rows": len(y_train), "validation_rows": len(y_val),
                              "selection": "deterministic split prefixes; not necessarily representative",
                              "epochs": epochs, "batch_rows": batch_rows,
                              "scope": "same bounded data and train-only transform; solver comparison only",
                              "is_equivalence_claim": False}
    if len(np.unique(y_train)) != 2 or len(y_val) == 0:
        return {**result, "status": "insufficient_sample_support"}
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        baseline = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(x_train, y_train)
    incremental = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / len(y_train),
                                learning_rate="invscaling", eta0=0.05, power_t=0.25,
                                average=True, shuffle=False, random_state=int(seed) % (2**32), n_jobs=1)
    rng = np.random.default_rng(int(seed) % (2**32))
    for _ in range(epochs):
        for start in range(0, len(y_train), batch_rows):
            indices = np.arange(start, min(start + batch_rows, len(y_train)))
            rng.shuffle(indices)
            incremental.partial_fit(x_train[indices], y_train[indices], classes=np.asarray([0, 1]))
        _check(monitor, "logistic bounded solver validation")
    p_reference = baseline.predict_proba(x_val)[:, 1]
    p_incremental = incremental.predict_proba(x_val)[:, 1]

    def metrics(p: np.ndarray) -> dict[str, float]:
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(abs(float(y_val[bins == i].sum()) - float(p[bins == i].sum())) for i in range(10)) / len(y_val)
        return {"log_loss": float(log_loss(y_val, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y_val, p)), "ece_10_bins": float(ece)}

    reference_metrics, incremental_metrics = metrics(p_reference), metrics(p_incremental)
    return {**result, "status": "diagnostic_complete", "reference": reference_metrics,
            "incremental": incremental_metrics,
            "incremental_minus_reference": {key: incremental_metrics[key] - reference_metrics[key]
                                               for key in reference_metrics},
            "probability_mean_absolute_difference": float(np.abs(p_reference - p_incremental).mean()),
            "note": "Differences are recorded without claiming optimizer or calibration equivalence; "
                    "production validation metrics determine model selection."}


def train_streaming_models(dataset: Any, experiment: Any, output_dir: Path, *,
                           feature_set: str = "observable", batch_rows: int = 4096,
                           logistic_epochs: int = 10, xgb_rounds: int = 300,
                           monitor: Any = None) -> dict[str, StreamingPredictor]:
    """Train all three comparators; every production data scan is bounded."""
    batch_rows = _positive_int(batch_rows, "batch_rows")
    _positive_int(logistic_epochs, "logistic_epochs")
    _positive_int(xgb_rounds, "xgb_rounds")
    require_external_memory_xgboost()  # Fail before lengthy preprocessing/training.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _check(monitor, "streaming training start")
    reference = validate_incremental_logistic_reference()
    (output_dir / f"logistic_solver_validation_{feature_set}.json").write_text(
        json.dumps(reference, indent=2, allow_nan=False), encoding="utf-8")
    if not reference["passed"]:
        raise RuntimeError("Incremental logistic solver failed the bounded reference validation; see report")
    numeric, categorical = feature_columns(feature_set)
    preprocessor = StreamingPreprocessor(numeric, categorical).fit(dataset, batch_rows=batch_rows, monitor=monitor)
    if np.any(preprocessor.class_counts_ == 0):
        raise ValueError("Training requires both profitable classes; generate more structures")
    joblib.dump(preprocessor, output_dir / f"streaming_preprocessor_{feature_set}.joblib", compress=3)
    bounded_comparison = validate_logistic_on_bounded_sample(
        dataset, preprocessor, epochs=logistic_epochs, batch_rows=batch_rows,
        seed=experiment.base_seed, monitor=monitor)
    (output_dir / f"logistic_data_validation_{feature_set}.json").write_text(
        json.dumps(bounded_comparison, indent=2, allow_nan=False), encoding="utf-8")
    logistic = train_incremental_logistic(dataset, preprocessor, seed=experiment.base_seed,
                                           epochs=logistic_epochs, batch_rows=batch_rows, monitor=monitor)
    joblib.dump(logistic, output_dir / f"streaming_logistic_{feature_set}.joblib", compress=3)
    xgboost, xgboost_metadata = train_external_xgboost(dataset, preprocessor, output_dir=output_dir,
                                                      seed=experiment.base_seed, rounds=xgb_rounds,
                                                      batch_rows=batch_rows, monitor=monitor)
    xgboost.save_model(output_dir / f"streaming_xgboost_{feature_set}.json")
    mlp = StreamingMLP(len(preprocessor.get_feature_names_out()), experiment)
    mlp.fit(dataset, preprocessor, batch_rows=batch_rows, monitor=monitor)
    mlp.save(output_dir / f"streaming_mlp_{feature_set}.pt")
    metadata = {"version": STREAMING_TRAINING_VERSION, "feature_set": feature_set,
                "batch_rows": batch_rows, "preprocessing": preprocessor.metadata(),
                "logistic": {"estimator": "SGDClassifier.partial_fit", "loss": "log_loss",
                              "class_weight": None, "alpha": float(logistic.alpha),
                              "learning_rate": "invscaling", "eta0": 0.05, "power_t": 0.25,
                              "average": True, "shuffle": "seeded within each disk batch; disk order fixed",
                              "epochs": logistic_epochs, "best_epoch": logistic.streaming_best_epoch_,
                              "selection": "lowest validation log loss across requested epochs",
                              "validation_history": logistic.streaming_history_,
                              "reference_validation": reference, "bounded_data_validation": bounded_comparison},
                "xgboost": xgboost_metadata,
                "mlp": {"device": "cpu", "threads": mlp.threads, "batch_size": mlp.batch_size,
                        "learning_rate": mlp.learning_rate, "patience": mlp.patience,
                        "epochs_requested": mlp.epochs, "epochs_trained": mlp.epochs_trained,
                        "best_epoch": mlp.best_epoch, "best_validation_loss": mlp.best_validation_loss,
                        "validation_history": mlp.history_},
                "procedure_changes_from_legacy": ["running mean replaces median imputation",
                    "fixed missing indicators and bounded categorical vocabulary",
                    "averaged incremental SGD replaces full-data lbfgs logistic regression",
                    "CPU neural network shuffles within disk batches and streams validation",
                    "CPU XGBoost ExtMemQuantileDMatrix with disk cache; requested rounds configurable"],
                "test_split_used_for_training": False}
    (output_dir / f"streaming_training_{feature_set}.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    _check(monitor, "streaming models complete")
    return {"logistic": StreamingPredictor("logistic", "Incremental Logistic Regression", preprocessor, logistic),
            "xgboost": StreamingPredictor("xgboost", "XGBoost", preprocessor, xgboost),
            "mlp": StreamingPredictor("mlp", "PyTorch MLP", preprocessor, mlp)}
