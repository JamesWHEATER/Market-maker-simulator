"""Bounded, structure-clustered evaluation for the Atlas disk pipeline.

Only one simulation run is retained for interval conflict resolution. Predictions
are computed in smaller batches. Every aggregate is fixed-size; neither run IDs
nor structure-level observations are collected over the full experiment.

ROC-AUC/AP use a fixed probability histogram (explicitly approximate). Economic
means, paired differences, structure-clustered Student-t intervals, Brier scores,
calibration and threshold confusion counts are streamed without approximation.
Medians are deliberately unavailable rather than silently sampled.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import ai_pattern_research_pipeline as ai


RANK_HISTOGRAM_BINS = 4096


@dataclass
class OnlineMoments:
    """Welford moments, including zero observations, in constant space."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("economic inference requires finite values")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def mean_ci(self) -> tuple[float, float, float]:
        if self.count == 0:
            return math.nan, math.nan, math.nan
        if self.count == 1:
            return self.mean, math.nan, math.nan
        sem = math.sqrt(max(self.m2, 0.0) / (self.count - 1) / self.count)
        width = float(stats.t.ppf(0.975, self.count - 1)) * sem
        return self.mean, self.mean - width, self.mean + width

    def positive_pvalue(self) -> float:
        if self.count < 2:
            return 1.0
        if self.m2 <= 0.0:
            return 0.0 if self.mean > 0.0 else 1.0
        sem = math.sqrt(self.m2 / (self.count - 1) / self.count)
        result = float(stats.t.sf(self.mean / sem, self.count - 1))
        return result if math.isfinite(result) else 1.0


class StructureMoments:
    """Average contiguous repeated seeds before adding an inference unit."""

    def __init__(self) -> None:
        self.moments = OnlineMoments()
        self.structure: str | None = None
        self.structure_index: int | None = None
        self.total = 0.0
        self.seeds = 0
        self.finished = False

    def add(self, summary: Mapping[str, Any], value: float) -> None:
        if self.finished:
            raise RuntimeError("cannot append after finishing structure inference")
        structure = str(summary["structure_id"])
        index = summary.get("structure_index")
        index = int(index) if index is not None else None
        if not math.isfinite(value):
            raise ValueError("economic inference requires finite values")
        if self.structure is not None and structure != self.structure:
            if index is not None and self.structure_index is not None and index <= self.structure_index:
                raise ValueError("runs must arrive in contiguous increasing structure order")
            self.moments.add(self.total / self.seeds)
            self.total = 0.0
            self.seeds = 0
        self.structure, self.structure_index = structure, index
        self.total += value
        self.seeds += 1

    def finish(self) -> OnlineMoments:
        if not self.finished:
            if self.seeds:
                self.moments.add(self.total / self.seeds)
            self.finished = True
        return self.moments


class ClassificationAccumulator:
    """Exact decomposable metrics and fixed-size approximate ranking metrics."""

    def __init__(self, thresholds: Sequence[float], bins: int = RANK_HISTOGRAM_BINS) -> None:
        if bins < 2:
            raise ValueError("ranking histogram requires at least two bins")
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.calibration_count = np.zeros(10, dtype=np.int64)
        self.calibration_positive = np.zeros(10, dtype=np.int64)
        self.calibration_probability = np.zeros(10, dtype=float)
        self.count = 0
        self.positives = 0
        self.squared_error = 0.0
        self.true_positive = np.zeros(len(self.thresholds), dtype=np.int64)
        self.predicted_positive = np.zeros(len(self.thresholds), dtype=np.int64)
        self.correct = np.zeros(len(self.thresholds), dtype=np.int64)

    def add(self, labels: Sequence[int], probabilities: Sequence[float]) -> None:
        y = np.asarray(labels)
        p = np.asarray(probabilities, dtype=float)
        if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
            raise ValueError("labels and probabilities must be matching one-dimensional arrays")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("profitable labels must be binary")
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be finite values in [0, 1]")
        y = y.astype(np.int64)
        self.count += len(y)
        self.positives += int(y.sum())
        self.squared_error += float(np.square(p - y).sum())
        indices = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(indices[y == 1], minlength=self.bins)
        self.negative += np.bincount(indices[y == 0], minlength=self.bins)
        calibration = np.minimum(np.searchsorted(np.linspace(0.0, 1.0, 11), p, side="right") - 1, 9)
        self.calibration_count += np.bincount(calibration, minlength=10)
        self.calibration_positive += np.bincount(calibration[y == 1], minlength=10)
        self.calibration_probability += np.bincount(calibration, weights=p, minlength=10)
        for index, threshold in enumerate(self.thresholds):
            selected = p >= threshold
            self.true_positive[index] += int(np.sum(selected & (y == 1)))
            self.predicted_positive[index] += int(selected.sum())
            self.correct[index] += int(np.sum(selected == y))

    def metrics(self, threshold_index: int = 0) -> dict[str, Any]:
        positives, negatives = self.positives, self.count - self.positives
        auc = average_precision = math.nan
        if positives and negatives:
            negatives_below = np.cumsum(self.negative, dtype=float) - self.negative
            auc = float(np.dot(self.positive, negatives_below + self.negative * 0.5) / positives / negatives)
            positive_above = np.cumsum(self.positive[::-1], dtype=float)
            count_above = np.cumsum((self.positive + self.negative)[::-1], dtype=float)
            precision = np.divide(positive_above, count_above, out=np.zeros_like(positive_above), where=count_above > 0)
            average_precision = float(np.dot(self.positive[::-1], precision) / positives)
        return {
            "roc_auc": auc,
            "average_precision": average_precision,
            "ranking_metric_method": "approximate_fixed_probability_histogram",
            "ranking_histogram_bins": self.bins,
            "brier_score": self.squared_error / self.count if self.count else math.nan,
            "expected_calibration_error": float(np.abs(self.calibration_positive - self.calibration_probability).sum() / self.count) if self.count else math.nan,
            "accuracy_at_threshold": float(self.correct[threshold_index] / self.count) if self.count else math.nan,
            "precision_at_threshold": float(self.true_positive[threshold_index] / max(self.predicted_positive[threshold_index], 1)),
            "recall_at_threshold": float(self.true_positive[threshold_index] / max(positives, 1)),
            "observations": self.count,
        }

    def calibration_table(self) -> pd.DataFrame:
        occupied = self.calibration_count > 0
        return pd.DataFrame({
            "bin_lower": np.arange(10)[occupied] / 10,
            "bin_upper": (np.arange(10)[occupied] + 1) / 10,
            "count": self.calibration_count[occupied],
            "mean_probability": self.calibration_probability[occupied] / self.calibration_count[occupied],
            "observed_success_rate": self.calibration_positive[occupied] / self.calibration_count[occupied],
        })


def _validated_run(rows: pd.DataFrame, summary: Mapping[str, Any]) -> None:
    if not summary.get("run_id") or not summary.get("structure_id"):
        raise ValueError("every evaluation run requires run_id and structure_id, including empty runs")
    if not rows.empty:
        if rows["run_id"].isna().any() or not rows["run_id"].astype(str).eq(str(summary["run_id"])).all():
            raise ValueError("iter_runs yielded observations from another run")
        if "structure_id" not in rows or not rows["structure_id"].astype(str).eq(str(summary["structure_id"])).all():
            raise ValueError("evaluation run/structure mapping differs from pattern rows")
        if not np.isfinite(rows["net_directional_return"].to_numpy(dtype=float)).all():
            raise ValueError("economic evaluation requires finite trade returns")


class EconomicAccumulator:
    def __init__(self) -> None:
        self.units = StructureMoments()
        self.rows = self.candidates = self.trades = self.selected_runs = self.runs = 0
        self.successes = 0.0
        self.return_sum = self.drawdown_sum = 0.0

    def add_run(self, rows: pd.DataFrame, probabilities: np.ndarray, threshold: float,
                summary: Mapping[str, Any], selected: pd.DataFrame | None = None) -> float:
        _validated_run(rows, summary)
        if selected is None:
            selected = ai._select_non_overlapping_trades(rows, probabilities, threshold) if len(rows) else rows
        self.rows += len(rows)
        self.runs += 1
        self.candidates += int(np.sum(probabilities >= threshold))
        self.trades += len(selected)
        self.selected_runs += int(len(selected) > 0)
        total = 0.0
        if len(selected):
            returns = selected.sort_values("entry_index")["net_directional_return"].to_numpy(dtype=float)
            total = float(returns.sum())
            path = np.concatenate(([0.0], np.cumsum(returns)))
            self.drawdown_sum += float((np.maximum.accumulate(path) - path).max())
            self.successes += float(selected["profitable"].sum())
            self.return_sum += total
        self.units.add(summary, total)
        return total

    def metrics(self) -> dict[str, Any]:
        moments = self.units.finish()
        mean, low, high = moments.mean_ci()
        return {
            "candidate_selected_trades": float(self.candidates),
            "selected_trades": float(self.trades),
            "selected_runs": float(self.selected_runs),
            "evaluation_runs": float(self.runs),
            "inference_structures": float(moments.count),
            "selection_rate": self.trades / max(self.rows, 1),
            "selected_success_rate": self.successes / self.trades if self.trades else math.nan,
            "mean_selected_net_return": self.return_sum / self.trades if self.trades else math.nan,
            "median_selected_net_return": math.nan,
            "mean_run_net_return_points": mean,
            "mean_run_net_return_points_ci_low": low,
            "mean_run_net_return_points_ci_high": high,
            "median_run_net_return_points": 0.0 if not self.trades else math.nan,
            "mean_run_max_drawdown_return_points": self.drawdown_sum / self.runs if self.runs else math.nan,
            "median_method": "not_computed_out_of_core",
        }


class PairedAccumulator:
    def __init__(self) -> None:
        self.units = StructureMoments()
        self.runs = self.wins = self.ties = 0

    def add_run(self, summary: Mapping[str, Any], ai_total: float, baseline_total: float) -> None:
        difference = ai_total - baseline_total
        self.runs += 1
        self.wins += int(difference > 0.0)
        self.ties += int(difference == 0.0)
        self.units.add(summary, difference)

    def metrics(self) -> dict[str, float]:
        moments = self.units.finish()
        mean, low, high = moments.mean_ci()
        return {
            "paired_ai_minus_baseline_mean_run_return_points": mean,
            "paired_ai_minus_baseline_return_points_ci_low": low,
            "paired_ai_minus_baseline_return_points_ci_high": high,
            "paired_ai_minus_baseline_one_sided_p_value": moments.positive_pvalue(),
            "paired_ai_outperforms_baseline_run_fraction": self.wins / self.runs if self.runs else math.nan,
            "paired_ai_equals_baseline_run_fraction": self.ties / self.runs if self.runs else math.nan,
            "paired_evaluation_runs": float(self.runs),
            "paired_inference_structures": float(moments.count),
        }


class RunTradeSelector:
    """Cache causal tie ordering once for a run, then scan intervals per threshold.

    This preserves the classical selector, including near-tied probabilities
    straddling a threshold and contradictory directions. State is one run only.
    """
    def __init__(self, rows: pd.DataFrame, probabilities: np.ndarray):
        self.rows = rows.reset_index(drop=True).copy()
        self.probabilities = np.asarray(probabilities, dtype=float)
        if self.probabilities.ndim != 1 or len(self.probabilities) != len(rows):
            raise ValueError("probabilities must match one run")
        if not np.isfinite(self.probabilities).all() or np.any((self.probabilities < 0) | (self.probabilities > 1)):
            raise ValueError("probabilities must be finite values in [0, 1]")
        self.rows["probability"] = self.probabilities
        self.groups = []
        if rows.empty:
            return
        if rows["run_id"].nunique(dropna=False) != 1:
            raise ValueError("RunTradeSelector accepts exactly one run")
        entry, exit_ = (rows[column].to_numpy(dtype=float) for column in ("entry_index", "exit_index"))
        if not np.isfinite(entry).all() or not np.isfinite(exit_).all() or np.any(entry != np.floor(entry)) or np.any(exit_ != np.floor(exit_)) or np.any(entry < 0) or np.any(exit_ < entry):
            raise ValueError("trade intervals must satisfy integer 0 <= entry_index <= exit_index")
        self.exits = exit_.astype(np.int64)
        self.directions = rows["trade_direction"].astype(str).to_numpy()
        for entry_value, group in self.rows.groupby("entry_index", sort=True):
            peak = float(group["probability"].max())
            tied = [int(i) for i in group.index if math.isclose(self.probabilities[i], peak, rel_tol=0.0, abs_tol=1e-12)]
            def priority(index):
                row = self.rows.iloc[index]
                return ai.causal_signal_priority(row["pattern_type"], row.get("available_at_index", 0),
                    row.get("pattern_start_index", 0), row.get("pattern_end_index", 0),
                    row.get("observation_window_length", 0), row.get("core_window_length", 0),
                    row.get("geometry_fit_score", 0))
            if len(tied) > 1:
                tied.sort(key=priority)
            self.groups.append((int(entry_value), peak, tied))

    def select(self, threshold: float) -> pd.DataFrame:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        selected, open_until = [], -1
        for entry, peak, tied in self.groups:
            if entry <= open_until or peak < threshold:
                continue
            eligible = [i for i in tied if self.probabilities[i] >= threshold]
            if not eligible or len({self.directions[i] for i in eligible}) > 1:
                continue
            winner = eligible[0]
            selected.append(winner)
            open_until = int(self.exits[winner])
        return self.rows.iloc[selected]


def threshold_grid(experiment: ai.AIExperimentConfig) -> list[float]:
    count = int(math.floor((experiment.validation_threshold_max - experiment.validation_threshold_min)
                           / experiment.validation_threshold_step + 1e-12)) + 1
    if count > 1001:
        raise ValueError("streaming validation supports at most 1001 pre-registered thresholds")
    return [min(experiment.validation_threshold_max,
                experiment.validation_threshold_min + i * experiment.validation_threshold_step) for i in range(count)]


def choose_validation_threshold(economics: Sequence[EconomicAccumulator], thresholds: Sequence[float],
                                experiment: ai.AIExperimentConfig) -> tuple[int, list[dict[str, Any]]]:
    results = [accumulator.metrics() for accumulator in economics]
    supported = [i for i, result in enumerate(results)
                 if result["selected_trades"] >= experiment.min_validation_trades
                 and result["selected_runs"] >= experiment.min_validation_runs
                 and math.isfinite(result["mean_run_net_return_points"])]
    if not supported:
        raise ValueError("INSUFFICIENT_VALIDATION_SUPPORT: no threshold met the pre-registered minimum "
                         "trade/run counts. Confirmatory test evaluation is disabled.")
    selected = max(supported, key=lambda i: (results[i]["mean_run_net_return_points"],
                                            results[i]["selected_runs"], -thresholds[i]))
    return selected, results


def _predict_run(model: Any, rows: pd.DataFrame, batch_rows: int) -> np.ndarray:
    result = np.empty(len(rows), dtype=float)
    for start in range(0, len(rows), batch_rows):
        stop = min(start + batch_rows, len(rows))
        values = np.asarray(model.predict_frame(rows.iloc[start:stop]), dtype=float)
        if values.ndim != 1 or len(values) != stop - start:
            raise ValueError("predict_frame must return one probability per input row")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("probabilities must be finite values in [0, 1]")
        result[start:stop] = values
    return result


class BoundedExplanationSample:
    """Uniform event sample by the smallest independent random priorities."""

    def __init__(self, limit: int, seed: int) -> None:
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.frame = pd.DataFrame()
        self.priorities = np.empty(0, dtype=float)
        self.population = 0

    def add(self, rows: pd.DataFrame) -> None:
        self.population += len(rows)
        if not self.limit or rows.empty:
            return
        priorities = self.rng.random(len(rows))
        # Keep at most limit candidates from a batch before copying feature data.
        keep = np.argpartition(priorities, self.limit - 1)[:self.limit] if len(rows) > self.limit else np.arange(len(rows))
        priorities = np.concatenate((self.priorities, priorities[keep]))
        frame = pd.concat((self.frame, rows.iloc[keep]), ignore_index=True)
        keep = np.argpartition(priorities, self.limit - 1)[:self.limit] if len(frame) > self.limit else np.arange(len(frame))
        self.priorities = priorities[keep]
        self.frame = frame.iloc[keep].reset_index(drop=True)


def _check(monitor: Any, stage: str) -> None:
    if monitor is not None:
        monitor.check(stage)


def _write_explanations(sample: BoundedExplanationSample, models: Mapping[str, Any], output_dir: Path,
                        feature_set: str, batch_rows: int, monitor: Any) -> None:
    path = output_dir / f"explanation_sample_{feature_set}.csv.gz"
    sample.frame.to_csv(path, index=False, compression="gzip")
    metadata: dict[str, Any] = {
        "sampling_unit": "test_event",
        "sampling_method": "uniform_random_priority_without_replacement",
        "maximum_rows": sample.limit,
        "sample_rows": len(sample.frame),
        "test_event_population": sample.population,
        "interpretation": "descriptive event-level explanation sample; not independent structure inference",
        "models": {},
    }
    for key, wrapper in models.items():
        underlying = getattr(wrapper, "model", None)
        preprocessor = getattr(wrapper, "preprocessor", None)
        if preprocessor is None or underlying is None:
            metadata["models"][key] = "wrapper does not expose explanation interface"
            continue
        names = list(map(str, preprocessor.get_feature_names_out()))
        if hasattr(underlying, "coef_"):
            coefficients = np.asarray(underlying.coef_).reshape(-1)
            pd.DataFrame({"feature": names, "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients)}).sort_values(
                "absolute_coefficient", ascending=False).to_csv(output_dir / f"{key}_importance_{feature_set}.csv", index=False)
            metadata["models"][key] = "incremental logistic coefficients in transformed feature units"
        elif getattr(wrapper, "kind", key) == "xgboost" and len(sample.frame):
            import xgboost as xgb
            absolute_sum = np.zeros(len(names), dtype=float)
            # Bound even a requested large explanation sample by the inference batch size.
            with gzip.open(output_dir / f"{key}_shap_sample_{feature_set}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
                for start in range(0, len(sample.frame), batch_rows):
                    frame = sample.frame.iloc[start:start + batch_rows]
                    transformed = preprocessor.transform(frame)
                    kwargs = {"iteration_range": wrapper.iteration_range} if getattr(wrapper, "iteration_range", None) else {}
                    contributions = np.asarray(underlying.predict(xgb.DMatrix(transformed), pred_contribs=True, **kwargs))
                    if contributions.shape != (len(frame), len(names) + 1):
                        raise ValueError("unexpected XGBoost contribution shape")
                    absolute_sum += np.abs(contributions[:, :-1]).sum(axis=0)
                    output = pd.DataFrame(contributions, columns=names + ["BIAS"])
                    for column in ("run_id", "structure_id", "event_id"):
                        if column in frame:
                            output.insert(0, column, frame[column].to_numpy())
                    output.to_csv(handle, index=False, header=start == 0)
                    _check(monitor, "explanation_batch")
            pd.DataFrame({"feature": names, "mean_absolute_shap_log_odds": absolute_sum / len(sample.frame)}).sort_values(
                "mean_absolute_shap_log_odds", ascending=False).to_csv(output_dir / f"{key}_shap_importance_{feature_set}.csv", index=False)
            metadata["models"][key] = "native tree SHAP on uniform bounded test-event sample; log-odds contributions"
        else:
            metadata["models"][key] = "bounded input sample exported; model-specific attribution not computed"
    (output_dir / f"explanation_metadata_{feature_set}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def evaluate_streaming_models(dataset: Any, models: Mapping[str, Any], experiment: ai.AIExperimentConfig,
                              output_dir: Path | str, *, feature_set: str, batch_rows: int = 4096,
                              monitor: Any = None, explanation_rows: int = 512) -> pd.DataFrame:
    """Freeze supported validation choices before ever opening the test split.

    ``dataset.iter_runs(split)`` must emit whole, bounded runs in contiguous
    structure/seed order, including zero-event runs. Models expose
    ``predict_frame`` and optionally ``name``, ``model`` and ``preprocessor``.
    Output files grow on disk; aggregate state is independent of dataset size.
    """
    if batch_rows < 1 or explanation_rows < 0:
        raise ValueError("batch_rows must be positive and explanation_rows nonnegative")
    if not models:
        raise ValueError("evaluation requires at least one model")
    if any(not key or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in key) for key in models):
        raise ValueError("model keys must be safe file-name components")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = threshold_grid(experiment)
    validation_economics = {key: [EconomicAccumulator() for _ in thresholds] for key in models}
    validation_classification = {key: ClassificationAccumulator(thresholds) for key in models}
    for rows, summary in dataset.iter_runs("validation"):
        _validated_run(rows, summary)
        for key, model in models.items():
            probabilities = _predict_run(model, rows, batch_rows)
            selector = RunTradeSelector(rows, probabilities)
            if len(rows):
                validation_classification[key].add(rows["profitable"].to_numpy(), probabilities)
            for threshold, accumulator in zip(thresholds, validation_economics[key]):
                accumulator.add_run(rows, probabilities, threshold, summary, selector.select(threshold))
        _check(monitor, "evaluate_validation_run")

    # No test stream is opened unless every pre-specified comparator is supported.
    choices: dict[str, float] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    grid_rows: list[dict[str, Any]] = []
    for key, model in models.items():
        index, economic_results = choose_validation_threshold(validation_economics[key], thresholds, experiment)
        choices[key] = thresholds[index]
        classification = validation_classification[key].metrics(index)
        economic = economic_results[index]
        validation_results[key] = {
            "model": getattr(model, "name", key), "model_kind": key, "feature_set": feature_set,
            "validation_selected_threshold": choices[key],
            **{f"validation_{metric}": classification[metric] for metric in ("roc_auc", "average_precision", "brier_score", "expected_calibration_error")},
            "validation_mean_run_net_return_points": economic["mean_run_net_return_points"],
            "validation_selected_trades": economic["selected_trades"],
            "validation_selected_runs": economic["selected_runs"],
            "validation_inference_structures": economic["inference_structures"],
            "validation_support_satisfied": True,
        }
        for threshold, values in zip(thresholds, economic_results):
            grid_rows.append({"model": key, "threshold": threshold, **values})
    def rank(key: str) -> tuple[float, float, float, str]:
        result = validation_results[key]
        auc, brier = result["validation_roc_auc"], result["validation_brier_score"]
        return (-auc if math.isfinite(auc) else math.inf, brier if math.isfinite(brier) else math.inf,
                -result["validation_mean_run_net_return_points"], str(result["model"]))
    primary = min(models, key=rank)
    frozen = {
        "feature_set": feature_set, "primary_model": validation_results[primary]["model"],
        "primary_model_kind": primary, "thresholds": choices, "validation_support_satisfied": True,
        "test_opened_after_selection": True,
        "selection_rule": "highest approximate validation ROC-AUC; lowest exact Brier; highest structure-mean run return; model name",
        "ranking_metric_method": "approximate_fixed_probability_histogram", "ranking_histogram_bins": RANK_HISTOGRAM_BINS,
        "inference_unit": "sampled structure; average repeated seeds, including zero-event runs",
        "median_method": "not_computed_out_of_core",
        "confirmatory_feature_family_bonferroni_factor": 2,
    }
    frozen_path = output_dir / f"primary_model_{feature_set}.json"
    temporary = frozen_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(frozen_path)
    pd.DataFrame(grid_rows).to_csv(output_dir / f"validation_threshold_grid_{feature_set}.csv", index=False)
    del validation_economics, validation_classification, grid_rows

    test_economics = {key: EconomicAccumulator() for key in models}
    test_classification = {key: ClassificationAccumulator([choices[key]]) for key in models}
    paired = {key: PairedAccumulator() for key in models}
    baseline = EconomicAccumulator()
    family_economics = {key: {family: EconomicAccumulator() for family in ("structured", "null")} for key in models}
    family_classification = {key: {family: ClassificationAccumulator([choices[key]]) for family in ("structured", "null")} for key in models}
    seed = experiment.base_seed
    sample = BoundedExplanationSample(explanation_rows, seed)
    # Open a fixed number of streams; each write contains at most one run.
    from contextlib import ExitStack
    with ExitStack() as stack:
        prediction_files = {key: stack.enter_context(gzip.open(output_dir / f"test_predictions_{key}_{feature_set}.csv.gz", "wt", encoding="utf-8", newline="")) for key in models}
        totals_file = stack.enter_context(gzip.open(output_dir / f"test_run_returns_{feature_set}.csv.gz", "wt", encoding="utf-8", newline=""))
        wrote_predictions = {key: False for key in models}
        wrote_totals = False
        for rows, summary in dataset.iter_runs("test"):
            _validated_run(rows, summary)
            ones = np.ones(len(rows), dtype=float)
            baseline_selected = RunTradeSelector(rows, ones).select(0.5)
            baseline_total = baseline.add_run(rows, ones, 0.5, summary, baseline_selected)
            totals: dict[str, Any] = {"run_id": str(summary["run_id"]), "structure_id": str(summary["structure_id"]),
                                      "observations": len(rows), "baseline_return_points": baseline_total}
            family_value = str(summary.get("world_family", rows["world_family"].iloc[0] if len(rows) and "world_family" in rows else ""))
            family = "structured" if family_value == "structured" else "null" if family_value.startswith("null_") else None
            for key, model in models.items():
                probabilities = _predict_run(model, rows, batch_rows)
                selected = RunTradeSelector(rows, probabilities).select(choices[key])
                total = test_economics[key].add_run(rows, probabilities, choices[key], summary, selected)
                paired[key].add_run(summary, total, baseline_total)
                totals[f"{key}_return_points"] = total
                if family is not None:
                    family_economics[key][family].add_run(rows, probabilities, choices[key], summary, selected)
                if len(rows):
                    test_classification[key].add(rows["profitable"].to_numpy(), probabilities)
                    if family is not None:
                        family_classification[key][family].add(rows["profitable"].to_numpy(), probabilities)
                    columns = [column for column in ("run_id", "structure_id", "event_id", "world_family", "pattern_type", "trade_direction", "entry_index", "exit_index", "profitable", "net_directional_return") if column in rows]
                    output = rows[columns].reset_index(drop=True).copy()
                    output["probability"] = probabilities
                    output["candidate_selected"] = (probabilities >= choices[key]).astype(np.int8)
                    output["selected"] = 0
                    output.loc[selected.index, "selected"] = 1
                    output["baseline_selected"] = 0
                    output.loc[baseline_selected.index, "baseline_selected"] = 1
                    output.to_csv(prediction_files[key], index=False, header=not wrote_predictions[key])
                    wrote_predictions[key] = True
            pd.DataFrame([totals]).to_csv(totals_file, index=False, header=not wrote_totals)
            wrote_totals = True
            for start in range(0, len(rows), batch_rows):
                sample.add(rows.iloc[start:start + batch_rows])
            _check(monitor, "evaluate_test_run")
        for key in models:
            if not wrote_predictions[key]:
                pd.DataFrame(columns=["run_id", "structure_id", "event_id", "profitable", "probability", "candidate_selected", "selected", "baseline_selected"]).to_csv(prediction_files[key], index=False)

    if not baseline.runs:
        raise ValueError("test evaluation requires at least one run, including zero-event runs")
    baseline_results = {f"baseline_{key}": value for key, value in baseline.metrics().items()}
    results = []
    for key in models:
        role = key == primary
        result = {**validation_results[key], **test_classification[key].metrics(), **test_economics[key].metrics(),
                  **baseline_results, **paired[key].metrics(), "primary_model_selected_on_validation": role,
                  "test_inference_role": "primary_within_feature_set" if role else "exploratory_comparator"}
        result["confirmatory_p_value_bonferroni"] = min(1.0, 2.0 * result["paired_ai_minus_baseline_one_sided_p_value"]) if role else math.nan
        for family in ("structured", "null"):
            economic = family_economics[key][family].metrics()
            classification = family_classification[key][family].metrics()
            result.update({f"{family}_test_observations": classification["observations"],
                           f"{family}_test_runs": economic["evaluation_runs"], f"{family}_roc_auc": classification["roc_auc"],
                           f"{family}_mean_run_net_return_points": economic["mean_run_net_return_points"],
                           f"{family}_selected_success_rate": economic["selected_success_rate"]})
        test_classification[key].calibration_table().to_csv(output_dir / f"{key}_calibration_{feature_set}.csv", index=False)
        results.append(result)
    _write_explanations(sample, models, output_dir, feature_set, batch_rows, monitor)
    comparison = pd.DataFrame(results)
    comparison.to_csv(output_dir / f"model_comparison_{feature_set}.csv", index=False)
    _check(monitor, "evaluation_complete")
    return comparison
