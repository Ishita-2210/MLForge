from __future__ import annotations

import platform
import time
import warnings
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from mlforge.datasets import Dataset
from mlforge.drift import reference
from mlforge.errors import ForgeError
from mlforge.metrics import evaluate, scorer, selection_value
from mlforge.models import build_pipeline, feature_frame, variants
from mlforge.schemas import ExperimentConfig
from mlforge.splitting import build_split


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
        "mlforge": "1.0.0",
    }


def validate_problem(data: Dataset, config: ExperimentConfig) -> tuple[pd.Series, list[str]]:
    frame = data.frame
    keys = [config.target, *config.features]
    keys += [k for k in (config.group_column, config.time_column) if k]
    if any(k not in frame for k in keys):
        raise ForgeError("Target, features and split keys must exist in this dataset")
    y = frame[config.target]
    if y.isna().any():
        raise ForgeError(
            "Target contains missing labels. Correct or explicitly remove those rows in your source CSV before training; MLForge never silently drops them"
        )
    if y.nunique() < 2:
        raise ForgeError("Target must have at least two distinct values")
    notes = list(data.profile["warnings"])
    if config.task == "classification":
        y = y.astype(str)
        if y.nunique() > 20:
            raise ForgeError(
                "Classification supports at most 20 target classes; consider regression for continuous targets"
            )
        if y.value_counts().min() < 10:
            notes.append(
                "Some classes have fewer than ten samples; validation estimates may be unreliable"
            )
    else:
        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            raise ForgeError("Regression requires a finite numeric target")
        y = y.astype(float)
        if not np.isfinite(y).all():
            raise ForgeError("Regression target must be finite")
        if y.nunique() <= 10:
            notes.append(
                "Low-cardinality regression target; check whether classification better matches the problem"
            )
    if len(frame) < 100:
        notes.append("Small dataset: treat holdout and feature importance estimates cautiously")
    encoded = 0
    by_name = {s["name"]: s for s in data.schema}
    for name in config.features:
        spec = by_name[name]
        encoded += 1 if spec["type"] == "number" else max(1, min(24, spec["unique"]))
        if frame[name].astype(str).equals(frame[config.target].astype(str)):
            notes.append(
                f"Possible direct leakage: {name} exactly matches the target. Verify information is available at prediction time"
            )
    if encoded * len(frame) > 1_500_000:
        raise ForgeError(
            "Estimated dense encoding exceeds the 1.5-million-cell CPU/memory budget; select fewer features or rows"
        )
    if config.split == "random":
        notes.append(
            "Random splitting assumes independent rows. For shared entities or time windows, select group/time splitting; heuristics cannot guarantee absence of domain leakage"
        )
    return y, notes


def train(
    data: Dataset,
    config: ExperimentConfig,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
):
    start = time.perf_counter()
    rss_start = psutil.Process().memory_info().rss
    y, notes = validate_problem(data, config)
    split = build_split(data.frame, y, config)
    notes.extend(split.warnings)
    x = feature_frame(data.frame, data.schema, config.features)
    results, successful = [], []

    def check() -> None:
        if cancelled and cancelled():
            raise ForgeError("Experiment cancelled", "cancelled", 409)
        if time.perf_counter() - start > config.time_budget_s:
            raise ForgeError(
                "Experiment exceeded its time budget; reduce models, folds, tuning or dataset size",
                "time_budget",
                408,
            )

    with threadpool_limits(limits=1):
        for algorithm in config.models:
            for parameters in variants(algorithm, config.tune):
                check()
                candidate = {"algorithm": algorithm, "parameters": parameters, "status": "running"}
                t0 = time.perf_counter()
                fit_seconds = 0.0
                try:
                    scores, caught_notes = [], []
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        for a, b in split.folds:
                            check()
                            fold = build_pipeline(config, data.schema, algorithm, parameters)
                            fit_start = time.perf_counter()
                            fold.fit(x.iloc[a], y.iloc[a])
                            fit_seconds += time.perf_counter() - fit_start
                            scores.append(
                                float(scorer(config.primary_metric)(fold, x.iloc[b], y.iloc[b]))
                            )
                        check()
                        pipeline = build_pipeline(config, data.schema, algorithm, parameters)
                        fit_start = time.perf_counter()
                        pipeline.fit(x.iloc[split.train], y.iloc[split.train])
                        fit_seconds += time.perf_counter() - fit_start
                        validation = evaluate(
                            pipeline,
                            x.iloc[split.validation],
                            y.iloc[split.validation],
                            config.task,
                        )
                        caught_notes = sorted({str(w.message)[:300] for w in caught})
                    value = (
                        float(np.mean(scores))
                        if scores
                        else selection_value(validation, config.primary_metric)
                    )
                    if not np.isfinite(value):
                        raise ValueError("Primary selection metric was not finite")
                    candidate.update(
                        status="succeeded",
                        validation=validation,
                        selection_score=value,
                        selection_source="training_cv_mean" if scores else "validation_holdout",
                        cv_scores=scores,
                        warnings=caught_notes,
                        estimator_parameters=pipeline.named_steps["model"].get_params(deep=False),
                    )
                    successful.append((value, pipeline, len(results)))
                except ForgeError:
                    raise
                except Exception as exc:
                    # Persist failures, but never expose exception tracebacks or invent scores.
                    candidate.update(
                        status="failed",
                        error=f"{type(exc).__name__}: candidate could not fit/evaluate. Check feature types, class support and metric preconditions.",
                    )
                candidate["duration_s"] = time.perf_counter() - t0
                candidate["fit_s"] = fit_seconds
                results.append(candidate)
                if progress:
                    progress(
                        {
                            "candidates": results,
                            "progress": f"Finished {algorithm} ({len(results)} candidate attempts)",
                        }
                    )
        check()
        if not successful:
            raise ForgeError(
                "Every candidate failed; inspect the recorded failures", "training_failed", 422
            )
        _, selected, selected_index = max(successful, key=lambda item: item[0])
        # This is the only final-test evaluation, after the selection decision is frozen.
        final_test = evaluate(selected, x.iloc[split.test], y.iloc[split.test], config.task)
        check()
        importance = {
            "method": "validation permutation importance",
            "values": [],
            "note": "Model behavior, not causation. Correlated features can share importance; negative values are valid.",
        }
        encoded_names = selected.named_steps["preprocess"].get_feature_names_out().tolist()
        estimator = selected.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            importance["impurity_encoded_features"] = sorted(
                [
                    {"feature": name, "importance": float(value)}
                    for name, value in zip(
                        encoded_names, estimator.feature_importances_, strict=True
                    )
                ],
                key=lambda r: r["importance"],
                reverse=True,
            )[:50]
            importance["impurity_note"] = (
                "Training impurity importance on encoded columns; can favor high-cardinality features. Prefer validation permutation importance when comparing original features."
            )
        try:
            limit = split.validation[:300]
            perm = permutation_importance(
                selected,
                x.iloc[limit],
                y.iloc[limit],
                scoring=scorer(config.primary_metric),
                n_repeats=3,
                random_state=config.seed,
                n_jobs=1,
            )
            importance["values"] = sorted(
                [
                    {"feature": k, "mean": float(a), "std": float(b)}
                    for k, a, b in zip(
                        config.features, perm.importances_mean, perm.importances_std, strict=False
                    )
                    if np.isfinite(a) and np.isfinite(b)
                ],
                key=lambda r: r["mean"],
                reverse=True,
            )
        except Exception:
            importance["note"] += " Importance unavailable for this split/metric."
        check()
        result = {
            "candidates": results,
            "selected_index": selected_index,
            "selected_algorithm": results[selected_index]["algorithm"],
            "selection_metric": config.primary_metric,
            "selection_source": results[selected_index]["selection_source"],
            "selection_rationale": "Highest permitted validation/CV score; ties use declared algorithm order. Loss metrics are negated for ranking. Final test excluded from selection.",
            "final_test": final_test,
            "final_test_evaluations": 1,
            "feature_importance": importance,
            "warnings": notes,
            "duration_s": time.perf_counter() - start,
            "memory": {
                "rss_start_bytes": rss_start,
                "rss_end_bytes": psutil.Process().memory_info().rss,
                "note": "Process resident memory snapshots, not peak memory or isolated per-model usage",
            },
            "split_indices": {
                "train": split.train.tolist(),
                "validation": split.validation.tolist(),
                "test": split.test.tolist(),
                "purged": split.purged,
            },
            "effective_cv_folds": len(split.folds),
            "environment": environment(),
            "feature_schema": [
                by for name in config.features for by in data.schema if by["name"] == name
            ],
            "reference": reference(x.iloc[split.train], data.schema),
            "training_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
            "preprocessing_features": selected.named_steps["preprocess"]
            .get_feature_names_out()
            .tolist(),
        }
    result["duration_s"] = time.perf_counter() - start
    return selected, result
