from __future__ import annotations

import numpy as np
from sklearn import metrics as skm
from sklearn.metrics import get_scorer

SCORERS = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "median_absolute_error": "neg_median_absolute_error",
    "roc_auc": "roc_auc_ovr_weighted",
}


def scorer(metric: str):
    return get_scorer(SCORERS.get(metric, metric))


def selection_value(report: dict, metric: str) -> float:
    value = report.get(metric)
    if value is None or not np.isfinite(value):
        raise ValueError(f"{metric} is unavailable for this split; choose a different metric")
    return -float(value) if metric in ("mae", "rmse", "median_absolute_error") else float(value)


def evaluate(pipeline, x, y, task: str) -> dict:
    prediction = pipeline.predict(x)
    if task == "regression":
        residual = np.asarray(y, dtype=float) - prediction
        return {
            "mae": float(skm.mean_absolute_error(y, prediction)),
            "rmse": float(skm.root_mean_squared_error(y, prediction)),
            "r2": float(skm.r2_score(y, prediction)),
            "median_absolute_error": float(skm.median_absolute_error(y, prediction)),
            "residuals": [
                {"row": int(i), "actual": float(a), "prediction": float(p), "residual": float(r)}
                for i, a, p, r in list(zip(x.index, y, prediction, residual, strict=False))[:100]
            ],
            "largest_errors": [
                {
                    "row": int(x.index[i]),
                    "actual": float(y.iloc[i]),
                    "prediction": float(prediction[i]),
                    "absolute_error": float(abs(residual[i])),
                }
                for i in np.argsort(np.abs(residual))[-10:][::-1]
            ],
        }
    classes = [str(c) for c in pipeline.classes_]
    report = {
        "accuracy": float(skm.accuracy_score(y, prediction)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y, prediction)),
        "f1_macro": float(
            skm.f1_score(y, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            skm.f1_score(y, prediction, labels=classes, average="weighted", zero_division=0)
        ),
        "classes": classes,
        "classification_report": skm.classification_report(
            y, prediction, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": skm.confusion_matrix(y, prediction, labels=classes).tolist(),
        "roc_auc": None,
        "average_precision": None,
        "calibration": [],
        "brier_score": None,
    }
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(x)
        if set(y) == set(classes):
            try:
                if len(classes) == 2:
                    binary = np.asarray(y) == classes[1]
                    report["roc_auc"] = float(skm.roc_auc_score(binary, probabilities[:, 1]))
                    report["average_precision"] = float(
                        skm.average_precision_score(binary, probabilities[:, 1])
                    )
                    report["brier_score"] = float(skm.brier_score_loss(binary, probabilities[:, 1]))
                else:
                    report["roc_auc"] = float(
                        skm.roc_auc_score(
                            y, probabilities, labels=classes, multi_class="ovr", average="weighted"
                        )
                    )
            except ValueError:
                pass  # Metric stays explicitly null when its mathematical preconditions fail.
        confidence = probabilities.max(axis=1)
        correct = np.asarray(prediction) == np.asarray(y)
        bin_ids = np.minimum(9, np.floor(confidence * 10).astype(int))
        for number in range(10):
            low = number / 10
            mask = bin_ids == number
            if np.any(mask):
                report["calibration"].append(
                    {
                        "bin_start": round(float(low), 1),
                        "count": int(mask.sum()),
                        "mean_confidence": float(confidence[mask].mean()),
                        "observed_accuracy": float(correct[mask].mean()),
                    }
                )
    errors = []
    for i, actual, pred in zip(x.index, y, prediction, strict=False):
        if actual != pred and len(errors) < 20:
            kind = (
                "misclassification"
                if len(classes) != 2
                else "false_positive"
                if pred == classes[1]
                else "false_negative"
            )
            errors.append(
                {"row": int(i), "actual": str(actual), "prediction": str(pred), "kind": kind}
            )
    report["errors"] = errors
    return report
