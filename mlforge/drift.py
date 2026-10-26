from __future__ import annotations

import numpy as np
import pandas as pd


def reference(frame: pd.DataFrame, schema: list[dict]) -> dict:
    output = {}
    by_name = {s["name"]: s for s in schema}
    for name in frame.columns:
        s = frame[name]
        valid = s.dropna()
        missing = float(s.isna().mean())
        if by_name[name]["type"] == "number":
            edges = (
                np.unique(np.quantile(valid.astype(float), np.linspace(0, 1, 6)))
                if len(valid)
                else np.array([0.0])
            )
            counts = np.bincount(
                np.searchsorted(edges, valid.astype(float), side="right"), minlength=len(edges) + 1
            )
            probability = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
            output[name] = {
                "kind": "number",
                "edges": edges.tolist(),
                "probabilities": probability.tolist(),
                "missing_rate": missing,
                "mean": float(valid.mean()) if len(valid) else None,
                "std": float(valid.std(ddof=0)) if len(valid) else None,
            }
        else:
            counts = valid.astype(str).value_counts()
            top = counts.head(24)
            probabilities = {str(k): float(v / max(1, len(valid))) for k, v in top.items()}
            output[name] = {
                "kind": "categorical",
                "probabilities": probabilities,
                "other": max(0.0, 1 - sum(probabilities.values())) if len(valid) else 0.0,
                "missing_rate": missing,
            }
    return output


def compare(frame: pd.DataFrame, baseline: dict) -> dict:
    features = {}
    enough = len(frame) >= 20
    for name, spec in baseline.items():
        s = frame[name]
        valid = s.dropna()
        missing = float(s.isna().mean())
        if spec["kind"] == "number":
            counts = np.bincount(
                np.searchsorted(spec["edges"], valid.astype(float), side="right"),
                minlength=len(spec["edges"]) + 1,
            )
            observed = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
            expected = np.asarray(spec["probabilities"])
            score = float(np.sum((observed - expected) * np.log(observed / expected)))
            result = {
                "metric": "psi",
                "score": score,
                "threshold": 0.2,
                "mean": float(valid.mean()) if len(valid) else None,
                "std": float(valid.std(ddof=0)) if len(valid) else None,
            }
        else:
            values = valid.astype(str)
            frequencies = values.value_counts(normalize=True)
            known = spec["probabilities"]
            unseen = float((~values.isin(known)).mean()) if len(values) else 0.0
            score = 0.5 * (
                sum(abs(float(frequencies.get(k, 0)) - v) for k, v in known.items())
                + abs(unseen - spec["other"])
            )
            result = {
                "metric": "total_variation",
                "score": float(score),
                "threshold": 0.2,
                "unseen_or_tail_rate": unseen,
            }
        result.update(
            missing_rate=missing,
            missing_rate_change=missing - spec["missing_rate"],
            flagged=bool(enough and (score >= 0.2 or abs(missing - spec["missing_rate"]) >= 0.2)),
        )
        features[name] = result
    return {
        "rows": len(frame),
        "status": "diagnostic" if enough else "insufficient_rows",
        "features": features,
        "flagged_features": [k for k, v in features.items() if v["flagged"]],
        "note": "Input shift diagnostic, not evidence of performance degradation. Batches below 20 rows are too small to flag. PSI/TV >=0.2 or missingness change >=0.2 are explicit heuristics, not significance tests.",
    }
