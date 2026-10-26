from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from mlforge.config import Settings
from mlforge.errors import ForgeError

MISSING = {"", "na", "n/a", "null", "none", "nan"}


@dataclass
class Dataset:
    frame: pd.DataFrame
    schema: list[dict]
    profile: dict
    fingerprint: str
    canonical: bytes


def _number(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def inspect_csv(raw: bytes, settings: Settings) -> Dataset:
    if not raw or not raw.strip():
        raise ForgeError("The CSV is empty")
    if len(raw) > settings.max_bytes:
        raise ForgeError("CSV exceeds the 8 MiB size limit", "too_large", 413)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ForgeError("Save the CSV using UTF-8 encoding") from exc
    if "\x00" in text:
        raise ForgeError("CSV contains binary/null bytes")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
        columns = [c.strip() for c in header]
        if not columns or any(not c or len(c) > 100 for c in columns):
            raise ForgeError("A nonempty header (at most 100 characters per column) is required")
        if len(columns) > settings.max_columns:
            raise ForgeError(f"CSV exceeds {settings.max_columns} columns")
        if len(columns) != len(set(columns)):
            raise ForgeError("Duplicate column names are not supported")
        if all(_number(c) is not None for c in columns):
            raise ForgeError("The first row looks numeric; add explicit column headers")
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(columns):
                raise ForgeError(f"Malformed CSV row {line}: expected {len(columns)} fields")
            if any(len(cell) > settings.max_cell_chars for cell in row):
                raise ForgeError(f"Row {line} exceeds the per-cell text limit")
            if len(rows) >= settings.max_rows:
                raise ForgeError(f"CSV exceeds the {settings.max_rows}-row limit")
            values = [None if c.strip().lower() in MISSING else c.strip() for c in row]
            for v in values:
                if v is not None and _number(v) is not None and not math.isfinite(float(v)):
                    raise ForgeError(f"Row {line} contains an infinite numeric value")
                if v is not None and _number(v) is not None and abs(float(v)) > 1e12:
                    raise ForgeError(f"Row {line} exceeds the numeric magnitude limit (1e12)")
            rows.append(values)
    except (csv.Error, StopIteration) as exc:
        raise ForgeError("Malformed CSV; check quoting and the header") from exc
    if not rows:
        raise ForgeError("CSV needs at least one data row")
    schema, normalized = [], {}
    warnings = []
    for i, name in enumerate(columns):
        values = [r[i] for r in rows]
        present = [v for v in values if v is not None]
        unique = len(set(present))
        if present and all(v.lower() in ("true", "false") for v in present):
            dtype = "boolean"
            values = [v.lower() == "true" if v is not None else None for v in values]
        elif present and all(_number(v) is not None for v in present):
            dtype = "number"
            values = [float(v) if v is not None else np.nan for v in values]
        else:
            dtype = "string"
        role = (
            "numeric" if dtype == "number" else "boolean" if dtype == "boolean" else "categorical"
        )
        if not present:
            role = "empty"
        elif unique == 1:
            role = "constant"
        elif re.search(r"(^id$|_id$|^uuid$)", name, re.I) or (
            unique / len(rows) >= 0.98
            and (dtype == "number" and all(float(v).is_integer() for v in present))
        ):
            role = "identifier"
        elif dtype == "string" and all(re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in present):
            dates = pd.to_datetime(pd.Series(present), errors="coerce", utc=True)
            if dates.notna().all():
                role = "date-like"
        if role in ("empty", "constant", "identifier", "date-like"):
            warnings.append(f"{name}: {role} column; review its use as a feature")
        if dtype == "string" and unique > 50:
            warnings.append(
                f"{name}: high cardinality ({unique}); encoding will group infrequent categories"
            )
        if (
            dtype == "string"
            and present
            and sum(_number(v) is not None for v in present) / len(present) > 0.8
        ):
            warnings.append(
                f"{name}: mixed numeric/text values; treated as categorical, not silently coerced"
            )
        schema.append(
            {
                "name": name,
                "type": dtype,
                "role": role,
                "missing": len(rows) - len(present),
                "missing_pct": round(100 * (len(rows) - len(present)) / len(rows), 3),
                "unique": unique,
            }
        )
        normalized[name] = values
    frame = pd.DataFrame(normalized)
    canonical = frame.to_csv(index=False, lineterminator="\n").encode()
    if len(canonical) > settings.max_bytes:
        raise ForgeError(
            "Normalized CSV exceeds the storage byte limit; select fewer rows or columns",
            "too_large",
            413,
        )
    signature = json.dumps([(c["name"], c["type"]) for c in schema]).encode()
    fingerprint = hashlib.sha256(signature + b"\n" + canonical).hexdigest()
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate rows retained; use groups if related rows could leak across splits"
        )
    profile = {
        "rows": len(frame),
        "columns": len(columns),
        "duplicates": duplicates,
        "missing_cells": int(frame.isna().sum().sum()),
        "types": dict(Counter(c["role"] for c in schema)),
        "warnings": warnings,
        "actions": [
            "Headers/cells trimmed; UTF-8 BOM accepted; empty/NA/N/A/null/None/NaN treated as missing; no rows removed"
        ],
        "numeric": {},
        "categorical": {},
    }
    for spec in schema:
        s = frame[spec["name"]]
        if spec["type"] == "number":
            valid = s.dropna()
            stats = {
                k: (float(v) if pd.notna(v) and np.isfinite(v) else None)
                for k, v in valid.describe().items()
            }
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            stats["outlier_iqr_count"] = (
                int(((valid < q1 - 1.5 * (q3 - q1)) | (valid > q3 + 1.5 * (q3 - q1))).sum())
                if len(valid)
                else 0
            )
            profile["numeric"][spec["name"]] = stats
        else:
            counts = s.dropna().astype(str).value_counts().head(20)
            profile["categorical"][spec["name"]] = {
                "top_values": {str(k): int(v) for k, v in counts.items()},
                "unique": spec["unique"],
                "truncated": spec["unique"] > 20,
            }
    return Dataset(frame, schema, profile, fingerprint, canonical)


def sample_csv(task: str, seed: int = 42, rows: int = 600) -> bytes:
    rng = np.random.default_rng(seed)
    if task == "classification":
        x, y = make_classification(
            n_samples=rows,
            n_features=6,
            n_informative=4,
            n_redundant=1,
            n_classes=3,
            class_sep=1.5,
            random_state=seed,
        )
        y = np.array(["Class A", "Class B", "Class C"])[y]
    elif task == "regression":
        x, y = make_regression(
            n_samples=rows, n_features=6, n_informative=5, noise=18, random_state=seed
        )
    else:
        raise ForgeError("Choose classification or regression")
    frame = pd.DataFrame(x, columns=[f"signal_{i + 1}" for i in range(6)])
    frame["segment"] = rng.choice(["North", "South", "West"], rows)
    frame["is_repeat"] = rng.choice([True, False], rows)
    frame["record_id"] = np.arange(rows)
    frame["target"] = y
    for name in ["signal_1", "segment"]:
        frame.loc[rng.random(rows) < 0.04, name] = None
    return frame.to_csv(index=False, lineterminator="\n").encode()
