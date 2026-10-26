from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib

from mlforge.errors import ForgeError
from mlforge.storage import identifier
from mlforge.training import environment


def artifact_path(home: Path, kind: str, key: str) -> Path:
    if kind not in ("experiments", "models"):
        raise ForgeError("Invalid artifact kind")
    root = (home / kind).resolve()
    path = root / identifier(key) / "pipeline.joblib"
    if not path.resolve().is_relative_to(root):
        raise ForgeError("Invalid artifact location")
    return path


def write_pipeline(pipeline, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ForgeError("Artifact already exists; create a new model version", "conflict", 409)
    temporary = path.with_suffix(".part")
    joblib.dump(pipeline, temporary, compress=3)
    os.replace(temporary, path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "format_version": 1,
    }


def load_pipeline(path: Path, metadata: dict):
    if not path.exists():
        raise ForgeError("Model artifact is unavailable", "artifact_missing", 409)
    current, saved = environment(), metadata["environment"]
    if (
        metadata.get("format_version") != 1
        or saved["sklearn"] != current["sklearn"]
        or saved["numpy"] != current["numpy"]
        or saved["python"].split(".")[:2] != current["python"].split(".")[:2]
    ):
        raise ForgeError(
            "Artifact environment is incompatible. Use its recorded Python/scikit-learn/NumPy versions or retrain",
            "artifact_incompatible",
            409,
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ForgeError(
            "Artifact integrity check failed; refusing to deserialize", "artifact_integrity", 409
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ForgeError(
            "Artifact could not be loaded; retrain it in this environment",
            "artifact_incompatible",
            409,
        ) from exc
