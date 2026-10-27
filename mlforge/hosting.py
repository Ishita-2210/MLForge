"""Explicit temporary-workspace policy for the public Streamlit demo."""

from __future__ import annotations

import os
from pathlib import Path

from mlforge.config import Settings
from mlforge.storage import identifier


def hosted_demo() -> bool:
    return os.environ.get("MLFORGE_HOSTED_DEMO") == "1"


def hosted_settings(root: Path, session: str) -> Settings:
    return Settings(
        home=root.resolve() / "sessions" / identifier(session),
        max_bytes=2 * 1024 * 1024,
        max_rows=2000,
        max_columns=30,
        max_features=20,
        max_datasets=5,
        max_experiments=8,
        max_models=8,
    )
