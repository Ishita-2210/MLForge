from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    max_bytes: int = 8 * 1024 * 1024
    max_rows: int = 10000
    max_columns: int = 60
    max_features: int = 30
    max_cell_chars: int = 1024
    max_predict_rows: int = 256
    max_request_bytes: int = 1024 * 1024
    max_logs: int = 1000
    retention_days: int = 30
    max_datasets: int | None = None
    max_experiments: int | None = None
    max_models: int | None = None

    @classmethod
    def default(cls) -> Settings:
        root = Path(__file__).resolve().parents[1]
        return cls(Path(os.environ.get("MLFORGE_HOME", root / ".mlforge")).resolve())

    def prepare(self) -> None:
        for folder in ("datasets", "experiments", "models"):
            (self.home / folder).mkdir(parents=True, exist_ok=True)
