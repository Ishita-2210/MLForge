from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CLASS_METRICS = {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc"}
REG_METRICS = {"mae", "rmse", "r2", "median_absolute_error"}
MODELS = ["baseline", "linear", "tree", "forest", "hist_gradient"]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(default="Untitled experiment", min_length=1, max_length=100)
    dataset_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    target: str = Field(min_length=1)
    task: Literal["classification", "regression"]
    features: list[str] = Field(min_length=1, max_length=30)
    models: list[Literal["baseline", "linear", "tree", "forest", "hist_gradient"]] = Field(
        default_factory=lambda: ["baseline", "linear", "tree", "forest"], min_length=1, max_length=5
    )
    primary_metric: str = "f1_macro"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    split: Literal["random", "group", "time"] = "random"
    validation_fraction: float = Field(default=0.2, ge=0.1, le=0.35)
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.35)
    group_column: str | None = None
    time_column: str | None = None
    time_gap: int = Field(default=0, ge=0, le=100)
    cv_folds: Literal[0, 3, 5] = 0
    tune: bool = False
    numeric_imputation: Literal["median", "mean"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    time_budget_s: int = Field(default=120, ge=5, le=180)

    @model_validator(mode="after")
    def coherent(self) -> ExperimentConfig:
        if self.primary_metric not in (
            CLASS_METRICS if self.task == "classification" else REG_METRICS
        ):
            raise ValueError("Select a metric appropriate to the chosen task")
        if self.target in self.features or len(self.features) != len(set(self.features)):
            raise ValueError("Features must be unique and must exclude the target")
        if len(self.models) != len(set(self.models)):
            raise ValueError("Candidate models must be unique")
        if "baseline" not in self.models:
            self.models = ["baseline", *self.models]
        if len(self.models) > 5:
            raise ValueError("At most five algorithms, including the baseline")
        if self.validation_fraction + self.test_fraction > 0.6:
            raise ValueError("Reserve at least 40% for training")
        key = (
            self.group_column
            if self.split == "group"
            else self.time_column
            if self.split == "time"
            else None
        )
        if self.split != "random" and not key:
            raise ValueError("Select the required group or time column")
        if key and (key in self.features or key == self.target):
            raise ValueError("Exclude the split key from features and target")
        fits = (len(self.models) - 1) * (2 if self.tune else 1) + 1
        if fits * (self.cv_folds + 1 if self.cv_folds else 1) > 40:
            raise ValueError(
                "This configuration exceeds the 40-fit CPU budget; reduce models, tuning or folds"
            )
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=256)


class PredictionResponse(BaseModel):
    model_id: str
    model_version: int
    task: str
    predictions: list[Any]
    probabilities: list[list[float]] | None = None
    classes: list[str] | None = None
    duration_ms: float
    drift: dict[str, Any]
