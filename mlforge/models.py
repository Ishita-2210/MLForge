from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlforge.errors import ForgeError
from mlforge.schemas import ExperimentConfig


def variants(algorithm: str, tune: bool) -> list[dict]:
    if algorithm == "baseline":
        return [{}]
    options = {
        "linear": [{"strength": 1.0}, {"strength": 10.0}],
        "tree": [{"max_depth": 6}, {"max_depth": 10}],
        "forest": [{"max_depth": 8}, {"max_depth": 12}],
        "hist_gradient": [{"learning_rate": 0.08}, {"learning_rate": 0.15}],
    }
    if algorithm not in options:
        raise ForgeError("Unsupported algorithm")
    return options[algorithm] if tune else options[algorithm][:1]


def build_pipeline(
    config: ExperimentConfig, schema: list[dict], algorithm: str, params: dict
) -> Pipeline:
    classification = config.task == "classification"
    if algorithm == "baseline":
        model = (
            DummyClassifier(strategy="most_frequent")
            if classification
            else DummyRegressor(strategy="mean")
        )
    elif algorithm == "linear":
        model = (
            LogisticRegression(
                C=1 / params.get("strength", 1), max_iter=500, random_state=config.seed
            )
            if classification
            else Ridge(alpha=params.get("strength", 1))
        )
    elif algorithm == "tree":
        cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
        model = cls(
            max_depth=params.get("max_depth", 6), min_samples_leaf=3, random_state=config.seed
        )
    elif algorithm == "forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        model = cls(
            n_estimators=48,
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=2,
            n_jobs=1,
            random_state=config.seed,
        )
    elif algorithm == "hist_gradient":
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        model = cls(
            max_iter=60,
            max_leaf_nodes=15,
            learning_rate=params.get("learning_rate", 0.08),
            early_stopping=False,
            random_state=config.seed,
        )
    else:
        raise ForgeError("Unsupported algorithm")
    numeric = [s["name"] for s in schema if s["name"] in config.features and s["type"] == "number"]
    categorical = [c for c in config.features if c not in numeric]
    num_steps = [
        ("impute", SimpleImputer(strategy=config.numeric_imputation, keep_empty_features=True))
    ]
    if algorithm == "linear":
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy=config.categorical_imputation,
                                fill_value="__missing__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=24,
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True),
            ),
            ("model", model),
        ]
    )


def feature_frame(frame, schema: list[dict], names: list[str]):
    """Deterministic type normalization, with no learned values or vocabulary."""
    result = frame[names].copy()
    by_name = {s["name"]: s for s in schema}
    for name in names:
        if by_name[name]["type"] == "number":
            result[name] = result[name].astype(float)
        else:
            result[name] = (
                result[name]
                .map(
                    lambda v: str(v)
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                    else np.nan
                )
                .astype(object)
            )
    return result
