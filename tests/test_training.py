import numpy as np
import pandas as pd
import pytest
from conftest import config_for
from pydantic import ValidationError

from mlforge.datasets import inspect_csv, sample_csv
from mlforge.errors import ForgeError
from mlforge.metrics import selection_value
from mlforge.models import feature_frame
from mlforge.schemas import ExperimentConfig
from mlforge.splitting import build_split
from mlforge.training import train, validate_problem


@pytest.mark.parametrize(
    "changes",
    [
        {"features": ["target"]},
        {"features": []},
        {"primary_metric": "rmse"},
        {"validation_fraction": 0.35, "test_fraction": 0.35},
        {"split": "group"},
        {"models": ["alien"]},
        {"models": ["tree", "tree"]},
        {"time_budget_s": 999},
    ],
)
def test_configuration_validation(forge, changes):
    record = forge.sample("classification")
    with pytest.raises(ValidationError):
        config_for(record, **changes)


def test_group_split_and_cv_do_not_share_entities():
    frame = pd.DataFrame({"group": np.repeat(np.arange(30), 4), "value": np.arange(120)})
    y = pd.Series(np.tile(["a", "b"], 60))
    config = ExperimentConfig(
        dataset_id="a" * 32,
        target="target",
        task="classification",
        features=["value"],
        split="group",
        group_column="group",
        cv_folds=3,
        models=["baseline"],
    )
    split = build_split(frame, y, config)
    groups = [set(frame.iloc[ids]["group"]) for ids in (split.train, split.validation, split.test)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    for a, b in split.folds:
        assert not set(frame.iloc[a]["group"]) & set(frame.iloc[b]["group"])
        assert set(a) | set(b) <= set(split.train)


def test_time_split_preserves_equal_timestamps_and_gap():
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2025-01-01", periods=60).astype(str), 2),
            "value": np.arange(120),
        }
    )
    y = pd.Series(np.tile(["a", "b"], 60))
    config = ExperimentConfig(
        dataset_id="b" * 32,
        target="target",
        task="classification",
        features=["value"],
        split="time",
        time_column="timestamp",
        time_gap=1,
        cv_folds=3,
    )
    split = build_split(frame, y, config)
    assert len(split.purged) == 4
    assert (
        frame.iloc[split.train]["timestamp"].max() < frame.iloc[split.validation]["timestamp"].min()
    )
    assert (
        frame.iloc[split.validation]["timestamp"].max() < frame.iloc[split.test]["timestamp"].min()
    )
    for a, b in split.folds:
        assert frame.iloc[a]["timestamp"].max() < frame.iloc[b]["timestamp"].min()
        assert not set(a) & set(b)


def test_split_reproducible_and_preprocessing_train_only(forge):
    record = forge.sample("regression")
    config = config_for(record, "regression", models=["linear"])
    data = forge.data(record["id"])
    y, _ = validate_problem(data, config)
    split = build_split(data.frame, y, config)
    data.frame.loc[split.validation, "signal_1"] = 99999
    data.frame.loc[split.test, "segment"] = "HOLDOUT_ONLY"
    pipeline, result = train(data, config)
    imputer = (
        pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    )
    assert imputer.statistics_[0] == pytest.approx(
        data.frame.iloc[split.train]["signal_1"].median()
    )
    encoder = (
        pipeline.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
    )
    assert "HOLDOUT_ONLY" not in encoder.categories_[0]
    assert result["split_indices"]["test"] == split.test.tolist()


def test_final_holdout_not_used_for_selection(forge, monkeypatch):
    import mlforge.training as module

    record = forge.sample("regression")
    config = config_for(record, "regression")
    data = forge.data(record["id"])
    y, _ = validate_problem(data, config)
    split = build_split(data.frame, y, config)
    original = module.evaluate
    accesses = []

    def tracked(pipeline, x, y, task):
        accesses.append(bool(set(x.index) & set(split.test)))
        return original(pipeline, x, y, task)

    monkeypatch.setattr(module, "evaluate", tracked)
    _, first = train(data, config)
    assert accesses[-1] and sum(accesses) == 1
    data.frame.loc[split.test, "target"] += 100000
    _, second = train(data, config)
    assert first["selected_algorithm"] == second["selected_algorithm"]
    assert [c["selection_score"] for c in first["candidates"]] == [
        c["selection_score"] for c in second["candidates"]
    ]
    assert first["final_test"]["rmse"] != second["final_test"]["rmse"]


def test_all_missing_unseen_and_booleans(forge):
    record = forge.sample("classification")
    data = forge.data(record["id"])
    data.frame["empty"] = np.nan
    data.schema.append({"name": "empty", "type": "number", "role": "empty", "unique": 0})
    config = config_for(
        record,
        features=["signal_1", "segment", "is_repeat", "empty"],
        models=["baseline", "linear", "hist_gradient"],
    )
    pipeline, result = train(data, config)
    assert all(c["status"] == "succeeded" for c in result["candidates"])
    row = pd.DataFrame(
        [{"signal_1": np.nan, "segment": "NEVER_SEEN", "is_repeat": True, "empty": np.nan}]
    )
    assert len(pipeline.predict(feature_frame(row, data.schema, config.features))) == 1


def test_partial_failure_preserved_and_all_failure_recorded(forge, monkeypatch):
    import mlforge.training as module

    record = forge.sample("classification")
    original = module.build_pipeline

    def fail_tree(config, schema, algorithm, params):
        if algorithm == "tree":
            raise ValueError("deliberate test failure")
        return original(config, schema, algorithm, params)

    monkeypatch.setattr(module, "build_pipeline", fail_tree)
    exp = forge.start(config_for(record), background=False)
    assert exp["status"] == "succeeded"
    assert any(c["status"] == "failed" for c in exp["candidates"])
    assert exp["result"]["final_test_evaluations"] == 1

    def fail_all(*args):
        raise ValueError("deliberate failure")

    monkeypatch.setattr(module, "build_pipeline", fail_all)
    failed = forge.start(config_for(record), background=False)
    assert failed["status"] == "failed" and all(
        c["status"] == "failed" for c in failed["candidates"]
    )
    assert "artifact" not in failed


def test_cross_validation_and_bounded_search(forge):
    record = forge.sample("classification")
    exp = forge.start(
        config_for(record, models=["baseline", "tree"], cv_folds=3, tune=True), background=False
    )
    assert exp["status"] == "succeeded", exp
    assert exp["result"]["effective_cv_folds"] == 3
    assert len(exp["candidates"]) == 3
    assert all(
        c["selection_source"] == "training_cv_mean" and len(c["cv_scores"]) == 3
        for c in exp["candidates"]
    )


def test_negative_r2_is_valid_and_losses_rank_down():
    assert selection_value({"r2": -2}, "r2") == -2
    assert selection_value({"rmse": 10}, "rmse") == -10
    with pytest.raises(ValueError):
        selection_value({"roc_auc": None}, "roc_auc")


def test_binary_probability_metrics(forge):
    raw = sample_csv("classification", rows=180)
    data = inspect_csv(raw, forge.settings)
    data.frame["target"] = data.frame["target"].replace({"Class C": "Class B"})
    record = forge.ingest(data.frame.to_csv(index=False).encode(), "binary.csv")
    exp = forge.start(config_for(record, models=["baseline", "linear"]), background=False)
    assert exp["status"] == "succeeded", exp
    final = exp["result"]["final_test"]
    assert 0 <= final["roc_auc"] <= 1
    assert 0 <= final["average_precision"] <= 1
    assert final["calibration"]
    assert all(e["kind"] in ("false_positive", "false_negative") for e in final["errors"])


def test_problem_errors_do_not_silently_drop_target_rows(forge):
    record = forge.sample("classification")
    data = forge.data(record["id"])
    config = config_for(record)
    data.frame.loc[0, "target"] = None
    with pytest.raises(ForgeError, match="missing labels"):
        validate_problem(data, config)
    data.frame["target"] = "only"
    with pytest.raises(ForgeError, match="two distinct"):
        validate_problem(data, config)


def test_rare_class_cv_falls_back_without_touching_final_holdout():
    frame = pd.DataFrame({"value": np.arange(90)})
    y = pd.Series(["a"] * 86 + ["b"] * 4)
    config = ExperimentConfig(
        dataset_id="c" * 32, target="target", task="classification", features=["value"], cv_folds=3
    )
    split = build_split(frame, y, config)
    assert not split.folds and any("Too few" in w for w in split.warnings)
    assert not set(split.train) & set(split.test)


def test_all_regression_algorithms_work(forge):
    record = forge.sample("regression")
    exp = forge.start(
        config_for(
            record, "regression", models=["baseline", "linear", "tree", "forest", "hist_gradient"]
        ),
        background=False,
    )
    assert exp["status"] == "succeeded", exp
    assert all(c["status"] == "succeeded" for c in exp["candidates"])
    assert all(c["fit_s"] >= 0 for c in exp["candidates"])
    assert exp["result"]["memory"]["rss_end_bytes"] > 0
