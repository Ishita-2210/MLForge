from pathlib import Path

import pytest

from mlforge.config import Settings
from mlforge.schemas import ExperimentConfig
from mlforge.service import Forge

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def forge(tmp_path):
    return Forge(Settings(tmp_path / "workspace"))


def config_for(record, task="classification", **changes):
    values = dict(
        name="Test experiment",
        dataset_id=record["id"],
        target="target",
        task=task,
        features=[s["name"] for s in record["schema"] if s["name"] not in ("target", "record_id")],
        models=["baseline", "linear", "tree"],
        primary_metric="f1_macro" if task == "classification" else "rmse",
    )
    values.update(changes)
    return ExperimentConfig(**values)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    f = Forge(Settings(tmp_path_factory.mktemp("trained")))
    out = {}
    for task in ("classification", "regression"):
        data = f.sample(task)
        exp = f.start(config_for(data, task), background=False)
        assert exp["status"] == "succeeded", exp
        model = f.register(exp["id"], f"test-{task}")
        frame = f.data(data["id"]).frame[model["features"]].iloc[:3]
        rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
        out[task] = (f, data, exp, model, rows)
    return out
