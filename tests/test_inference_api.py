import copy
import json

import numpy as np
import pytest
from conftest import config_for
from fastapi.testclient import TestClient

from app.api import create_app
from mlforge.artifacts import artifact_path, load_pipeline
from mlforge.drift import compare, reference
from mlforge.errors import ForgeError
from mlforge.service import Forge


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_api_full_lifecycle_and_reload(trained, task):
    f, data, exp, model, rows = trained[task]
    fresh = Forge(f.settings)
    before = fresh.predict(model["id"], rows)
    with TestClient(create_app(f.settings)) as client:
        assert client.get("/health").json()["cpu_only"]
        assert client.get("/models").status_code == 200
        assert client.get(f"/models/{model['id']}").json()["experiment_id"] == exp["id"]
        actual = client.post(
            f"/predict/{model['id']}",
            json={"rows": [dict(reversed(list(row.items()))) for row in rows]},
        )
        assert actual.status_code == 200, actual.text
        assert actual.json()["predictions"] == before.predictions
        assert actual.json()["model_version"] == model["version"]
        assert (
            client.get(f"/experiments/{exp['id']}").json()["result"]["final_test_evaluations"] == 1
        )
        assert client.get("/monitoring").json()["requests"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "wrong_type",
        "boolean_numeric",
        "nonfinite",
        "too_many",
        "empty",
        "nonobject",
    ],
)
def test_invalid_prediction_inputs_safe_and_logged(trained, mutation):
    f, _, _, model, base = trained["classification"]
    rows = copy.deepcopy(base)
    if mutation == "missing":
        rows[0].pop("signal_1")
    if mutation == "extra":
        rows[0]["unexpected"] = "private value"
    if mutation == "wrong_type":
        rows[0]["signal_1"] = "private invalid value"
    if mutation == "boolean_numeric":
        rows[0]["signal_1"] = True
    if mutation == "nonfinite":
        rows[0]["signal_1"] = 1e99
    if mutation == "too_many":
        rows *= 100
    if mutation == "empty":
        rows = []
    if mutation == "nonobject":
        rows = ["private invalid value"]
    with TestClient(create_app(f.settings)) as client:
        response = client.post(f"/predict/{model['id']}", json={"rows": rows})
        assert response.status_code == 422
        assert "Traceback" not in response.text and "private invalid value" not in response.text
    logs = f.store.list("prediction")
    assert logs[0]["status"] == "failed"
    assert "private invalid value" not in json.dumps(logs)
    assert "rows_data" not in logs[0]


def test_request_size_chunked_and_path_limits(trained):
    f, _, _, model, _ = trained["classification"]
    with TestClient(create_app(f.settings)) as client:
        assert (
            client.post(f"/predict/{model['id']}", content=b"x" * (1024 * 1024 + 1)).status_code
            == 413
        )

        def chunks():
            for _ in range(20):
                yield b"x" * 65536

        assert client.post(f"/predict/{model['id']}", content=chunks()).status_code == 413
        assert client.post("/predict/not-a-valid-id", json={"rows": [{"x": 1}]}).status_code == 404
        assert client.get("/models/not-a-valid-id").status_code == 404
        assert client.get("/health", headers={"host": "untrusted.example"}).status_code == 400


def test_artifact_integrity_before_deserialization(forge, monkeypatch):
    record = forge.sample("regression")
    exp = forge.start(
        config_for(record, "regression", models=["baseline", "linear"]), background=False
    )
    model = forge.register(exp["id"], "integrity")
    path = artifact_path(forge.settings.home, "models", model["id"])
    path.write_bytes(path.read_bytes() + b"modified")

    def forbidden(*args):
        raise AssertionError("joblib.load must not run")

    monkeypatch.setattr("mlforge.artifacts.joblib.load", forbidden)
    with pytest.raises(ForgeError, match="integrity"):
        load_pipeline(path, model)
    model["environment"]["sklearn"] = "0.0"
    with pytest.raises(ForgeError, match="incompatible"):
        load_pipeline(path, model)


def test_registry_versions_deletion_and_dataset_independence(forge):
    record = forge.sample("regression")
    exp = forge.start(
        config_for(record, "regression", models=["baseline", "linear"]), background=False
    )
    first = forge.register(exp["id"], "family")
    forge.delete_model(first["id"])
    second = forge.register(exp["id"], "family")
    assert second["version"] == 2 and first["id"] != second["id"]
    frame = forge.data(record["id"]).frame[second["features"]].iloc[:1]
    row = frame.astype(object).where(frame.notna(), None).to_dict("records")
    forge.delete_dataset(record["id"])
    assert len(forge.predict(second["id"], row).predictions) == 1
    with pytest.raises(ForgeError, match="deleted"):
        forge.predict(first["id"], row)
    assert forge.store.get("experiment", exp["id"])["status"] == "succeeded"


def test_drift_known_shift_and_no_raw_rows_in_logs(trained):
    f, _, _, model, _ = trained["regression"]
    result = f.drift_demo(model["id"])
    assert len(result["drift"]["flagged_features"]) >= 6
    assert "not production" in result["demonstration"]
    log = f.store.list("prediction")[0]
    assert log["synthetic_demo"] is True
    assert "predictions" not in log and "probabilities" not in log
    assert log["rows"] == 100 and "prediction_summary" in log


def test_drift_small_batches_not_flagged():
    import pandas as pd

    frame = pd.DataFrame({"x": np.arange(100, dtype=float), "category": ["a"] * 100})
    baseline = reference(
        frame, [{"name": "x", "type": "number"}, {"name": "category", "type": "string"}]
    )
    shift = frame.iloc[:2].copy()
    shift["x"] = 10000
    report = compare(shift, baseline)
    assert report["status"] == "insufficient_rows" and not report["flagged_features"]


def test_unchanged_all_missing_features_have_no_false_drift():
    import pandas as pd

    frame = pd.DataFrame({"x": [np.nan] * 100, "category": [None] * 100})
    baseline = reference(
        frame, [{"name": "x", "type": "number"}, {"name": "category", "type": "string"}]
    )
    report = compare(frame, baseline)
    assert not report["flagged_features"]
    assert all(f["score"] == 0 for f in report["features"].values())


def test_monitoring_retention(forge):
    from mlforge.storage import now, uid

    for i in range(10):
        forge.store.save(
            "prediction", {"id": uid(), "created_at": now(), "status": "failed", "rows": i}
        )
    forge.store.retain_logs(3, 30)
    assert len(forge.store.list("prediction")) == 3
    forge.store.retain_logs(0, 0)
    assert not forge.store.list("prediction")


def test_auc_unavailable_for_one_class_is_null(trained):
    from mlforge.metrics import evaluate
    from mlforge.models import feature_frame

    f, record, _, model, _ = trained["classification"]
    frame = f.data(record["id"]).frame
    frame = frame[frame["target"] == "Class A"]
    pipeline = load_pipeline(artifact_path(f.settings.home, "models", model["id"]), model)
    with pytest.warns(UserWarning, match="y_pred contains classes"):
        report = evaluate(
            pipeline,
            feature_frame(frame, model["feature_schema"], model["features"]),
            frame["target"],
            "classification",
        )
    assert report["roc_auc"] is None and report["average_precision"] is None
