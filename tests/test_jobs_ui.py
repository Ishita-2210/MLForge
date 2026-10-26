import os
import time

import pytest
from conftest import ROOT, config_for
from streamlit.testing.v1 import AppTest

from mlforge.config import Settings
from mlforge.errors import ForgeError
from mlforge.service import Forge
from mlforge.storage import now, uid


def wait_for(forge, key, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exp = forge.store.get("experiment", key)
        if exp["status"] not in ("queued", "running"):
            return exp
        time.sleep(0.1)
    raise AssertionError("Background worker did not finish within the test deadline")


def test_background_training_process_and_history(forge):
    record = forge.sample("classification")
    exp = forge.start(config_for(record, models=["baseline", "linear"]))
    completed = wait_for(forge, exp["id"])
    assert completed["status"] == "succeeded", completed
    assert completed["pid"] != os.getpid()
    assert Forge(forge.settings).store.get("experiment", exp["id"])["status"] == "succeeded"


def test_queued_cancel_and_single_job_policy(forge):
    record = forge.sample("classification")
    config = config_for(record)
    queued = {
        "id": uid(),
        "name": "queued",
        "created_at": now(),
        "status": "queued",
        "config": config.model_dump(),
        "pid": None,
    }
    forge.store.reserve_experiment(queued)
    with pytest.raises(ForgeError, match="active"):
        forge.start(config)
    with pytest.raises(ForgeError, match="active"):
        forge.delete_dataset(record["id"])
    forge.cancel(queued["id"])
    assert forge.store.get("experiment", queued["id"])["status"] == "cancelled"


def button(at, label):
    return next(b for b in at.button if b.label == label)


def page(at, name):
    at.sidebar.radio[0].set_value(name).run()
    assert not at.exception, at.exception


def test_empty_ui_and_all_navigation(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFORGE_HOME", str(tmp_path / "empty"))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    assert all(m.value == "0" for m in at.metric)
    for name in (
        "Datasets",
        "Data profiling",
        "Experiment builder",
        "Training & comparison",
        "Model registry",
        "Playground",
        "Monitoring",
        "Overview",
    ):
        page(at, name)


def test_ui_complete_classification_workflow(tmp_path, monkeypatch):
    home = tmp_path / "ui"
    monkeypatch.setenv("MLFORGE_HOME", str(home))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    page(at, "Datasets")
    button(at, "Load classification sample").click().run()
    assert not at.exception
    page(at, "Data profiling")
    page(at, "Experiment builder")
    button(at, "Validate and review").click().run()
    assert not at.exception
    button(at, "Start reviewed experiment").click().run()
    assert not at.exception
    forge = Forge(Settings(home))
    exp = forge.store.list("experiment")[0]
    completed = wait_for(forge, exp["id"])
    assert completed["status"] == "succeeded", completed
    page(at, "Training & comparison")
    button(at, "Register selected pipeline").click().run()
    assert not at.exception
    assert len(forge.store.list("model")) == 1
    page(at, "Model registry")
    page(at, "Playground")
    button(at, "Predict with registered model").click().run()
    assert not at.exception
    assert forge.store.list("prediction")[0]["status"] == "succeeded"
    page(at, "Monitoring")
    button(at, "Run synthetic input-shift demonstration").click().run()
    assert not at.exception
    assert forge.store.list("prediction")[0]["synthetic_demo"]


def test_running_cancel_only_terminates_matching_owned_worker(forge, monkeypatch):
    record = forge.sample("classification")
    key = uid()
    exp = {
        "id": key,
        "name": "cancellable",
        "created_at": now(),
        "status": "running",
        "config": config_for(record).model_dump(),
        "pid": 12345678,
        "process_created": 100.0,
    }
    forge.store.reserve_experiment(exp)
    events = []

    class Process:
        def create_time(self):
            return 100.0

        def cmdline(self):
            return ["python", "-m", "mlforge.worker", "--id", key]

        def terminate(self):
            events.append("terminate")

        def wait(self, seconds):
            events.append("wait")

    monkeypatch.setattr("mlforge.service.psutil.Process", lambda pid: Process())
    forge.cancel(key)
    assert events == ["terminate", "wait"]
    assert forge.store.get("experiment", key)["status"] == "cancelled"


def test_interrupted_worker_recovered_as_failed(forge):
    record = forge.sample("classification")
    exp = {
        "id": uid(),
        "name": "interrupted",
        "created_at": now(),
        "status": "running",
        "config": config_for(record).model_dump(),
        "pid": 2147483647,
        "process_created": 0,
    }
    forge.store.reserve_experiment(exp)
    forge.recover_jobs()
    assert forge.store.get("experiment", exp["id"])["status"] == "failed"
