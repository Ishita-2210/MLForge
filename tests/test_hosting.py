from dataclasses import replace

import pytest
from conftest import ROOT, config_for
from streamlit.testing.v1 import AppTest

from mlforge.errors import ForgeError
from mlforge.hosting import hosted_settings
from mlforge.service import Forge
from mlforge.storage import uid


def test_hosted_settings_are_temporary_bounded_and_path_safe(tmp_path):
    first, second = hosted_settings(tmp_path, uid()), hosted_settings(tmp_path, uid())
    assert first.home != second.home
    assert first.home.is_relative_to(tmp_path.resolve() / "sessions")
    assert first.max_rows == 2000 and first.max_bytes == 2 * 1024 * 1024
    assert first.max_features == 20
    with pytest.raises(ForgeError):
        hosted_settings(tmp_path, "../escape")


def test_demo_workspace_caps_preserve_existing_data(tmp_path):
    settings = replace(
        hosted_settings(tmp_path, uid()), max_datasets=1, max_experiments=1, max_models=1
    )
    forge = Forge(settings)
    record = forge.sample("classification")
    assert forge.sample("classification")["id"] == record["id"]
    with pytest.raises(ForgeError, match="dataset limit"):
        forge.sample("regression")
    config = config_for(record, models=["baseline", "linear"])
    exp = forge.start(config, background=False)
    assert exp["status"] == "succeeded", exp
    with pytest.raises(ForgeError, match="experiment limit"):
        forge.start(config, background=False)
    model = forge.register(exp["id"], "demo")
    with pytest.raises(ForgeError, match="model limit"):
        forge.register(exp["id"], "demo")
    forge.delete_model(model["id"])
    assert forge.register(exp["id"], "demo")["version"] == 2


def test_hosted_browser_sessions_do_not_share_datasets(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFORGE_HOSTED_DEMO", "1")
    monkeypatch.setenv("MLFORGE_HOME", str(tmp_path))
    first = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not first.exception
    assert any("temporary workspace" in item.value for item in first.info)
    first.sidebar.radio[0].set_value("Datasets").run()
    next(b for b in first.button if b.label == "Load classification sample").click().run()
    assert not first.exception
    own = Forge(hosted_settings(tmp_path, first.session_state["hosted_workspace"]))
    assert own.store.overview()["datasets"] == 1
    second = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not second.exception
    assert first.session_state["hosted_workspace"] != second.session_state["hosted_workspace"]
    assert all(m.value == "0" for m in second.metric)
    second.sidebar.radio[0].set_value("Datasets").run()
    assert not second.exception
    assert not second.dataframe


def test_container_entrypoint_keeps_xsrf_enabled(monkeypatch):
    from scripts import serve_hosted

    captured = []
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PORT", "8017")
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "mlforge.example")
    monkeypatch.setattr(serve_hosted.os, "execv", lambda executable, args: captured.append(args))
    serve_hosted.main()
    args = captured[0]
    assert "8017" in args and "0.0.0.0" in args
    assert "mlforge.example" in args
    assert not any("enableXsrfProtection" in arg or "enableCORS" in arg for arg in args)
