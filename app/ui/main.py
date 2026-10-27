from __future__ import annotations

import os
from html import escape

import streamlit as st

from app.ui.common import forge
from app.ui.data_views import datasets, profiling
from app.ui.design import apply_theme, brand, card_intro, icon, section, topbar
from app.ui.experiment_views import builder, details
from app.ui.model_views import playground, registry
from app.ui.monitor_views import monitoring
from mlforge.hosting import hosted_demo


def navigate(page: str) -> None:
    st.session_state.workspace_page = page


def load_sample(task: str) -> None:
    try:
        record = forge().sample(task)
        st.session_state.selected_dataset = record["id"]
        navigate("Experiment builder")
    except Exception as exc:
        from app.ui.common import show_error

        show_error(exc)


def overview() -> None:
    f = forge()
    f.recover_jobs()
    counts = f.store.overview()
    with st.container(key="overview-hero"):
        intro, action = st.columns([3, 1], gap="large", vertical_alignment="center")
        with intro:
            st.html(
                '<div class="hero-copy"><h1>Your workspace</h1>'
                "<p>A little data. A new idea. Your next experiment starts here.</p></div>"
            )
        with action:
            st.button(
                "New experiment",
                type="primary",
                icon=":material/add:",
                use_container_width=True,
                on_click=navigate,
                args=("Experiment builder" if counts["datasets"] else "Datasets",),
            )
    with st.container(key="overview-metrics"):
        for col, label, key in zip(
            st.columns(4),
            ["Datasets", "Experiments", "Successful runs", "Registered models"],
            ["datasets", "experiments", "succeeded", "models"],
            strict=False,
        ):
            col.metric(label, counts[key])
    section(
        "Get started",
        "Bring your own data or try a sample.",
    )
    upload, classification, regression = st.columns(3, gap="medium")
    with upload, st.container(key="start-upload"):
        card_intro(
            "upload",
            "Import your data",
            "A CSV is all you need. We’ll help you inspect the rest.",
        )
        st.button(
            "Import a dataset",
            icon=":material/arrow_forward:",
            use_container_width=True,
            on_click=navigate,
            args=("Datasets",),
        )
    with classification, st.container(key="start-classification"):
        card_intro(
            "box",
            "Explore classification",
            "Predict a category with a ready-to-run synthetic dataset.",
            "blue",
        )
        st.button(
            "Load classification sample",
            icon=":material/arrow_forward:",
            use_container_width=True,
            on_click=load_sample,
            args=("classification",),
        )
    with regression, st.container(key="start-regression"):
        card_intro(
            "chart",
            "Explore regression",
            "Predict a number and compare models on a synthetic dataset.",
            "teal",
        )
        st.button(
            "Load regression sample",
            icon=":material/arrow_forward:",
            use_container_width=True,
            on_click=load_sample,
            args=("regression",),
        )
    label, action = st.columns([4, 1], vertical_alignment="center")
    with label:
        section("Recent experiments")
    action.button(
        "View all runs",
        type="tertiary",
        icon=":material/arrow_forward:",
        on_click=navigate,
        args=("Training & comparison",),
        use_container_width=True,
    )
    recent = f.store.list("experiment", 5)
    with st.container(key="overview-history"):
        if recent:
            rows = []
            for record in recent:
                state = record["status"]
                state_class = state if state in ("succeeded", "failed", "cancelled") else ""
                config = record.get("config", {})
                rows.append(
                    f"<tr><td>{escape(record['name'])}</td><td>{escape(config.get('task', '—').title())}</td>"
                    f'<td><span class="state-badge {state_class}">{escape(state.title())}</span></td>'
                    f"<td>{escape(record['created_at'][:10])}</td></tr>"
                )
            st.html(
                '<table class="experiment-list"><thead><tr><th>EXPERIMENT</th><th>TASK</th>'
                "<th>STATUS</th><th>CREATED</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )
        else:
            st.html(
                f'<div class="empty-history"><span class="card-icon">{icon("flask", 18)}</span>'
                "<h3>No experiments yet</h3>"
                "<p>Start with a dataset. Your runs will appear here.</p></div>"
            )
    st.html(
        f'<div class="workspace-footer"><span>{icon("check", 12)} '
        "Your data, your experiments.</span>"
        "<span>Powered by scikit-learn</span></div>"
    )


def main() -> None:
    st.set_page_config(page_title="MLForge · Your ML Workspace", page_icon="◈", layout="wide")
    apply_theme()
    with st.sidebar:
        brand()
        st.html(
            '<div class="workspace-picker"><span class="workspace-avatar">ML</span><span>'
            "<strong>Personal workspace</strong>"
            f"<small>{'Public demo · Temporary session' if hosted_demo() else 'Local · Persistent storage'}</small>"
            '</span></div><div class="nav-label">WORKSPACE</div>'
        )
        pages = {
            "Overview": overview,
            "Datasets": datasets,
            "Data profiling": profiling,
            "Experiment builder": builder,
            "Training & comparison": details,
            "Model registry": registry,
            "Playground": playground,
            "Monitoring": monitoring,
        }
        selected = st.radio(
            "Workspace", list(pages), label_visibility="collapsed", key="workspace_page"
        )
        st.html(
            '<div class="sidebar-runtime"><div class="runtime-title"><span class="status-dot"></span>'
            "CPU workspace</div><p>"
            + (
                "Temporary session. Use public data."
                if hosted_demo()
                else "Your data stays on this computer."
            )
            + "</p></div>"
        )
        if hosted_demo():
            st.link_button(
                "Source and local setup",
                "https://github.com/Ishita-2210/MLForge#run-on-windows",
                use_container_width=True,
            )
            st.download_button(
                "Download local run instructions",
                "MLForge local setup (PowerShell; Python 3.11-3.13)\n\nSource: https://github.com/Ishita-2210/MLForge\n\nIf the folder does not already exist:\ngit clone https://github.com/Ishita-2210/MLForge.git 'D:\\ml forge'\n\ncd 'D:\\ml forge'\npython -m venv .venv\n.\\.venv\\Scripts\\python -m pip install -r requirements-lock.txt\n.\\.venv\\Scripts\\python -m pip install -e . --no-deps --no-build-isolation\n.\\.venv\\Scripts\\python scripts/dev.py\n\nWeb UI: http://127.0.0.1:8501\nLocal REST API: http://127.0.0.1:8000/docs\nCtrl+C stops both services.\n",
                file_name="mlforge-local-setup.txt",
                use_container_width=True,
            )
            with st.expander("About this demo"):
                st.info(
                    "Hosted demo: isolated temporary workspace for this browser session. Models and history can disappear on restart or reconnection. Use synthetic/public data here; run locally for private data and persistent storage. The dedicated REST API is included in local mode."
                )
        else:
            st.link_button(
                "Prediction API documentation",
                os.environ.get("MLFORGE_API_URL", "http://127.0.0.1:8000") + "/docs",
                use_container_width=True,
            )
        st.html(
            '<div class="sidebar-footer">MLForge 1.0 &nbsp; / &nbsp; An open-source workspace</div>'
        )
    topbar(selected, hosted_demo())
    pages[selected]()
