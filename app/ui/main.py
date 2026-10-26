from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from app.ui.common import forge, heading
from app.ui.data_views import datasets, profiling
from app.ui.experiment_views import builder, details
from app.ui.model_views import playground, registry
from app.ui.monitor_views import monitoring


def overview() -> None:
    heading(
        "LOCAL FIRST · CPU READY",
        "From raw data to a model you can trust",
        "A focused tabular ML workbench: validate data, compare models honestly, register a pipeline, and observe its predictions.",
    )
    f = forge()
    f.recover_jobs()
    counts = f.store.overview()
    for col, label, key in zip(
        st.columns(4),
        ["Datasets", "Experiments", "Successful runs", "Registered models"],
        ["datasets", "experiments", "succeeded", "models"],
        strict=False,
    ):
        col.metric(label, counts[key])
    st.divider()
    for col, step, title, text in zip(
        st.columns(3),
        ["01 / UNDERSTAND", "02 / EXPERIMENT", "03 / DELIVER"],
        ["Start with the data", "Keep evaluation honest", "Ship a traceable pipeline"],
        [
            "Upload CSV or load a synthetic sample. Inspect missingness, types and warnings.",
            "Choose features and a split that matches your problem. Fit preprocessing only on training rows.",
            "Review validation and holdout results, register a version, and make predictions from the same persisted pipeline.",
        ],
        strict=False,
    ):
        with col:
            st.caption(step)
            st.subheader(title)
            st.write(text)
    st.info(
        "First run: open Datasets → Load classification sample → Experiment builder → Validate and review → Start reviewed experiment."
    )
    st.subheader("Recent experiments")
    recent = f.store.list("experiment", 5)
    if recent:
        st.dataframe(
            pd.DataFrame(
                [
                    {k: r.get(k) for k in ("name", "status", "created_at", "progress")}
                    for r in recent
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("Your workspace is empty. Counts and metrics appear after you run real workflows.")
    st.caption(
        "Classification & regression · one training job · SQLite metadata · local files · no cloud inference"
    )


def main() -> None:
    st.set_page_config(page_title="MLForge · Local ML Workbench", page_icon="⚒️", layout="wide")
    with st.sidebar:
        st.markdown("## ⚒️ MLForge")
        st.caption("TABULAR ML WORKBENCH")
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
        selected = st.radio("Workspace", list(pages), label_visibility="collapsed")
        st.divider()
        st.caption("Local storage · CPU only\nOne experiment at a time")
        st.link_button(
            "Prediction API documentation",
            os.environ.get("MLFORGE_API_URL", "http://127.0.0.1:8000") + "/docs",
            use_container_width=True,
        )
    pages[selected]()
