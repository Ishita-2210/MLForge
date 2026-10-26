from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.common import forge, heading, show_error
from app.ui.model_views import choose_model


def monitoring() -> None:
    heading(
        "07 / Observe",
        "Monitoring & input drift",
        "Inspect actual prediction requests. Input shift is different from model performance degradation, which needs ground-truth labels.",
    )
    f = forge()
    logs = f.logs(1000)
    a, b, c = st.columns(3)
    a.metric("Requests retained", len(logs))
    b.metric("Failed requests", sum(r["status"] == "failed" for r in logs))
    c.metric(
        "Mean observed latency",
        f"{sum(r['duration_ms'] for r in logs) / len(logs):.2f} ms" if logs else "—",
    )
    st.caption(
        "Retention: latest 1000 requests, at most 30 days. No raw prediction input rows are retained. Aggregated summaries can still be sensitive; local files are not encrypted."
    )
    if st.button("Refresh monitoring"):
        st.rerun()
    if logs:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        k: r.get(k)
                        for k in (
                            "created_at",
                            "model_id",
                            "version",
                            "rows",
                            "status",
                            "duration_ms",
                            "synthetic_demo",
                            "error",
                        )
                    }
                    for r in logs[:100]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        by_id = {r["id"]: r for r in logs[:100]}
        selected = st.selectbox(
            "Inspect retained request",
            list(by_id),
            format_func=lambda k: f"{by_id[k]['created_at']} · {by_id[k]['status']} · {by_id[k]['rows']} rows",
        )
        st.json(by_id[selected])
    else:
        st.info("No prediction requests yet. Register a model and use the Playground or API.")
    st.subheader("Deterministic drift demonstration")
    st.write(
        "Generate 100 artificial rows shifted eight training standard deviations, with an unseen text category. This executes actual predictions and logs a labeled synthetic demonstration."
    )
    model = choose_model()
    if model and st.button("Run synthetic input-shift demonstration"):
        try:
            st.json(f.drift_demo(model["id"]))
        except Exception as exc:
            show_error(exc)
    with st.expander("Delete retained monitoring history"):
        confirmed = st.checkbox("Confirm deletion of all prediction request logs")
        if st.button("Clear monitoring history", disabled=not confirmed):
            f.store.retain_logs(0, 0)
            st.rerun()
