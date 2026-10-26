from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from app.ui.common import choose_dataset, forge, heading, show_error


def datasets() -> None:
    heading(
        "01 / Data workspace",
        "Bring your data into focus",
        "Upload a UTF-8 CSV, or start with a reproducible CPU-friendly sample. Raw data stays on this computer.",
    )
    f = forge()
    left, right = st.columns(2)
    for task, column in (("classification", left), ("regression", right)):
        with column:
            st.subheader(task.title())
            st.write(
                "600 synthetic rows · mixed numeric, categorical and boolean features · seed 42"
            )
            if st.button(f"Load {task} sample", use_container_width=True):
                try:
                    record = f.sample(task)
                    st.session_state.selected_dataset = record["id"]
                    st.success("Synthetic sample loaded. No real-world benchmark is implied.")
                except Exception as exc:
                    show_error(exc)
    upload = st.file_uploader("Upload CSV · maximum 8 MiB, 10,000 rows, 60 columns", type=["csv"])
    if upload is not None and st.button("Validate and save upload", type="primary"):
        try:
            record = f.ingest(upload.getvalue(), upload.name)
            st.session_state.selected_dataset = record["id"]
            st.success("Dataset validated and saved locally. Identical content is deduplicated.")
        except Exception as exc:
            show_error(exc)
    st.divider()
    record = choose_dataset()
    if record is None:
        return
    a, b, c, d = st.columns(4)
    for col, label, value in (
        (a, "Rows", record["rows"]),
        (b, "Columns", record["columns"]),
        (c, "Missing cells", record["profile"]["missing_cells"]),
        (d, "Duplicate rows", record["profile"]["duplicates"]),
    ):
        col.metric(label, f"{value:,}")
    st.caption(f"Source: {record['source']} · Fingerprint: {record['fingerprint']}")
    frame = f.data(record["id"]).frame
    start = st.number_input(
        "Preview starts at row", min_value=0, max_value=max(0, len(frame) - 1), value=0, step=25
    )
    st.dataframe(frame.iloc[start : start + 25], use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame(record["schema"]), use_container_width=True, hide_index=True)
    st.json(record["profile"]["actions"])
    with st.expander("Manage stored dataset"):
        st.write(
            "Deletion removes the stored CSV and dataset listing. Experiment metadata and registered pipelines remain usable; reproducing training needs the dataset again."
        )
        confirmed = st.checkbox(
            f"Confirm deletion of {record['name']}", key="confirm_dataset_delete"
        )
        if st.button("Delete dataset", disabled=not confirmed):
            try:
                f.delete_dataset(record["id"])
                st.session_state.pop("selected_dataset", None)
                st.rerun()
            except Exception as exc:
                show_error(exc)


def profiling() -> None:
    heading(
        "02 / Understand",
        "Profile before you train",
        "Quality flags are diagnostics. Review them in the context of your prediction problem.",
    )
    record = choose_dataset()
    if record is None:
        return
    data = forge().data(record["id"])
    for note in record["profile"]["warnings"]:
        st.warning(note)
    missing = pd.DataFrame(record["schema"]).set_index("name")[["missing_pct"]]
    st.subheader("Missingness by feature (%)")
    st.bar_chart(missing)
    cols = data.frame.columns.tolist()
    selected = st.selectbox(
        "Inspect column", cols, index=cols.index("target") if "target" in cols else 0
    )
    if selected in data.profile["numeric"]:
        st.json(data.profile["numeric"][selected])
        valid = data.frame[selected].dropna()
        if len(valid):
            counts, bins = np.histogram(valid, bins=min(20, max(1, valid.nunique())))
            st.bar_chart(
                pd.DataFrame(
                    {
                        "range": [
                            f"{a:.3g}–{b:.3g}" for a, b in zip(bins[:-1], bins[1:], strict=False)
                        ],
                        "count": counts,
                    }
                ).set_index("range")
            )
    else:
        st.json(data.profile["categorical"].get(selected, {}))
        st.bar_chart(data.frame[selected].dropna().astype(str).value_counts().head(20))
    st.download_button(
        "Download profile JSON",
        json.dumps(record["profile"], indent=2),
        file_name="profile.json",
        mime="application/json",
    )
