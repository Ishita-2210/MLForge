from __future__ import annotations

from pathlib import Path

import streamlit as st

from mlforge.config import Settings
from mlforge.errors import ForgeError
from mlforge.service import Forge


@st.cache_resource
def _forge(home: str) -> Forge:
    return Forge(Settings(Path(home)))


def forge() -> Forge:
    return _forge(str(Settings.default().home))


def heading(eyebrow: str, title: str, description: str) -> None:
    st.caption(eyebrow.upper())
    st.title(title)
    st.write(description)


def choose_dataset():
    f = forge()
    records = f.store.list("dataset")
    if not records:
        st.info("No datasets yet. Open Datasets to upload a CSV or load a synthetic sample.")
        return None
    ids = [r["id"] for r in records]
    by_id = {r["id"]: r for r in records}
    current = st.session_state.get("selected_dataset")
    selected = st.selectbox(
        "Dataset",
        ids,
        index=ids.index(current) if current in ids else 0,
        format_func=lambda key: f"{by_id[key]['name']} · {by_id[key]['rows']:,} rows · {key[:6]}",
    )
    st.session_state.selected_dataset = selected
    return by_id[selected]


def show_error(exc: Exception) -> None:
    if isinstance(exc, ForgeError):
        st.error(str(exc))
    else:
        st.error("Check the configuration and input format. The operation could not complete.")
