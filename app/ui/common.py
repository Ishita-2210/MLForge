from __future__ import annotations

import uuid
from html import escape
from pathlib import Path

import streamlit as st

from mlforge.config import Settings
from mlforge.errors import ForgeError
from mlforge.hosting import hosted_demo, hosted_settings
from mlforge.service import Forge


@st.cache_resource(ttl=3600, max_entries=32)
def _forge(home: str, demo: bool = False, session: str = "") -> Forge:
    return Forge(hosted_settings(Path(home), session) if demo else Settings(Path(home)))


def forge() -> Forge:
    if hosted_demo():
        if "hosted_workspace" not in st.session_state:
            st.session_state.hosted_workspace = uuid.uuid4().hex
        return _forge(str(Settings.default().home), True, st.session_state.hosted_workspace)
    return _forge(str(Settings.default().home))


def heading(eyebrow: str, title: str, description: str) -> None:
    st.html(
        '<div class="page-heading">'
        f'<div class="page-eyebrow">{escape(eyebrow.upper())}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(description)}</p></div>"
    )


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
