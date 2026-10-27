"""Presentation helpers shared by the Streamlit workspace."""

from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
ICONS = {
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "data": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
    "chart": '<path d="M4 3v17h17M8 15v-4m5 4V7m5 8V4"/>',
    "flask": '<path d="M9 3h6m-5 0v6l-6 10a1.5 1.5 0 0 0 1.3 2h13.4a1.5 1.5 0 0 0 1.3-2L14 9V3M8 14h8"/>',
    "layers": '<path d="m12 3 10 5-10 5L2 8l10-5Zm-9 10 9 5 9-5M3 18l9 5 9-5"/>',
    "box": '<path d="m12 2 9 5v10l-9 5-9-5V7l9-5Zm0 10 9-5M12 12 3 7m9 5v10M7.5 4.5l9 5"/>',
    "play": '<path d="m8 4 12 8-12 8V4Z"/>',
    "pulse": '<path d="M2 12h5l3-8 4 16 3-8h5"/>',
    "arrow": '<path d="M4 12h16m-6-6 6 6-6 6"/>',
    "upload": '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/>',
    "spark": '<path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
    "code": '<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-16-2 20"/>',
    "cpu": '<rect x="5" y="5" width="14" height="14" rx="3"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3m6-3v3m-6 14v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"/>',
}


def _svg(name: str, size: int = 20) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="#80679e" stroke-width="1.65" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def svg_image(svg: str, class_name: str = "forge-icon") -> str:
    uri = base64.b64encode(svg.encode()).decode()
    return f'<img class="{class_name}" src="data:image/svg+xml;base64,{uri}" alt="">'


def icon(name: str, size: int = 20) -> str:
    return svg_image(_svg(name, size))


@lru_cache(maxsize=1)
def stylesheet() -> str:
    font = base64.b64encode((ASSETS / "fonts" / "InterVariable.woff2").read_bytes()).decode()
    css = (ASSETS / "workspace.css").read_text(encoding="utf-8")
    nav = []
    for index, name in enumerate(
        ("grid", "data", "chart", "flask", "layers", "box", "play", "pulse"), 1
    ):
        svg = _svg(name)
        uri = base64.b64encode(svg.encode()).decode()
        nav.append(
            f'[data-testid="stSidebar"] [role="radiogroup"] label:nth-child({index}) '
            f'{{ --nav-icon: url("data:image/svg+xml;base64,{uri}"); }}'
        )
    return (
        '@font-face {font-family: "Inter"; font-style: normal; font-weight: 100 900; '
        f'font-display: swap; src: url(data:font/woff2;base64,{font}) format("woff2");}}'
        + css
        + "".join(nav)
    )


def apply_theme() -> None:
    st.html(f"<style>{stylesheet()}</style>")


def brand() -> None:
    mark = svg_image(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="26" height="26" fill="none">'
        '<path d="m5 5 7-3 7 3-7 3-7-3Zm0 7 7-3 7 3-7 3-7-3Zm0 7 7-3 7 3-7 3-7-3Z" '
        'fill="#f4efff"/></svg>'
    )
    st.html(
        '<div class="forge-brand"><span class="forge-mark">' + mark + "</span>"
        '<span>MLForge<span class="brand-version">WORKSPACE</span></span></div>'
    )


def topbar(page: str, demo: bool) -> None:
    st.html(
        '<div class="workspace-topbar"><span class="breadcrumb">Workspace '
        f'<span class="breadcrumb-slash">/</span> <strong>{escape(page)}</strong></span>'
        '<span class="runtime-pill"><span class="status-dot"></span>'
        f"{'Demo workspace' if demo else 'Local workspace'}</span></div>"
    )


def section(title: str, description: str = "", tag: str = "") -> None:
    st.html(
        '<div class="section-heading"><div>'
        f"<h2>{escape(title)}</h2>"
        + (f"<p>{escape(description)}</p>" if description else "")
        + "</div>"
        + (f'<span class="quiet-tag">{escape(tag)}</span>' if tag else "")
        + "</div>"
    )


def card_intro(name: str, title: str, description: str, tone: str = "violet") -> None:
    st.html(
        f'<div class="action-card-intro"><span class="card-icon {escape(tone)}">'
        f"{icon(name, 21)}</span><h3>{escape(title)}</h3><p>{escape(description)}</p></div>"
    )
