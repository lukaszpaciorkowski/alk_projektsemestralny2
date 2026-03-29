"""
state.py — Streamlit session state helpers for the diabetes analysis app.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def init_state() -> None:
    """Initialise all required session state keys if not already set."""
    defaults: dict[str, Any] = {
        "report_items": [],      # list of {"title": str, "fig": go.Figure}
        "db_loaded": False,
        "config_path": "config.json",
        "import_log": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def add_to_report(fig: go.Figure, title: str) -> None:
    """Add a Plotly figure + title to the report queue."""
    init_state()
    st.session_state["report_items"].append({"title": title, "fig": fig})


def get_report_items() -> list[dict[str, Any]]:
    """Return the current list of report items."""
    init_state()
    return st.session_state["report_items"]


def remove_report_item(index: int) -> None:
    """Remove a report item by index."""
    init_state()
    items: list = st.session_state["report_items"]
    if 0 <= index < len(items):
        st.session_state["report_items"] = items[:index] + items[index + 1 :]


def move_report_item(index: int, direction: int) -> None:
    """Move a report item up (direction=-1) or down (direction=+1)."""
    init_state()
    items: list = st.session_state["report_items"]
    new_index = index + direction
    if 0 <= new_index < len(items):
        items[index], items[new_index] = items[new_index], items[index]
        st.session_state["report_items"] = items


def clear_report() -> None:
    """Remove all items from the report queue."""
    st.session_state["report_items"] = []
