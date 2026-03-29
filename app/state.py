"""
state.py — Streamlit session state helpers for the generic data pipeline app.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def init_state(con=None) -> None:
    """
    Initialise all required session state keys if not already set.

    Args:
        con: Optional SQLAlchemy engine. If provided, re-hydrates
             loaded_datasets from _datasets on every call (handles refresh).
    """
    defaults: dict[str, Any] = {
        # Reports
        "report_items": [],
        # Import log (list of str)
        "import_log": [],
        # Active dataset (table_name string)
        "active_dataset": None,
        # Active dataset type: 'generic' | 'diabetes' | ...
        "active_dataset_type": "generic",
        # Active dataset column metadata: list[dict] from _datasets.columns JSON
        "active_dataset_meta": [],
        # Active dataset display name
        "active_dataset_name": "",
        # Active dataset enrichment status
        "active_enrichment_status": "none",
        # Exploration page filters: list[dict] with keys: column, op, value
        "filters": [],
        # Ad hoc chart session history: list[dict]
        "adhoc_chart_history": [],
        # Selected analytics function id
        "selected_analysis": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Re-hydrate active dataset from DB if it was lost on refresh
    if con is not None and st.session_state.get("active_dataset") is None:
        _rehydrate_active_dataset(con)


def _rehydrate_active_dataset(con) -> None:
    """
    If session state lost the active dataset on a browser refresh,
    try to restore from the most recently imported _datasets row.
    """
    try:
        from sqlalchemy import text
        with con.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT table_name, display_name, dataset_type, "
                    "enrichment_status, columns FROM _datasets ORDER BY id DESC LIMIT 1"
                )
            ).fetchone()
        if row:
            import json as _json
            st.session_state["active_dataset"] = row[0]
            st.session_state["active_dataset_name"] = row[1]
            st.session_state["active_dataset_type"] = row[2]
            st.session_state["active_enrichment_status"] = row[3]
            st.session_state["active_dataset_meta"] = (
                _json.loads(row[4]) if row[4] else []
            )
    except Exception:
        pass  # DB may not exist yet — silent fail


def set_active_dataset(
    table_name: str,
    display_name: str,
    dataset_type: str,
    enrichment_status: str,
    meta: list[dict],
) -> None:
    """Update the active dataset in session state."""
    st.session_state["active_dataset"] = table_name
    st.session_state["active_dataset_name"] = display_name
    st.session_state["active_dataset_type"] = dataset_type
    st.session_state["active_enrichment_status"] = enrichment_status
    st.session_state["active_dataset_meta"] = meta
    st.session_state["filters"] = []


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

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
        st.session_state["report_items"] = items[:index] + items[index + 1:]


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
