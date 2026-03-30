"""
state.py — Streamlit session state helpers for the generic data pipeline app.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Internal helper — lazy engine reference for report persistence
# ---------------------------------------------------------------------------

def _get_report_engine():
    """Return the SQLAlchemy engine stored in session state, or None."""
    return st.session_state.get("_db_engine")


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

    if con is not None:
        # Store engine so report helpers can reach it without extra args
        st.session_state["_db_engine"] = con
        try:
            from app.core.pipelines import ensure_pipelines_tables
            ensure_pipelines_tables(con)
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _filter_dicts_from(filters: list | None) -> list[dict]:
    """Convert Filter dataclass instances (or plain dicts) to plain dicts."""
    result: list[dict] = []
    for f in (filters or []):
        if isinstance(f, dict):
            result.append(f)
        else:
            result.append({"column": f.column, "op": f.op, "value": f.value})
    return result


def add_to_report(
    fig: go.Figure,
    title: str,
    filters: list | None = None,
    dataset_name: str = "",
    row_count: int | None = None,
    total_rows: int | None = None,
) -> None:
    """Persist a Plotly figure + metadata to the DB (and session state cache)."""
    init_state()
    filter_dicts = _filter_dicts_from(filters)

    engine = _get_report_engine()
    item_id: int | None = None
    if engine is not None:
        try:
            from app.core.reports import save_report_item
            item_id = save_report_item(
                engine, title, fig, filter_dicts, dataset_name, row_count, total_rows
            )
        except Exception:
            pass  # fall back to session-only

    st.session_state["report_items"].append(
        {
            "id": item_id,
            "title": title,
            "fig": fig,
            "filters": filter_dicts,
            "dataset_name": dataset_name,
            "row_count": row_count,
            "total_rows": total_rows,
        }
    )


def get_report_items() -> list[dict[str, Any]]:
    """
    Return report items, loading from DB if the session cache is empty.

    On first call after a page refresh the session list is empty; we reload
    from DB so items survive refreshes.
    """
    init_state()
    if not st.session_state["report_items"]:
        engine = _get_report_engine()
        if engine is not None:
            try:
                from app.core.reports import list_report_items
                st.session_state["report_items"] = list_report_items(engine)
            except Exception:
                pass
    return st.session_state["report_items"]


def remove_report_item(index: int) -> None:
    """Remove a report item by index (updates DB and session cache)."""
    init_state()
    items: list = st.session_state["report_items"]
    if 0 <= index < len(items):
        item = items[index]
        engine = _get_report_engine()
        if engine is not None and item.get("id") is not None:
            try:
                from app.core.reports import delete_report_item
                delete_report_item(engine, item["id"])
            except Exception:
                pass
        st.session_state["report_items"] = items[:index] + items[index + 1:]


def move_report_item(index: int, direction: int) -> None:
    """Move a report item up (direction=-1) or down (direction=+1)."""
    init_state()
    items: list = st.session_state["report_items"]
    new_index = index + direction
    if 0 <= new_index < len(items):
        engine = _get_report_engine()
        id_a = items[index].get("id")
        id_b = items[new_index].get("id")
        if engine is not None and id_a is not None and id_b is not None:
            try:
                from app.core.reports import swap_report_items
                swap_report_items(engine, id_a, id_b)
            except Exception:
                pass
        items[index], items[new_index] = items[new_index], items[index]
        st.session_state["report_items"] = items


def clear_report() -> None:
    """Remove all items from DB and session cache."""
    engine = _get_report_engine()
    if engine is not None:
        try:
            from app.core.reports import clear_all_reports
            clear_all_reports(engine)
        except Exception:
            pass
    st.session_state["report_items"] = []
