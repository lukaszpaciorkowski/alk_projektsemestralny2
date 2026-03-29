"""
filter_panel.py — Reusable interactive filter panel component.

render_filter_panel(table_name, meta, engine, key_prefix) -> list[Filter]

Each page uses a unique key_prefix so filter state is independent.
State is stored in st.session_state[f"{key_prefix}_filters"].
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.core.query import Filter, fetch_distinct_values, row_count


_NUMERIC_OPS: dict[str, str] = {
    "= (equals)": "eq",
    "!= (not equals)": "neq",
    ">= (gte)": "gte",
    "<= (lte)": "lte",
    "> (gt)": "gt",
    "< (lt)": "lt",
    "is null": "isnull",
    "not null": "notnull",
}

_CAT_OPS: dict[str, str] = {
    "is one of (IN)": "in",
    "is not one of (NOT IN)": "nin",
    "contains": "like",
    "is null": "isnull",
    "not null": "notnull",
}


def _render_filter_row(
    i: int,
    columns: list[dict],
    engine: Any,
    table_name: str,
    key_prefix: str,
) -> dict | None:
    """Render one filter row. Returns filter dict or None when removed."""
    col_names = [c["name"] for c in columns]
    col_dtypes = {c["name"]: c.get("dtype", "object") for c in columns}

    f1, f2, f3, f4 = st.columns([2, 2, 3, 0.5])

    with f1:
        col_name = st.selectbox(
            "Column",
            col_names,
            key=f"{key_prefix}_filter_col_{i}",
            label_visibility="collapsed",
        )

    dtype = col_dtypes.get(col_name, "object")
    is_numeric = "int" in dtype or "float" in dtype
    ops = _NUMERIC_OPS if is_numeric else _CAT_OPS

    with f2:
        op_label = st.selectbox(
            "Op",
            list(ops.keys()),
            key=f"{key_prefix}_filter_op_{i}",
            label_visibility="collapsed",
        )
    op = ops[op_label]

    value: Any = None
    with f3:
        if op in ("isnull", "notnull"):
            st.caption("(no value needed)")
        elif op in ("in", "nin"):
            choices = fetch_distinct_values(table_name, col_name, engine, limit=200)
            value = st.multiselect(
                "Values",
                options=choices,
                key=f"{key_prefix}_filter_val_{i}",
                label_visibility="collapsed",
            )
        elif op == "like":
            raw = st.text_input(
                "Value",
                key=f"{key_prefix}_filter_val_{i}",
                label_visibility="collapsed",
                placeholder="substring to match",
            )
            value = f"%{raw}%" if raw else None
        elif is_numeric:
            value = st.number_input(
                "Value",
                key=f"{key_prefix}_filter_val_{i}",
                label_visibility="collapsed",
            )
        else:
            value = st.text_input(
                "Value",
                key=f"{key_prefix}_filter_val_{i}",
                label_visibility="collapsed",
            )

    with f4:
        removed = st.button("✕", key=f"{key_prefix}_filter_rm_{i}")

    if removed:
        return None
    return {"column": col_name, "op": op, "value": value}


def render_filter_panel(
    table_name: str,
    meta: list[dict],
    engine: Any,
    key_prefix: str = "filter",
) -> list[Filter]:
    """
    Render an interactive filter builder and return active Filter objects.

    Args:
        table_name: Table to filter (used for distinct-value lookups and row count).
        meta: List of column dicts with "name" and "dtype" keys.
        engine: SQLAlchemy engine.
        key_prefix: Unique prefix for all session-state keys on this page.

    Returns:
        List of Filter objects to pass to fetch_table() / row_count().
    """
    state_key = f"{key_prefix}_filters"
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    # Add / Clear controls
    btn_add, btn_clear, _ = st.columns([1, 1, 5])
    with btn_add:
        if st.button("+ Add filter", key=f"{key_prefix}_add_filter"):
            st.session_state[state_key].append(None)

    # Render each filter row
    active_filters: list[dict] = []
    surviving: list = []

    removed_any = False
    for i, _ in enumerate(st.session_state[state_key]):
        row = _render_filter_row(i, meta, engine, table_name, key_prefix)
        if row is not None:
            active_filters.append(row)
            surviving.append(row)
        else:
            removed_any = True

    st.session_state[state_key] = surviving
    if removed_any:
        st.rerun()

    filter_objs = [Filter(f["column"], f["op"], f["value"]) for f in active_filters]

    if active_filters:
        with btn_clear:
            if st.button("Clear all", key=f"{key_prefix}_clear_filters"):
                st.session_state[state_key] = []
                st.rerun()
        try:
            matching = row_count(table_name, engine, filter_objs)
            st.caption(
                f"**{matching:,}** rows match "
                f"{len(active_filters)} active filter(s)"
            )
        except Exception as exc:
            st.caption(f"Filter error: {exc}")

    return filter_objs


def active_filter_count(key_prefix: str) -> int:
    """Return the number of active filter rows without rendering anything."""
    return len(st.session_state.get(f"{key_prefix}_filters", []))
