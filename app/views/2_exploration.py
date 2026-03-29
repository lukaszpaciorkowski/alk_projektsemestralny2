"""
2_exploration.py — Data Exploration page.

Features:
- Dataset selector (all imported datasets)
- Dynamic filter builder (column → op → value)
- Paginated table via query.fetch_table
- Column inspector with stats + mini histogram
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.core.pipeline import DB_PATH, get_engine, get_dataset_meta, list_datasets
from app.core.query import Filter, fetch_column_stats, fetch_distinct_values, fetch_table, row_count
from app.core.type_detector import dataset_type_icon
from app.state import init_state, set_active_dataset

PAGE_SIZE = 50


def _get_engine():
    return get_engine(DB_PATH)


# ---------------------------------------------------------------------------
# Filter widget helpers
# ---------------------------------------------------------------------------

_NUMERIC_OPS = {
    "= (equals)": "eq",
    "!= (not equals)": "neq",
    ">= (gte)": "gte",
    "<= (lte)": "lte",
    "> (gt)": "gt",
    "< (lt)": "lt",
    "is null": "isnull",
    "not null": "notnull",
}

_CAT_OPS = {
    "is one of (IN)": "in",
    "is not one of (NOT IN)": "nin",
    "is null": "isnull",
    "not null": "notnull",
}


def _render_filter_row(
    i: int,
    columns: list[dict],
    engine,
    table_name: str,
) -> dict | None:
    """Render a single filter row. Returns filter dict or None if removed."""
    col_names = [c["name"] for c in columns]
    col_dtypes = {c["name"]: c.get("dtype", "object") for c in columns}

    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 3, 0.5])

    with f_col1:
        col_name = st.selectbox(
            "Column",
            col_names,
            key=f"filter_col_{i}",
            label_visibility="collapsed",
        )

    dtype = col_dtypes.get(col_name, "object")
    is_numeric = "int" in dtype or "float" in dtype

    ops = _NUMERIC_OPS if is_numeric else _CAT_OPS
    with f_col2:
        op_label = st.selectbox(
            "Op",
            list(ops.keys()),
            key=f"filter_op_{i}",
            label_visibility="collapsed",
        )
    op = ops[op_label]

    value: Any = None
    with f_col3:
        if op in ("isnull", "notnull"):
            st.caption("(no value needed)")
        elif op in ("in", "nin"):
            choices = fetch_distinct_values(table_name, col_name, engine, limit=200)
            value = st.multiselect(
                "Values",
                options=choices,
                key=f"filter_val_{i}",
                label_visibility="collapsed",
            )
        elif is_numeric:
            value = st.number_input(
                "Value",
                key=f"filter_val_{i}",
                label_visibility="collapsed",
            )
        else:
            value = st.text_input(
                "Value",
                key=f"filter_val_{i}",
                label_visibility="collapsed",
            )

    with f_col4:
        removed = st.button("✕", key=f"filter_rm_{i}")

    if removed:
        return None

    return {"column": col_name, "op": op, "value": value}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_sidebar()

try:
    engine = _get_engine()
except Exception:
    engine = None

init_state(engine)

st.title("🔍 Data Exploration")

if engine is None:
    st.warning("Database not found. Import a dataset on the **Data Sources** page first.")
    st.stop()

datasets = list_datasets(engine)
if not datasets:
    st.warning("No datasets imported yet. Go to **Data Sources** to upload a CSV.")
    st.stop()

# ---- Dataset Selector ----
ds_options = {
    f"{dataset_type_icon(d['dataset_type'])} {d['display_name']} "
    f"({d['row_count']:,} rows, {d['col_count']} cols)": d
    for d in datasets
}

active_name = st.session_state.get("active_dataset_name", "")
default_idx = 0
for idx, label in enumerate(ds_options):
    if active_name and active_name in label:
        default_idx = idx
        break

selected_label = st.selectbox("Dataset", list(ds_options.keys()), index=default_idx)
selected_ds = ds_options[selected_label]
table_name = selected_ds["table_name"]
if selected_ds.get("description"):
    st.caption(selected_ds["description"])

# Load column metadata
meta_raw = selected_ds.get("columns") or []
if isinstance(meta_raw, str):
    meta_raw = json.loads(meta_raw)

# Sync active dataset
if st.session_state.get("active_dataset") != table_name:
    set_active_dataset(
        table_name=table_name,
        display_name=selected_ds["display_name"],
        dataset_type=selected_ds["dataset_type"],
        enrichment_status=selected_ds["enrichment_status"],
        meta=meta_raw,
    )
    st.rerun()

columns = meta_raw  # list of dicts with name, dtype, etc.

# ---- Dynamic Filter Builder ----
st.divider()
st.subheader("Filters")

if "filters" not in st.session_state:
    st.session_state["filters"] = []

if st.button("+ Add filter"):
    st.session_state["filters"].append(None)

active_filters: list[dict] = []
new_filter_state: list = []

for i, _ in enumerate(st.session_state["filters"]):
    result = _render_filter_row(i, columns, engine, table_name)
    if result is not None:
        active_filters.append(result)
        new_filter_state.append(result)
    # None means removed → drop from list

st.session_state["filters"] = new_filter_state

# Convert to Filter objects
filter_objs = [Filter(f["column"], f["op"], f["value"]) for f in active_filters]

# Show filter summary
try:
    total_matching = row_count(table_name, engine, filter_objs)
    full_count = selected_ds["row_count"] or 0
    if active_filters:
        st.caption(
            f"**{total_matching:,}** rows match filters "
            f"(out of {full_count:,} total)"
        )
        if st.button("Clear all filters"):
            st.session_state["filters"] = []
            st.rerun()
    else:
        st.caption(f"Total rows: **{full_count:,}** (no filters applied)")
except Exception as exc:
    st.error(f"Filter error: {exc}")
    total_matching = selected_ds["row_count"] or 0

# ---- Paginated Table ----
st.divider()
st.subheader("Table")

# Sort controls
col_names = [c["name"] for c in columns]
sort_col1, sort_col2, sort_col3 = st.columns([3, 1, 1])
with sort_col1:
    sort_by = st.selectbox("Sort by", ["(none)"] + col_names, key="explore_sort")
with sort_col2:
    ascending = st.radio("Direction", ["Ascending", "Descending"], horizontal=True) == "Ascending"

order_by = sort_by if sort_by != "(none)" else None

# Pagination
n_pages = max(1, (total_matching - 1) // PAGE_SIZE + 1)
with sort_col3:
    page_num = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)

offset = (page_num - 1) * PAGE_SIZE

try:
    page_df = fetch_table(
        table_name,
        engine,
        filters=filter_objs,
        order_by=order_by,
        ascending=ascending,
        limit=PAGE_SIZE,
        offset=offset,
    )
    st.caption(
        f"Showing rows {offset + 1}–{min(offset + PAGE_SIZE, total_matching):,} "
        f"of {total_matching:,}  |  Page {page_num} / {n_pages}"
    )
    st.dataframe(page_df, use_container_width=True, hide_index=True)
except Exception as exc:
    st.error(f"Failed to load data: {exc}")

# ---- Column Inspector ----
st.divider()
st.subheader("Column Inspector")

inspector_col, _ = st.columns([2, 3])
with inspector_col:
    inspect_col_name = st.selectbox(
        "Select column", col_names, key="inspect_col"
    )

col_meta = next((c for c in columns if c["name"] == inspect_col_name), None)
if col_meta:
    dtype = col_meta.get("dtype", "object")
    st.caption(f"Type: `{dtype}`")

    try:
        stats = fetch_column_stats(table_name, inspect_col_name, dtype, engine)

        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Total", f"{stats['total']:,}")
        with info_cols[1]:
            st.metric("Non-null", f"{stats['non_null_count']:,}")
        with info_cols[2]:
            st.metric("Null %", f"{stats['null_pct']}%")
        with info_cols[3]:
            st.metric("Unique", f"{stats['unique_count']:,}")

        if "min" in stats:
            st.caption(
                f"Min: **{stats['min']}** | Max: **{stats['max']}** | "
                f"Mean: **{stats.get('mean')}** | Median: **{stats.get('median')}** | "
                f"Std: **{stats.get('std')}**"
            )

        # Mini histogram / bar chart
        if "int" in dtype or "float" in dtype:
            mini_df = pd.read_sql(
                text(
                    f"SELECT [{inspect_col_name}] FROM [{table_name}] "
                    f"WHERE [{inspect_col_name}] IS NOT NULL LIMIT 10000"
                ),
                engine,
            )
            if not mini_df.empty:
                fig = px.histogram(
                    mini_df,
                    x=inspect_col_name,
                    nbins=30,
                    height=200,
                    title="",
                )
                fig.update_layout(margin={"t": 10, "b": 10, "l": 10, "r": 10})
                st.plotly_chart(fig, use_container_width=True)
        else:
            vc_df = pd.read_sql(
                text(
                    f"SELECT [{inspect_col_name}], COUNT(*) AS cnt FROM [{table_name}] "
                    f"WHERE [{inspect_col_name}] IS NOT NULL "
                    f"GROUP BY [{inspect_col_name}] ORDER BY cnt DESC LIMIT 20"
                ),
                engine,
            )
            if not vc_df.empty:
                fig = px.bar(
                    vc_df,
                    x=inspect_col_name,
                    y="cnt",
                    height=200,
                    title="",
                )
                fig.update_layout(margin={"t": 10, "b": 40, "l": 10, "r": 10})
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Column stats failed: {exc}")
