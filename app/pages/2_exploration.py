"""
2_exploration.py — Data Exploration page for the Streamlit app.

Features:
- Table selector
- Paginated searchable data table
- Column histogram
- Summary statistics (describe)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.state import init_state

CONFIG_PATH = "config.json"
PAGE_SIZE = 50

TABLES = [
    "admissions",
    "patients",
    "medications",
    "diagnosis_encounters",
    "admission_types",
    "discharge_types",
    "diagnoses_lookup",
]


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_engine(config: dict):
    db_path = config["database"]["path"]
    if not Path(db_path).exists():
        return None
    return create_engine(f"sqlite:///{db_path}", echo=False)


def _load_table(engine, table: str, search: str = "", limit: int = 5000) -> pd.DataFrame:
    """Load table data with optional text search (across all string columns)."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"SELECT * FROM {table} LIMIT :lim"),
            conn,
            params={"lim": limit},
        )
    if search:
        mask = df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        df = df[mask]
    return df


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

render_sidebar()
init_state()

st.title("🔍 Data Exploration")

try:
    config = _load_config()
except FileNotFoundError:
    st.error("config.json not found.")
    st.stop()

engine = _get_engine(config)
if engine is None:
    st.warning(
        "Database not found. Please run the import pipeline on the **Data Sources** page first."
    )
else:
    # ---- Table Selector ----
    col_left, col_right = st.columns([2, 3])
    with col_left:
        selected_table = st.selectbox("Select Table", TABLES)
    with col_right:
        search_term = st.text_input("Search (filter rows)", placeholder="Type to filter...")

    # Load data
    try:
        df = _load_table(engine, selected_table, search=search_term)
    except Exception as exc:
        st.error(f"Failed to load table '{selected_table}': {exc}")
        df = pd.DataFrame()

    if not df.empty:
        st.caption(f"Showing {len(df):,} rows from `{selected_table}`")

        # ---- Paginated Data Table ----
        st.subheader(f"Table: {selected_table}")

        page_num = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, (len(df) - 1) // PAGE_SIZE + 1),
            value=1,
            step=1,
        )
        start = (page_num - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        st.dataframe(df.iloc[start:end], use_container_width=True, hide_index=True)

        st.divider()

        # ---- Column Histogram ----
        st.subheader("Column Distribution")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        all_cols = numeric_cols + cat_cols

        if all_cols:
            col_sel = st.selectbox("Choose column", all_cols)
            if col_sel in numeric_cols:
                bins = st.slider("Histogram bins", min_value=5, max_value=100, value=20)
                fig = px.histogram(
                    df,
                    x=col_sel,
                    nbins=bins,
                    title=f"Distribution of {col_sel}",
                )
            else:
                vc = df[col_sel].value_counts().reset_index()
                vc.columns = [col_sel, "count"]
                fig = px.bar(vc, x=col_sel, y="count", title=f"Value Counts: {col_sel}")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No columns available.")

        st.divider()

        # ---- Summary Statistics ----
        st.subheader("Summary Statistics")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns in this table.")
