"""
3_analytics.py — Analytics page for the Streamlit app.

Features:
- Filters: age group, race, admission type
- Model selector (4 analysis models)
- Run Analysis button
- Results chart and table
- Download CSV + Add to Report buttons
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.state import add_to_report, init_state
from scripts.query_helpers import (
    hba1c_vs_readmission,
    los_by_readmission,
    medication_counts,
    readmission_by_group,
    top_diagnoses_by_readmission,
)

CONFIG_PATH = "config.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_engine(config: dict):
    db_path = config["database"]["path"]
    if not Path(db_path).exists():
        return None
    return create_engine(f"sqlite:///{db_path}", echo=False)


def _get_filter_options(engine: Engine) -> dict:
    """Fetch available filter values from database."""
    opts: dict = {}
    with engine.connect() as conn:
        try:
            ag = conn.execute(text("SELECT DISTINCT age_group FROM patients WHERE age_group IS NOT NULL ORDER BY age_group")).fetchall()
            opts["age_groups"] = [r[0] for r in ag]
        except Exception:
            opts["age_groups"] = []

        try:
            rc = conn.execute(text("SELECT DISTINCT race FROM patients WHERE race IS NOT NULL ORDER BY race")).fetchall()
            opts["races"] = [r[0] for r in rc]
        except Exception:
            opts["races"] = []

        try:
            at = conn.execute(text("SELECT DISTINCT admission_type_id FROM admissions WHERE admission_type_id IS NOT NULL ORDER BY admission_type_id")).fetchall()
            opts["admission_types"] = [str(r[0]) for r in at]
        except Exception:
            opts["admission_types"] = []
    return opts


MODELS = {
    "Readmission by Age Group": "readmission_age",
    "Readmission by Race": "readmission_race",
    "HbA1c vs Readmission": "hba1c",
    "Top Diagnoses by Readmission": "top_diagnoses",
    "Medication Frequency": "medication_counts",
    "Length of Stay by Readmission": "los",
}


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

render_sidebar()
init_state()

st.title("📊 Analytics")
st.markdown("Select filters and an analysis model, then click **Run Analysis**.")

try:
    config = _load_config()
except FileNotFoundError:
    st.error("config.json not found.")
    st.stop()

engine = _get_engine(config)
if engine is None:
    st.warning(
        "Database not found. Run the import pipeline on the **Data Sources** page first."
    )
    st.stop()

filter_opts = _get_filter_options(engine)

# ---- Filters ----
st.subheader("Filters")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    age_filter = st.multiselect(
        "Age Group",
        options=filter_opts.get("age_groups", []),
        default=[],
        placeholder="All age groups",
    )
with fcol2:
    race_filter = st.multiselect(
        "Race",
        options=filter_opts.get("races", []),
        default=[],
        placeholder="All races",
    )
with fcol3:
    admission_filter = st.multiselect(
        "Admission Type",
        options=filter_opts.get("admission_types", []),
        default=[],
        placeholder="All types",
    )

st.divider()

# ---- Model Selector ----
st.subheader("Analysis Model")
model_label = st.radio(
    "Select analysis",
    options=list(MODELS.keys()),
    horizontal=True,
)
model_key = MODELS[model_label]

# Model-specific parameters
top_n = config["pipeline"]["top_n_diagnoses"]
binary_mode = False
if model_key in ("top_diagnoses",):
    top_n = st.slider("Top N diagnoses", min_value=5, max_value=30, value=top_n)
if model_key in ("readmission_age", "readmission_race"):
    binary_mode = st.checkbox("Binary readmission (1=readmitted, 0=not)", value=False)

st.divider()

# ---- Run Analysis ----
if st.button("Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Running analysis..."):
        try:
            result_df: pd.DataFrame = pd.DataFrame()
            fig: go.Figure = go.Figure()

            if model_key == "readmission_age":
                result_df = readmission_by_group(engine, group_col="age_group", binary=binary_mode)
                fig = px.bar(
                    result_df,
                    x="group_value",
                    y="count",
                    color="readmission" if not binary_mode else "readmission",
                    barmode="group",
                    title="Readmission by Age Group",
                    labels={"group_value": "Age Group"},
                )

            elif model_key == "readmission_race":
                result_df = readmission_by_group(engine, group_col="race", binary=binary_mode)
                fig = px.bar(
                    result_df,
                    x="group_value",
                    y="count",
                    color="readmission",
                    barmode="group",
                    title="Readmission by Race",
                    labels={"group_value": "Race"},
                )

            elif model_key == "hba1c":
                result_df = hba1c_vs_readmission(engine)
                fig = px.bar(
                    result_df,
                    x="hba1c_result",
                    y="rate",
                    color="readmission",
                    barmode="group",
                    title="HbA1c Result vs Readmission",
                )

            elif model_key == "top_diagnoses":
                result_df = top_diagnoses_by_readmission(engine, top_n=top_n)
                totals = result_df.groupby("icd9_code")["count"].sum().reset_index()
                totals.columns = ["icd9_code", "total"]
                readmitted = result_df[result_df["readmission"].isin(["<30", ">30"])].groupby("icd9_code")["count"].sum().reset_index()
                readmitted.columns = ["icd9_code", "readmitted"]
                merged = totals.merge(readmitted, on="icd9_code", how="left").fillna(0)
                merged["rate"] = merged["readmitted"] / merged["total"] * 100
                merged = merged.sort_values("rate")
                fig = px.bar(
                    merged, x="rate", y="icd9_code", orientation="h",
                    title=f"Top {top_n} Diagnoses by Readmission Rate",
                    labels={"rate": "Readmission %", "icd9_code": "ICD-9 Code"},
                )

            elif model_key == "medication_counts":
                result_df = medication_counts(engine, top_n=top_n)
                fig = px.bar(
                    result_df,
                    x="count",
                    y="drug_name",
                    orientation="h",
                    title="Most Prescribed Medications",
                )

            elif model_key == "los":
                result_df = los_by_readmission(engine)
                fig = px.bar(
                    result_df,
                    x="group_value",
                    y="mean_los",
                    title="Mean Length of Stay by Readmission Class",
                    labels={"group_value": "Readmission Class", "mean_los": "Mean LOS (days)"},
                    error_y=None,
                )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # Download and report buttons
            dcol, rcol = st.columns(2)
            with dcol:
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=f"{model_key}_results.csv",
                    mime="text/csv",
                )
            with rcol:
                if st.button("Add to Report"):
                    add_to_report(fig, model_label)
                    st.success(f"Added '{model_label}' to report.")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
