"""
diabetes.py — Diabetes-specific analytics functions.

Operate on the flat raw table (ds_diabetic_<hash>) imported via pipeline.py.
Enrichment-dependent functions (medication_frequency, top_diagnoses) check
for the existence of _meds / _diag tables and raise EnrichmentRequiredError if missing.

Every function: run_*(df, meta, **params) -> tuple[pd.DataFrame, go.Figure | None]
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.pipeline import EnrichmentRequiredError

logger = logging.getLogger(__name__)

_READMIT_COL = "readmitted"


def _readmit_binary(series: pd.Series) -> pd.Series:
    """Map readmission values to binary 0/1."""
    return series.map(lambda v: 0 if str(v).upper() in ("NO", "0") else 1)


def _ensure_col(df: pd.DataFrame, col: str, default: str = "") -> str:
    """Return col if it exists, else try to find a close match."""
    if col in df.columns:
        return col
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(col.lower(), default)


# ---------------------------------------------------------------------------
# 1. Readmission Rate by Group
# ---------------------------------------------------------------------------

def run_readmission_by_group(
    df: pd.DataFrame,
    meta: list[dict],
    group_by: str = "age",
    readmission_binary: bool = True,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Readmission rate broken down by a categorical column."""
    group_by = _ensure_col(df, group_by, "age")
    readmit_col = _ensure_col(df, _READMIT_COL)

    if not group_by or group_by not in df.columns:
        return pd.DataFrame({"message": [f"Column '{group_by}' not found."]}), None
    if not readmit_col or readmit_col not in df.columns:
        return pd.DataFrame({"message": ["Readmission column not found."]}), None

    work = df[[group_by, readmit_col]].dropna()

    if readmission_binary:
        work = work.copy()
        work["readmitted_bin"] = _readmit_binary(work[readmit_col])
        result = (
            work.groupby(group_by)["readmitted_bin"]
            .agg(["sum", "count"])
            .reset_index()
        )
        result.columns = [group_by, "readmitted_count", "total"]
        result["rate_pct"] = (result["readmitted_count"] / result["total"] * 100).round(2)
        result = result.sort_values(group_by)

        fig = px.bar(
            result,
            x=group_by,
            y="rate_pct",
            title=f"Readmission Rate (%) by {group_by}",
            labels={"rate_pct": "Readmission %"},
            text_auto=".1f",
        )
    else:
        result = (
            work.groupby([group_by, readmit_col])
            .size()
            .reset_index(name="count")
        )
        totals = result.groupby(group_by)["count"].transform("sum")
        result["rate_pct"] = (result["count"] / totals * 100).round(2)

        fig = px.bar(
            result,
            x=group_by,
            y="count",
            color=readmit_col,
            barmode="group",
            title=f"Readmission by {group_by}",
        )

    fig.update_layout(xaxis_tickangle=-45)
    return result, fig


# ---------------------------------------------------------------------------
# 2. HbA1c vs Readmission
# ---------------------------------------------------------------------------

def run_hba1c_vs_readmission(
    df: pd.DataFrame,
    meta: list[dict],
    readmission_binary: bool = True,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Readmission rate by HbA1c test result (A1Cresult column)."""
    a1c_col = _ensure_col(df, "A1Cresult") or _ensure_col(df, "a1cresult")
    if not a1c_col or a1c_col not in df.columns:
        # Try case-insensitive search
        for c in df.columns:
            if "a1c" in c.lower():
                a1c_col = c
                break
    readmit_col = _ensure_col(df, _READMIT_COL)

    if not a1c_col or a1c_col not in df.columns:
        return pd.DataFrame({"message": ["HbA1c (A1Cresult) column not found."]}), None
    if not readmit_col:
        return pd.DataFrame({"message": ["Readmission column not found."]}), None

    work = df[[a1c_col, readmit_col]].dropna()
    if readmission_binary:
        work = work.copy()
        work["readmitted_bin"] = _readmit_binary(work[readmit_col])
        result = (
            work.groupby(a1c_col)["readmitted_bin"]
            .agg(["sum", "count"])
            .reset_index()
        )
        result.columns = [a1c_col, "readmitted_count", "total"]
        result["rate_pct"] = (result["readmitted_count"] / result["total"] * 100).round(2)
        fig = px.bar(
            result,
            x=a1c_col,
            y="rate_pct",
            title="Readmission Rate (%) by HbA1c Result",
            labels={"rate_pct": "Readmission %"},
            text_auto=".1f",
        )
    else:
        result = work.groupby([a1c_col, readmit_col]).size().reset_index(name="count")
        totals = result.groupby(a1c_col)["count"].transform("sum")
        result["rate_pct"] = (result["count"] / totals * 100).round(2)
        fig = px.bar(
            result,
            x=a1c_col,
            y="count",
            color=readmit_col,
            barmode="group",
            title="HbA1c Result vs Readmission",
        )

    return result, fig


# ---------------------------------------------------------------------------
# 3. Top Diagnoses by Readmission (requires enrichment)
# ---------------------------------------------------------------------------

def run_top_diagnoses(
    df: pd.DataFrame,
    meta: list[dict],
    top_n: int = 10,
    con: Engine | None = None,
    table_name: str = "",
    enrichment_status: str = "none",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Top N primary diagnoses and their readmission rates (uses _diag enrichment table)."""
    if enrichment_status != "done":
        raise EnrichmentRequiredError(
            "Top Diagnoses requires enrichment. Run enrichment on the Data Sources page."
        )
    if con is None or not table_name:
        raise EnrichmentRequiredError("Database connection required for enrichment-based analysis.")

    diag_table = f"{table_name}_diag"
    readmit_col = _ensure_col(df, _READMIT_COL)

    with con.connect() as conn:
        diag_df = pd.read_sql(text(f"SELECT * FROM [{diag_table}]"), conn)

    if diag_df.empty:
        return pd.DataFrame({"message": ["No diagnosis data found in enrichment table."]}), None

    primary = diag_df[diag_df["diagnosis_position"] == 1][["encounter_id", "icd9_code"]].copy()

    # Merge with readmission
    id_col = "encounter_id" if "encounter_id" in df.columns else df.columns[0]
    merged = primary.merge(
        df[[id_col, readmit_col]].rename(columns={id_col: "encounter_id"}),
        on="encounter_id",
        how="inner",
    )

    top_codes = (
        merged.groupby("icd9_code")["encounter_id"]
        .count()
        .nlargest(top_n)
        .index.tolist()
    )
    merged = merged[merged["icd9_code"].isin(top_codes)]
    merged["readmitted_bin"] = _readmit_binary(merged[readmit_col])

    result = (
        merged.groupby("icd9_code")["readmitted_bin"]
        .agg(["sum", "count"])
        .reset_index()
    )
    result.columns = ["icd9_code", "readmitted", "total"]
    result["rate_pct"] = (result["readmitted"] / result["total"] * 100).round(2)
    result = result.sort_values("rate_pct", ascending=True)

    fig = px.bar(
        result,
        x="rate_pct",
        y="icd9_code",
        orientation="h",
        title=f"Top {top_n} Primary Diagnoses by Readmission Rate",
        labels={"rate_pct": "Readmission %", "icd9_code": "ICD-9 Code"},
        text_auto=".1f",
    )
    return result, fig


# ---------------------------------------------------------------------------
# 4. Medication Frequency (requires enrichment)
# ---------------------------------------------------------------------------

def run_medication_frequency(
    df: pd.DataFrame,
    meta: list[dict],
    top_n: int = 15,
    con: Engine | None = None,
    table_name: str = "",
    enrichment_status: str = "none",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Most prescribed medications from the _meds enrichment table."""
    if enrichment_status != "done":
        raise EnrichmentRequiredError(
            "Medication Frequency requires enrichment. Run enrichment on the Data Sources page."
        )
    if con is None or not table_name:
        raise EnrichmentRequiredError("Database connection required for enrichment-based analysis.")

    meds_table = f"{table_name}_meds"
    with con.connect() as conn:
        meds_df = pd.read_sql(text(f"SELECT * FROM [{meds_table}]"), conn)

    if meds_df.empty:
        return pd.DataFrame({"message": ["No medication data in enrichment table."]}), None

    result = (
        meds_df.groupby("drug")["encounter_id"]
        .count()
        .reset_index()
        .rename(columns={"encounter_id": "count"})
        .nlargest(top_n, "count")
        .sort_values("count")
    )

    fig = px.bar(
        result,
        x="count",
        y="drug",
        orientation="h",
        title=f"Top {top_n} Most Prescribed Medications",
        text_auto=True,
    )
    return result, fig


# ---------------------------------------------------------------------------
# 5. Length of Stay by Readmission
# ---------------------------------------------------------------------------

def run_los_by_readmission(
    df: pd.DataFrame,
    meta: list[dict],
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Mean length of stay broken down by readmission class."""
    los_col = _ensure_col(df, "time_in_hospital")
    readmit_col = _ensure_col(df, _READMIT_COL)

    if not los_col or los_col not in df.columns:
        return pd.DataFrame({"message": ["time_in_hospital column not found."]}), None
    if not readmit_col or readmit_col not in df.columns:
        return pd.DataFrame({"message": ["Readmission column not found."]}), None

    work = df[[readmit_col, los_col]].dropna()
    result = (
        work.groupby(readmit_col)[los_col]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    result.columns = [readmit_col, "mean_los", "median_los", "min_los", "max_los", "n"]
    result["mean_los"] = result["mean_los"].round(2)

    fig = px.bar(
        result,
        x=readmit_col,
        y="mean_los",
        title="Mean Length of Stay by Readmission Class",
        labels={"mean_los": "Mean LOS (days)", readmit_col: "Readmission"},
        text_auto=".2f",
    )
    return result, fig


# ---------------------------------------------------------------------------
# 6. Medications vs Length of Stay
# ---------------------------------------------------------------------------

def run_medications_vs_los(
    df: pd.DataFrame,
    meta: list[dict],
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Scatter: num_medications vs mean length of stay."""
    num_meds_col = _ensure_col(df, "num_medications")
    los_col = _ensure_col(df, "time_in_hospital")

    if not num_meds_col or num_meds_col not in df.columns:
        return pd.DataFrame({"message": ["num_medications column not found."]}), None
    if not los_col or los_col not in df.columns:
        return pd.DataFrame({"message": ["time_in_hospital column not found."]}), None

    work = df[[num_meds_col, los_col]].dropna()
    result = (
        work.groupby(num_meds_col)[los_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    result.columns = ["num_medications", "mean_los", "count"]
    result["mean_los"] = result["mean_los"].round(2)

    # Use OLS trendline if statsmodels is available, otherwise skip
    try:
        import statsmodels  # noqa: F401
        trendline = "ols"
    except ImportError:
        trendline = None

    fig = px.scatter(
        result,
        x="num_medications",
        y="mean_los",
        size="count",
        title="Number of Medications vs Mean Length of Stay",
        labels={"mean_los": "Mean LOS (days)", "num_medications": "Number of Medications"},
        trendline=trendline,
    )
    return result, fig
