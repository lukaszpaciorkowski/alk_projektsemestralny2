"""
generic.py — Generic analytics functions that work on any dataset.

Every function follows the contract:
    run_*(df, meta, **params) -> tuple[pd.DataFrame, go.Figure | None]

where:
    df   — the full dataset DataFrame
    meta — list of ColumnMeta dicts (from _datasets.columns JSON)
    **params — kwargs matching the function's ParamSpec list in registry.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


# ---------------------------------------------------------------------------
# 1. Descriptive Statistics
# ---------------------------------------------------------------------------

def run_describe(
    df: pd.DataFrame,
    meta: list[dict],
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """pandas .describe(include='all') transposed for readability."""
    result = df.describe(include="all").T.reset_index()
    result.rename(columns={"index": "column"}, inplace=True)
    return result, None


# ---------------------------------------------------------------------------
# 2. Correlation Matrix
# ---------------------------------------------------------------------------

def run_correlation(
    df: pd.DataFrame,
    meta: list[dict],
    method: str = "pearson",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Pearson / Spearman / Kendall correlation heatmap (numeric cols only)."""
    num_df = df[_numeric_cols(df)]
    if num_df.shape[1] < 2:
        return pd.DataFrame({"message": ["Need at least 2 numeric columns."]}), None

    corr = num_df.corr(method=method).round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Correlation Matrix ({method.capitalize()})",
        aspect="auto",
    )
    fig.update_layout(height=max(400, 50 * len(corr)))
    return corr.reset_index().rename(columns={"index": "column"}), fig


# ---------------------------------------------------------------------------
# 3. Value Counts
# ---------------------------------------------------------------------------

def run_value_counts(
    df: pd.DataFrame,
    meta: list[dict],
    column: str = "",
    top_n: int = 20,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Value counts for a selected column."""
    if not column or column not in df.columns:
        # default to first categorical column
        cats = _categorical_cols(df)
        column = cats[0] if cats else (df.columns[0] if len(df.columns) else "")

    if not column:
        return pd.DataFrame({"message": ["No columns available."]}), None

    vc = df[column].value_counts().head(top_n).reset_index()
    vc.columns = [column, "count"]
    vc["pct"] = (vc["count"] / len(df) * 100).round(2)

    fig = px.bar(
        vc,
        x=column,
        y="count",
        title=f"Value Counts: {column} (top {top_n})",
        text_auto=True,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return vc, fig


# ---------------------------------------------------------------------------
# 4. Group By / Aggregate
# ---------------------------------------------------------------------------

def run_groupby(
    df: pd.DataFrame,
    meta: list[dict],
    group_col: str = "",
    agg_col: str = "",
    agg_func: str = "mean",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Group by a categorical column and aggregate a numeric column."""
    cats = _categorical_cols(df)
    nums = _numeric_cols(df)

    if not group_col and cats:
        group_col = cats[0]
    if not agg_col and nums:
        agg_col = nums[0]

    if not group_col or not agg_col:
        return pd.DataFrame({"message": ["Need at least one categorical and one numeric column."]}), None

    if group_col not in df.columns or agg_col not in df.columns:
        return pd.DataFrame({"message": [f"Column not found: {group_col!r} or {agg_col!r}"]}), None

    funcs = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
    fn = funcs.get(agg_func, "mean")
    result = df.groupby(group_col)[agg_col].agg(fn).reset_index()
    result.columns = [group_col, f"{fn}_{agg_col}"]
    result = result.sort_values(f"{fn}_{agg_col}", ascending=False)

    fig = px.bar(
        result,
        x=group_col,
        y=f"{fn}_{agg_col}",
        title=f"{fn.capitalize()}({agg_col}) by {group_col}",
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return result, fig


# ---------------------------------------------------------------------------
# 5. Cross-tabulation (Pivot Table)
# ---------------------------------------------------------------------------

def run_crosstab(
    df: pd.DataFrame,
    meta: list[dict],
    row_col: str = "",
    col_col: str = "",
    values_col: str = "",
    agg_func: str = "count",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Cross-tabulation / pivot table of two categorical columns."""
    cats = _categorical_cols(df)
    if len(cats) < 2:
        return pd.DataFrame({"message": ["Need at least 2 categorical columns."]}), None

    if not row_col:
        row_col = cats[0]
    if not col_col:
        col_col = cats[1] if len(cats) > 1 else cats[0]

    if row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame({"message": ["Columns not found."]}), None

    if agg_func == "count" or not values_col or values_col not in df.columns:
        ct = pd.crosstab(df[row_col], df[col_col])
    else:
        ct = pd.crosstab(df[row_col], df[col_col], values=df[values_col], aggfunc=agg_func)

    fig = px.imshow(
        ct,
        text_auto=True,
        title=f"Cross-tabulation: {row_col} × {col_col}",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    return ct.reset_index(), fig


# ---------------------------------------------------------------------------
# 6. Distribution
# ---------------------------------------------------------------------------

def run_distribution(
    df: pd.DataFrame,
    meta: list[dict],
    column: str = "",
    bins: int = 30,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Histogram with optional KDE overlay for a numeric column."""
    nums = _numeric_cols(df)
    if not column or column not in nums:
        column = nums[0] if nums else ""
    if not column:
        return pd.DataFrame({"message": ["No numeric columns available."]}), None

    series = df[column].dropna()
    if series.empty:
        return pd.DataFrame({"message": [f"Column '{column}' has no non-null values."]}), None

    fig = go.Figure()

    # Histogram
    counts, bin_edges = np.histogram(series, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            name="Count",
            marker_color="steelblue",
            opacity=0.7,
        )
    )

    # KDE overlay (only if enough data points)
    if len(series) >= 5:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 200)
            kde_values = kde(x_range) * len(series) * (bin_edges[1] - bin_edges[0])
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode="lines",
                    name="KDE",
                    line={"color": "crimson", "width": 2},
                )
            )
        except Exception:
            pass  # KDE optional

    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Count",
        bargap=0.05,
    )

    stats_df = series.describe().reset_index()
    stats_df.columns = ["stat", column]
    return stats_df, fig


# ---------------------------------------------------------------------------
# 7. Null Analysis
# ---------------------------------------------------------------------------

def run_null_analysis(
    df: pd.DataFrame,
    meta: list[dict],
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Show null counts and percentages per column, sorted by null %."""
    total = len(df)
    null_counts = df.isnull().sum()
    result = pd.DataFrame(
        {
            "column": null_counts.index,
            "null_count": null_counts.values,
            "null_pct": (null_counts.values / total * 100).round(2),
            "non_null_count": total - null_counts.values,
        }
    ).sort_values("null_pct", ascending=False).reset_index(drop=True)

    fig = px.bar(
        result[result["null_pct"] > 0],
        x="column",
        y="null_pct",
        title="Null Percentage by Column",
        labels={"null_pct": "Null %", "column": "Column"},
        text_auto=".1f",
    )
    if fig.data:
        fig.update_layout(xaxis_tickangle=-45)
    return result, fig if fig.data else None


# ---------------------------------------------------------------------------
# 8. Data Types Summary
# ---------------------------------------------------------------------------

def run_dtypes(
    df: pd.DataFrame,
    meta: list[dict],
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Summary of column data types and cardinality."""
    rows = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        cat = "numeric" if "int" in dtype_str or "float" in dtype_str else "categorical"
        rows.append(
            {
                "column": col,
                "pandas_dtype": dtype_str,
                "category": cat,
                "unique_values": int(df[col].nunique(dropna=True)),
                "null_count": int(df[col].isna().sum()),
            }
        )
    result = pd.DataFrame(rows)

    type_counts = result["category"].value_counts().reset_index()
    type_counts.columns = ["category", "count"]
    fig = px.pie(type_counts, names="category", values="count", title="Column Type Distribution")
    return result, fig
