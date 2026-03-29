"""
chart_builder.py — build_chart() dispatcher for Ad Hoc Charts page.

Supported chart types: Bar, Line, Scatter, Box, Histogram, Heatmap, Choropleth Map.
Never raises — catches all plotly errors and returns an annotated figure.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.query import Filter, fetch_table

logger = logging.getLogger(__name__)


def build_chart(
    table_name: str,
    engine: Engine,
    chart_type: str,
    x_col: str,
    y_col: str | None = None,
    color_col: str | None = None,
    facet_col: str | None = None,
    agg_func: str = "mean",
    bins: int = 30,
    filters: list[Filter] | None = None,
    sample_limit: int = 50_000,
    location_mode: str = "country names",
) -> go.Figure:
    """
    Build a Plotly figure for the given chart configuration.

    Args:
        table_name: Dataset table name.
        engine: SQLAlchemy engine.
        chart_type: One of "Bar", "Line", "Scatter", "Box", "Histogram", "Heatmap",
                    "Choropleth Map".
        x_col: X-axis / location column.
        y_col: Y-axis / value column (None for Histogram).
        color_col: Optional color / group-by column.
        facet_col: Optional facet column.
        agg_func: Aggregation function for Bar / Line / Heatmap / Choropleth.
        bins: Bin count for Histogram.
        filters: Optional filter list.
        sample_limit: Max rows to load (avoids OOM on huge datasets).
        location_mode: Plotly locationmode for Choropleth ("country names" or "ISO-3").

    Returns:
        A Plotly Figure. On error, returns a figure with an annotation.
    """
    try:
        # Load data
        df = fetch_table(
            table_name,
            engine,
            filters=filters or [],
            limit=sample_limit,
        )

        if df.empty:
            return _error_fig("No data matched the current filters.")

        return _dispatch(df, chart_type, x_col, y_col, color_col, facet_col,
                         agg_func, bins, location_mode)

    except Exception as exc:
        logger.exception("build_chart failed")
        return _error_fig(str(exc))


def _dispatch(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str | None,
    color_col: str | None,
    facet_col: str | None,
    agg_func: str,
    bins: int,
    location_mode: str = "country names",
) -> go.Figure:
    """Dispatch to the correct chart builder."""
    ct = chart_type.lower()

    if ct == "histogram":
        return _histogram(df, x_col, bins, color_col)
    if ct == "scatter":
        return _scatter(df, x_col, y_col or x_col, color_col, facet_col)
    if ct == "box":
        return _box(df, x_col, y_col or x_col, color_col)
    if ct == "heatmap":
        return _heatmap(df, x_col, y_col or x_col, agg_func)
    if ct in ("bar", "line"):
        return _bar_or_line(df, x_col, y_col, color_col, facet_col, agg_func, ct)
    if ct == "choropleth map":
        return _choropleth(df, x_col, y_col or x_col, agg_func, location_mode)

    return _error_fig(f"Unknown chart type: {chart_type!r}")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _agg_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None,
    agg_func: str,
) -> pd.DataFrame:
    """Aggregate df by x_col (and optionally color_col) with the given function."""
    group_cols = [x_col] + ([color_col] if color_col and color_col in df.columns else [])
    fn_map = {
        "mean": "mean", "sum": "sum", "count": "count",
        "min": "min", "max": "max", "median": "median",
    }
    fn = fn_map.get(agg_func, "mean")
    if fn == "count":
        result = df.groupby(group_cols).size().reset_index(name=f"count_{y_col}")
        result.rename(columns={f"count_{y_col}": y_col}, inplace=True)
    else:
        result = df.groupby(group_cols)[y_col].agg(fn).reset_index()
    return result


def _bar_or_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    color_col: str | None,
    facet_col: str | None,
    agg_func: str,
    chart_type: str,
) -> go.Figure:
    if not y_col or y_col not in df.columns:
        return _error_fig(f"Y column '{y_col}' not found.")
    if x_col not in df.columns:
        return _error_fig(f"X column '{x_col}' not found.")

    plot_df = _agg_df(df, x_col, y_col, color_col, agg_func)
    color = color_col if color_col and color_col in df.columns else None
    facet = facet_col if facet_col and facet_col in df.columns else None

    kwargs = {
        "data_frame": plot_df,
        "x": x_col,
        "y": y_col,
        "color": color,
        "facet_col": facet,
        "title": f"{agg_func.capitalize()}({y_col}) by {x_col}",
    }
    if chart_type == "bar":
        fig = px.bar(**kwargs, barmode="group" if color else "relative", text_auto=".2s")
    else:
        fig = px.line(**kwargs, markers=True)

    fig.update_layout(xaxis_tickangle=-45)
    return fig


def _scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None,
    facet_col: str | None,
) -> go.Figure:
    if x_col not in df.columns or y_col not in df.columns:
        return _error_fig("X or Y column not found.")
    color = color_col if color_col and color_col in df.columns else None
    facet = facet_col if facet_col and facet_col in df.columns else None
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        facet_col=facet,
        title=f"Scatter: {x_col} vs {y_col}",
        opacity=0.6,
    )
    return fig


def _box(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None) -> go.Figure:
    color = color_col if color_col and color_col in df.columns else None
    fig = px.box(
        df,
        x=x_col if x_col in df.columns else None,
        y=y_col if y_col in df.columns else None,
        color=color,
        title=f"Box Plot: {y_col} by {x_col}",
    )
    return fig


def _histogram(df: pd.DataFrame, x_col: str, bins: int, color_col: str | None) -> go.Figure:
    if x_col not in df.columns:
        return _error_fig(f"Column '{x_col}' not found.")
    color = color_col if color_col and color_col in df.columns else None
    fig = px.histogram(
        df,
        x=x_col,
        nbins=bins,
        color=color,
        title=f"Histogram: {x_col}",
        opacity=0.8,
    )
    return fig


def _heatmap(df: pd.DataFrame, x_col: str, y_col: str, agg_func: str) -> go.Figure:
    if x_col not in df.columns or y_col not in df.columns:
        return _error_fig("X or Y column not found for heatmap.")
    ct = pd.crosstab(df[y_col], df[x_col])
    fig = px.imshow(
        ct,
        text_auto=True,
        title=f"Heatmap: {y_col} × {x_col}",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    return fig


def _auto_location_mode(series: pd.Series) -> str:
    """Guess whether values are ISO-3 codes or country names."""
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return "country names"
    avg_len = sample.str.len().mean()
    pct_upper = sample.str.isupper().mean()
    if avg_len <= 3.5 and pct_upper >= 0.7:
        return "ISO-3"
    return "country names"


def _choropleth(
    df: pd.DataFrame,
    location_col: str,
    value_col: str,
    agg_func: str,
    location_mode: str,
) -> go.Figure:
    if location_col not in df.columns:
        return _error_fig(f"Location column '{location_col}' not found.")
    if value_col not in df.columns:
        return _error_fig(f"Value column '{value_col}' not found.")

    # Auto-detect location mode if not explicitly overridden
    if location_mode == "auto":
        location_mode = _auto_location_mode(df[location_col])

    # Aggregate per location
    fn_map = {"mean": "mean", "sum": "sum", "count": "count",
               "min": "min", "max": "max", "median": "median"}
    fn = fn_map.get(agg_func, "mean")
    if fn == "count":
        agg_df = df.groupby(location_col).size().reset_index(name=value_col)
    else:
        agg_df = df.groupby(location_col)[value_col].agg(fn).reset_index()
    agg_df = agg_df.dropna(subset=[value_col])

    if agg_df.empty:
        return _error_fig("No data after aggregation.")

    plotly_mode = "ISO-3" if location_mode == "ISO-3" else "country names"

    fig = px.choropleth(
        agg_df,
        locations=location_col,
        color=value_col,
        locationmode=plotly_mode,
        color_continuous_scale="Viridis",
        title=f"Choropleth: {agg_func}({value_col}) by {location_col}",
        projection="natural earth",
    )
    fig.update_layout(
        coloraxis_colorbar={"title": f"{agg_func}({value_col})"},
        margin={"r": 0, "l": 0, "t": 40, "b": 0},
        height=500,
    )
    return fig


def _error_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart error: {message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "red"},
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=300,
    )
    return fig
