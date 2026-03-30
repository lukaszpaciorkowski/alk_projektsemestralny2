"""
chart_builder.py — build_chart() dispatcher for Ad Hoc Charts page.

Supported chart types: Bar, Line, Scatter, Box, Histogram, Heatmap,
Choropleth Map, Pie, Donut, Multi-Line, Area (Stacked), 3D Scatter,
Sunburst, Treemap, Bubble, Animated Bubble, Animated Bar.
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
    x_col: str | None = None,
    y_col: str | None = None,
    color_col: str | None = None,
    facet_col: str | None = None,
    agg_func: str = "mean",
    bins: int = 30,
    filters: list[Filter] | None = None,
    sample_limit: int = 50_000,
    location_mode: str = "country names",
    z_col: str | None = None,
    size_col: str | None = None,
    path_cols: list[str] | None = None,
    top_n: int = 10,
    hover_col: str | None = None,
    animation_col: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """
    Build a Plotly figure for the given chart configuration.

    Args:
        table_name: Dataset table name.
        engine: SQLAlchemy engine.
        chart_type: One of "Bar", "Line", "Scatter", "Box", "Histogram", "Heatmap",
                    "Choropleth Map", "Pie", "Donut", "Multi-Line", "Area (Stacked)",
                    "3D Scatter", "Sunburst", "Treemap", "Bubble",
                    "Animated Bubble", "Animated Bar".
        x_col: X-axis / category / location column.
        y_col: Y-axis / value column.
        color_col: Optional color / group-by column.
        facet_col: Optional facet column.
        agg_func: Aggregation function.
        bins: Bin count for Histogram.
        filters: Optional filter list.
        sample_limit: Max rows to load.
        location_mode: Plotly locationmode for Choropleth.
        z_col: Z-axis column for 3D Scatter.
        size_col: Size / bubble-size column.
        path_cols: Hierarchical path columns for Sunburst / Treemap.
        top_n: Max slices before grouping into "Other" (Pie / Donut).
        hover_col: Hover-name column for Bubble charts.
        animation_col: Animation-frame column for animated charts.
        log_x: Use log scale on X axis (Bubble charts).
        log_y: Use log scale on Y axis (Bubble charts).

    Returns:
        A Plotly Figure. On error, returns a figure with an annotation.
    """
    try:
        df = fetch_table(
            table_name,
            engine,
            filters=filters or [],
            limit=sample_limit,
        )

        if df.empty:
            return _error_fig("No data matched the current filters.")

        return _dispatch(
            df, chart_type, x_col, y_col, color_col, facet_col,
            agg_func, bins, location_mode,
            z_col=z_col, size_col=size_col, path_cols=path_cols or [], top_n=top_n,
            hover_col=hover_col, animation_col=animation_col,
            log_x=log_x, log_y=log_y,
        )

    except Exception as exc:
        logger.exception("build_chart failed")
        return _error_fig(str(exc))


def _dispatch(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None,
    y_col: str | None,
    color_col: str | None,
    facet_col: str | None,
    agg_func: str,
    bins: int,
    location_mode: str = "country names",
    z_col: str | None = None,
    size_col: str | None = None,
    path_cols: list[str] | None = None,
    top_n: int = 10,
    hover_col: str | None = None,
    animation_col: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """Dispatch to the correct chart builder."""
    ct = chart_type.lower()

    if ct == "histogram":
        return _histogram(df, x_col or "", bins, color_col)
    if ct == "scatter":
        return _scatter(df, x_col or "", y_col or x_col or "", color_col, facet_col)
    if ct == "box":
        return _box(df, x_col or "", y_col or x_col or "", color_col)
    if ct == "heatmap":
        return _heatmap(df, x_col or "", y_col or x_col or "", agg_func)
    if ct in ("bar", "line"):
        return _bar_or_line(df, x_col, y_col, color_col, facet_col, agg_func, ct)
    if ct == "choropleth map":
        return _choropleth(df, x_col or "", y_col or x_col or "", agg_func, location_mode)
    if ct in ("pie", "donut"):
        return _pie_or_donut(df, x_col or "", y_col, top_n, hole=0.4 if ct == "donut" else 0.0)
    if ct in ("multi-line", "area (stacked)"):
        return _multi_line_or_area(df, x_col or "", y_col or "", color_col, agg_func, ct)
    if ct == "3d scatter":
        return _scatter_3d(df, x_col or "", y_col or x_col or "", z_col or "", color_col, size_col)
    if ct == "sunburst":
        return _sunburst(df, path_cols or [], y_col)
    if ct == "treemap":
        return _treemap(df, path_cols or [], y_col)
    if ct == "bubble":
        return _bubble(df, x_col or "", y_col or "", size_col or "", color_col, hover_col,
                       log_x=log_x, log_y=log_y)
    if ct == "animated bubble":
        return _bubble(df, x_col or "", y_col or "", size_col or "", color_col, hover_col,
                       animation_col=animation_col, log_x=log_x, log_y=log_y)
    if ct == "animated bar":
        return _animated_bar(df, x_col or "", y_col or "", color_col, animation_col or "", agg_func)

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


def _pie_or_donut(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str | None,
    top_n: int,
    hole: float,
) -> go.Figure:
    if cat_col not in df.columns:
        return _error_fig(f"Category column '{cat_col}' not found.")

    if val_col and val_col in df.columns:
        agg = df.groupby(cat_col)[val_col].sum().reset_index()
        values_col = val_col
    else:
        agg = df[cat_col].value_counts().reset_index()
        agg.columns = [cat_col, "count"]
        values_col = "count"

    agg = agg.sort_values(values_col, ascending=False).reset_index(drop=True)
    if len(agg) > top_n:
        top = agg.head(top_n)
        other_val = agg.iloc[top_n:][values_col].sum()
        other_row = pd.DataFrame({cat_col: ["Other"], values_col: [other_val]})
        agg = pd.concat([top, other_row], ignore_index=True)

    kind = "Donut" if hole > 0 else "Pie"
    title = f"{kind}: {cat_col}" + (f" (sum of {val_col})" if val_col else " (counts)")
    fig = px.pie(agg, names=cat_col, values=values_col, hole=hole, title=title)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def _multi_line_or_area(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str | None,
    agg_func: str,
    chart_type: str,
) -> go.Figure:
    if x_col not in df.columns:
        return _error_fig(f"X column '{x_col}' not found.")
    if y_col not in df.columns:
        return _error_fig(f"Y column '{y_col}' not found.")

    group_cols = [x_col] + ([group_col] if group_col and group_col in df.columns else [])
    fn_map = {"mean": "mean", "sum": "sum", "count": "count",
              "min": "min", "max": "max", "median": "median"}
    fn = fn_map.get(agg_func, "mean")
    if fn == "count":
        plot_df = df.groupby(group_cols).size().reset_index(name=y_col)
    else:
        plot_df = df.groupby(group_cols)[y_col].agg(fn).reset_index()

    color = group_col if group_col and group_col in df.columns else None
    kind = "Area" if "area" in chart_type else "Multi-Line"
    title = f"{kind}: {agg_func}({y_col}) by {x_col}"
    if color:
        title += f" — grouped by {color}"

    kwargs: dict = dict(
        data_frame=plot_df, x=x_col, y=y_col, color=color,
        title=title, markers=True,
    )
    if "area" in chart_type:
        fig = px.area(**kwargs)
    else:
        fig = px.line(**kwargs)
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def _scatter_3d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str | None,
    size_col: str | None,
) -> go.Figure:
    for col, label in ((x_col, "X"), (y_col, "Y"), (z_col, "Z")):
        if not col:
            return _error_fig(f"{label} axis column is required for 3D Scatter.")
        if col not in df.columns:
            return _error_fig(f"Column '{col}' not found.")

    color = color_col if color_col and color_col in df.columns else None
    size = size_col if size_col and size_col in df.columns else None

    use_cols = [c for c in [x_col, y_col, z_col, color, size] if c]
    plot_df = df[use_cols].dropna()
    if plot_df.empty:
        return _error_fig("No data after dropping nulls.")

    # Clamp to 10k points for performance
    if len(plot_df) > 10_000:
        plot_df = plot_df.sample(10_000, random_state=42)

    # Ensure size column is non-negative (required by Plotly)
    if size:
        min_val = plot_df[size].min()
        if min_val < 0:
            plot_df = plot_df.copy()
            plot_df[size] = plot_df[size] - min_val

    fig = px.scatter_3d(
        plot_df, x=x_col, y=y_col, z=z_col,
        color=color, size=size,
        opacity=0.7,
        title=f"3D Scatter: {x_col} × {y_col} × {z_col}",
    )
    if not size:
        fig.update_traces(marker_size=4)
    return fig


def _sunburst(
    df: pd.DataFrame,
    path_cols: list[str],
    val_col: str | None,
) -> go.Figure:
    if not path_cols:
        return _error_fig("Select at least one path column for Sunburst.")
    missing = [c for c in path_cols if c not in df.columns]
    if missing:
        return _error_fig(f"Path columns not found: {missing}")

    if val_col and val_col in df.columns:
        plot_df = df[path_cols + [val_col]].copy()
    else:
        plot_df = df[path_cols].copy()
        plot_df["_count"] = 1
        val_col = "_count"

    # Convert path cols to string for Plotly compatibility
    for c in path_cols:
        plot_df[c] = plot_df[c].fillna("(blank)").astype(str)

    plot_df = plot_df[plot_df[val_col] > 0] if pd.api.types.is_numeric_dtype(plot_df[val_col]) else plot_df

    fig = px.sunburst(
        plot_df,
        path=path_cols,
        values=val_col,
        title=f"Sunburst: {' > '.join(path_cols)}",
    )
    fig.update_traces(textinfo="label+percent entry")
    return fig


def _treemap(
    df: pd.DataFrame,
    path_cols: list[str],
    val_col: str | None,
) -> go.Figure:
    if not path_cols:
        return _error_fig("Select at least one path column for Treemap.")
    missing = [c for c in path_cols if c not in df.columns]
    if missing:
        return _error_fig(f"Path columns not found: {missing}")

    if val_col and val_col in df.columns:
        plot_df = df[path_cols + [val_col]].copy()
    else:
        plot_df = df[path_cols].copy()
        plot_df["_count"] = 1
        val_col = "_count"

    for c in path_cols:
        plot_df[c] = plot_df[c].fillna("(blank)").astype(str)

    plot_df = plot_df[plot_df[val_col] > 0] if pd.api.types.is_numeric_dtype(plot_df[val_col]) else plot_df

    fig = px.treemap(
        plot_df,
        path=path_cols,
        values=val_col,
        title=f"Treemap: {' > '.join(path_cols)}",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textinfo="label+value+percent entry")
    return fig


def _bubble(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    color_col: str | None,
    hover_col: str | None,
    animation_col: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    for col, label in ((x_col, "X"), (y_col, "Y"), (size_col, "Size")):
        if not col:
            return _error_fig(f"{label} column is required for Bubble chart.")
        if col not in df.columns:
            return _error_fig(f"Column '{col}' not found.")

    # Determine which columns to keep before dropping nulls
    use_cols = [c for c in [x_col, y_col, size_col, color_col, hover_col, animation_col]
                if c and c in df.columns]
    required = [x_col, y_col, size_col]
    plot_df = df[use_cols].dropna(subset=required)

    if plot_df.empty:
        return _error_fig("No data after dropping nulls in X, Y, or Size columns.")

    # Size must be non-negative
    if plot_df[size_col].min() < 0:
        plot_df = plot_df.copy()
        plot_df[size_col] = plot_df[size_col] - plot_df[size_col].min()

    kwargs: dict = {
        "data_frame": plot_df,
        "x":          x_col,
        "y":          y_col,
        "size":       size_col,
        "size_max":   60,
        "log_x":      log_x,
        "log_y":      log_y,
        "opacity":    0.7,
    }
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    if hover_col and hover_col in df.columns:
        kwargs["hover_name"] = hover_col
    if animation_col and animation_col in df.columns:
        # Sort by animation column so frames appear in order
        plot_df = plot_df.sort_values(animation_col)
        kwargs["data_frame"] = plot_df
        kwargs["animation_frame"] = animation_col
        # Fix axis ranges so they don't jump between frames
        x_vals = pd.to_numeric(plot_df[x_col], errors="coerce").dropna()
        y_vals = pd.to_numeric(plot_df[y_col], errors="coerce").dropna()
        if not x_vals.empty and not y_vals.empty:
            x_pad = (x_vals.max() - x_vals.min()) * 0.1 or x_vals.max() * 0.1 or 1
            y_pad = (y_vals.max() - y_vals.min()) * 0.1 or y_vals.max() * 0.1 or 1
            kwargs["range_x"] = [max(0, x_vals.min() - x_pad) if not log_x else x_vals.min() * 0.9,
                                  x_vals.max() + x_pad]
            kwargs["range_y"] = [max(0, y_vals.min() - y_pad) if not log_y else y_vals.min() * 0.9,
                                  y_vals.max() + y_pad]

    title_parts = [f"Bubble: {x_col} vs {y_col} (size={size_col})"]
    if animation_col:
        title_parts.append(f"animated by {animation_col}")
    kwargs["title"] = " — ".join(title_parts)

    fig = px.scatter(**kwargs)
    return fig


def _animated_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None,
    animation_col: str,
    agg_func: str,
) -> go.Figure:
    for col, label in ((x_col, "X"), (y_col, "Y"), (animation_col, "Animation frame")):
        if not col:
            return _error_fig(f"{label} column is required for Animated Bar.")
        if col not in df.columns:
            return _error_fig(f"Column '{col}' not found.")

    # Aggregate: group by [animation_col, x_col, color_col]
    group_cols = [animation_col, x_col]
    if color_col and color_col in df.columns:
        group_cols.append(color_col)

    fn_map = {"mean": "mean", "sum": "sum", "count": "count",
              "min": "min", "max": "max", "median": "median"}
    fn = fn_map.get(agg_func, "mean")

    if fn == "count":
        plot_df = df.groupby(group_cols).size().reset_index(name=y_col)
    else:
        plot_df = df.groupby(group_cols)[y_col].agg(fn).reset_index()

    plot_df = plot_df.sort_values(animation_col)

    y_max = pd.to_numeric(plot_df[y_col], errors="coerce").max()
    y_range = [0, y_max * 1.1 if y_max and y_max > 0 else 1]

    color = color_col if color_col and color_col in df.columns else None
    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        color=color,
        animation_frame=animation_col,
        barmode="group" if color else "relative",
        range_y=y_range,
        title=f"Animated Bar: {agg_func}({y_col}) by {x_col} over {animation_col}",
    )
    fig.update_layout(xaxis_tickangle=-45)
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
