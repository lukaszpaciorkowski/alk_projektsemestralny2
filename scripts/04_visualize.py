"""
04_visualize.py — Generate all visualizations for the diabetes analysis.

Produces 6 figures using both Plotly (for Streamlit) and Matplotlib/Seaborn
(for PDF embedding). PNG files are saved to outputs/figures/.

Usage:
    python scripts/04_visualize.py [--config config.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats as scipy_stats
from sqlalchemy.engine import Engine

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.query_helpers import (
    get_engine,
    hba1c_vs_readmission,
    load_config,
    los_by_readmission,
    medications_vs_los,
    readmission_by_group,
    top_diagnoses_by_readmission,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_mpl_fig(fig: plt.Figure, filename: str, figures_dir: str, dpi: int) -> str:
    """Save matplotlib figure to disk and close it."""
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure 1 — Readmission Rate by Age Group
# ---------------------------------------------------------------------------

def fig_01_readmission_by_age(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
) -> go.Figure:
    """Bar chart of readmission percentage by age group."""
    df = readmission_by_group(engine, group_col="age_group")
    if df.empty:
        logger.warning("fig_01: no data, returning empty figure.")
        return go.Figure()

    df_readmitted = df[df["readmission"].isin(["<30", ">30"])].copy()
    total = df.groupby("group_value")["count"].sum().reset_index()
    total.columns = ["group_value", "total"]
    readmit_sum = df_readmitted.groupby("group_value")["count"].sum().reset_index()
    readmit_sum.columns = ["group_value", "readmitted"]
    merged = total.merge(readmit_sum, on="group_value", how="left").fillna(0)
    merged["readmission_rate"] = merged["readmitted"] / merged["total"] * 100
    merged = merged.sort_values("group_value")

    plotly_fig = px.bar(
        merged,
        x="group_value",
        y="readmission_rate",
        labels={"group_value": "Age Group", "readmission_rate": "Readmission Rate (%)"},
        title="Readmission Rate by Age Group",
        color="readmission_rate",
        color_continuous_scale=palette,
    )
    plotly_fig.update_layout(showlegend=False, coloraxis_showscale=False)

    # Matplotlib version for PDF
    colors = sns.color_palette(palette, len(merged))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(merged["group_value"], merged["readmission_rate"], color=colors)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title("Readmission Rate by Age Group")
    ax.tick_params(axis="x", rotation=20)
    _save_mpl_fig(fig, "fig_01_readmission_by_age.png", figures_dir, dpi)

    return plotly_fig


# ---------------------------------------------------------------------------
# Figure 2 — Readmission by Admission Type (stacked bar)
# ---------------------------------------------------------------------------

def fig_02_readmission_by_admission_type(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
) -> go.Figure:
    """Stacked bar chart of readmission outcomes by admission type."""
    df = readmission_by_group(engine, group_col="admission_type_id")
    if df.empty:
        logger.warning("fig_02: no data, returning empty figure.")
        return go.Figure()

    df = df.copy()
    df["group_value"] = df["group_value"].astype(str)

    plotly_fig = px.bar(
        df,
        x="group_value",
        y="count",
        color="readmission",
        barmode="stack",
        labels={"group_value": "Admission Type ID", "count": "Encounters"},
        title="Readmission Outcomes by Admission Type",
    )

    pivot = df.pivot_table(
        index="group_value", columns="readmission", values="count", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap=palette)
    ax.set_xlabel("Admission Type ID")
    ax.set_ylabel("Encounters")
    ax.set_title("Readmission Outcomes by Admission Type")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Readmission", bbox_to_anchor=(1.05, 1))
    _save_mpl_fig(fig, "fig_02_readmission_by_admission_type.png", figures_dir, dpi)

    return plotly_fig


# ---------------------------------------------------------------------------
# Figure 3 — Length of Stay Distribution by Readmission Class (bar)
# ---------------------------------------------------------------------------

def fig_03_los_distribution(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
) -> go.Figure:
    """Bar chart of mean length of stay by readmission class."""
    df = los_by_readmission(engine)
    if df.empty:
        logger.warning("fig_03: no data, returning empty figure.")
        return go.Figure()

    plotly_fig = px.bar(
        df,
        x="group_value",
        y="mean_los",
        labels={"group_value": "Readmission Class", "mean_los": "Mean Length of Stay (days)"},
        title="Mean Length of Stay by Readmission Class",
        color="mean_los",
        color_continuous_scale=palette,
    )
    plotly_fig.update_layout(coloraxis_showscale=False)

    colors = sns.color_palette(palette, len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["group_value"], df["mean_los"], color=colors)
    ax.set_xlabel("Readmission Class")
    ax.set_ylabel("Mean Length of Stay (days)")
    ax.set_title("Mean Length of Stay by Readmission Class")
    _save_mpl_fig(fig, "fig_03_los_distribution.png", figures_dir, dpi)

    return plotly_fig


# ---------------------------------------------------------------------------
# Figure 4 — Top Diagnoses by Readmission Rate (horizontal bar)
# ---------------------------------------------------------------------------

def fig_04_top_diagnoses(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
    top_n: int = 10,
) -> go.Figure:
    """Horizontal bar chart of top N diagnoses by readmission rate."""
    df = top_diagnoses_by_readmission(engine, top_n=top_n)
    if df.empty:
        logger.warning("fig_04: no data, returning empty figure.")
        return go.Figure()

    df_readmitted = df[df["readmission"].isin(["<30", ">30"])].copy()
    total_by_code = df.groupby("icd9_code")["count"].sum().reset_index()
    total_by_code.columns = ["icd9_code", "total"]
    readmit_by_code = df_readmitted.groupby("icd9_code")["count"].sum().reset_index()
    readmit_by_code.columns = ["icd9_code", "readmitted"]
    merged = total_by_code.merge(readmit_by_code, on="icd9_code", how="left").fillna(0)
    merged["rate"] = merged["readmitted"] / merged["total"] * 100
    merged = merged.sort_values("rate", ascending=True)

    plotly_fig = px.bar(
        merged,
        x="rate",
        y="icd9_code",
        orientation="h",
        labels={"rate": "Readmission Rate (%)", "icd9_code": "ICD-9 Code"},
        title=f"Top {top_n} Diagnoses by Readmission Rate",
        color="rate",
        color_continuous_scale=palette,
    )
    plotly_fig.update_layout(coloraxis_showscale=False)

    colors = sns.color_palette(palette, len(merged))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(merged["icd9_code"], merged["rate"], color=colors)
    ax.set_xlabel("Readmission Rate (%)")
    ax.set_ylabel("ICD-9 Code")
    ax.set_title(f"Top {top_n} Diagnoses by Readmission Rate")
    _save_mpl_fig(fig, "fig_04_top_diagnoses.png", figures_dir, dpi)

    return plotly_fig


# ---------------------------------------------------------------------------
# Figure 5 — HbA1c Result vs Readmission (grouped bar)
# ---------------------------------------------------------------------------

def fig_05_hba1c_vs_readmission(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
) -> go.Figure:
    """Grouped bar chart: HbA1c test result vs readmission outcome."""
    df = hba1c_vs_readmission(engine)
    if df.empty:
        logger.warning("fig_05: no data, returning empty figure.")
        return go.Figure()

    plotly_fig = px.bar(
        df,
        x="hba1c_result",
        y="rate",
        color="readmission",
        barmode="group",
        labels={
            "hba1c_result": "HbA1c Result",
            "rate": "Rate",
            "readmission": "Readmission",
        },
        title="HbA1c Result vs Readmission Outcome",
    )

    pivot = df.pivot_table(
        index="hba1c_result", columns="readmission", values="rate", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, colormap=palette)
    ax.set_xlabel("HbA1c Result")
    ax.set_ylabel("Rate")
    ax.set_title("HbA1c Result vs Readmission Outcome")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Readmission", bbox_to_anchor=(1.05, 1))
    _save_mpl_fig(fig, "fig_05_hba1c_vs_readmission.png", figures_dir, dpi)

    return plotly_fig


# ---------------------------------------------------------------------------
# Figure 6 — Medications vs Length of Stay (scatter + regression)
# ---------------------------------------------------------------------------

def fig_06_medications_vs_los(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
) -> go.Figure:
    """Scatter plot with regression line: num_medications vs mean LOS."""
    df = medications_vs_los(engine)
    if df.empty:
        logger.warning("fig_06: no data, returning empty figure.")
        return go.Figure()

    x = df["num_medications"].values.astype(float)
    y = df["mean_los"].values.astype(float)

    slope, intercept, r_value, _p_value, _std_err = scipy_stats.linregress(x, y)
    y_pred = slope * x + intercept

    plotly_fig = go.Figure()
    plotly_fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=np.clip(df["count"] / df["count"].max() * 20 + 5, 5, 30),
                color=x,
                colorscale=palette,
                showscale=True,
                colorbar=dict(title="Num Meds"),
            ),
            name="Data",
            text=[f"n={n}" for n in df["count"]],
        )
    )
    plotly_fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            line=dict(color="red", dash="dash"),
            name=f"Regression (r²={r_value**2:.3f})",
        )
    )
    plotly_fig.update_layout(
        title="Number of Medications vs Mean Length of Stay",
        xaxis_title="Number of Medications",
        yaxis_title="Mean Length of Stay (days)",
    )

    sizes = np.clip(df["count"] / df["count"].max() * 200 + 20, 20, 250)
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(x, y, s=sizes, c=x, cmap=palette, alpha=0.7)
    ax.plot(x, y_pred, "r--", label=f"Regression r²={r_value**2:.3f}")
    plt.colorbar(sc, ax=ax, label="Num Medications")
    ax.set_xlabel("Number of Medications")
    ax.set_ylabel("Mean Length of Stay (days)")
    ax.set_title("Number of Medications vs Mean Length of Stay")
    ax.legend()
    _save_mpl_fig(fig, "fig_06_medications_vs_los.png", figures_dir, dpi)

    return plotly_fig


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    engine: Engine,
    figures_dir: str,
    dpi: int = 150,
    palette: str = "viridis",
    top_n: int = 10,
) -> dict[str, go.Figure]:
    """Generate all 6 figures and return them keyed by name."""
    generators: list[tuple[str, object]] = [
        ("fig_01", lambda: fig_01_readmission_by_age(engine, figures_dir, dpi, palette)),
        ("fig_02", lambda: fig_02_readmission_by_admission_type(engine, figures_dir, dpi, palette)),
        ("fig_03", lambda: fig_03_los_distribution(engine, figures_dir, dpi, palette)),
        ("fig_04", lambda: fig_04_top_diagnoses(engine, figures_dir, dpi, palette, top_n)),
        ("fig_05", lambda: fig_05_hba1c_vs_readmission(engine, figures_dir, dpi, palette)),
        ("fig_06", lambda: fig_06_medications_vs_los(engine, figures_dir, dpi, palette)),
    ]
    figs: dict[str, go.Figure] = {}
    for name, gen_fn in generators:
        try:
            figs[name] = gen_fn()
            logger.info("Generated %s", name)
        except Exception as exc:
            logger.error("Failed to generate %s: %s", name, exc)
            figs[name] = go.Figure()
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all diabetes analysis figures.")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config)
    pipeline = config["pipeline"]
    output = config["output"]

    generate_all_figures(
        engine=engine,
        figures_dir=output["figures_dir"],
        dpi=output["dpi"],
        palette=pipeline["palette"],
        top_n=pipeline["top_n_diagnoses"],
    )
    logger.info("All figures generated.")


if __name__ == "__main__":
    main()
