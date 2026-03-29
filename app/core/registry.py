"""
registry.py — Analytics function registry.

Single source of truth for all analytical functions.
Generic functions work on every dataset.
Specialized functions are gated on dataset_type.

Usage:
    from app.core.registry import REGISTRY, get_functions_for, AnalyticsFunction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.analytics.generic import (
    run_correlation,
    run_crosstab,
    run_describe,
    run_distribution,
    run_dtypes,
    run_groupby,
    run_null_analysis,
    run_value_counts,
)
from app.core.analytics.diabetes import (
    run_hba1c_vs_readmission,
    run_los_by_readmission,
    run_medication_frequency,
    run_medications_vs_los,
    run_readmission_by_group,
    run_top_diagnoses,
)


@dataclass
class ParamSpec:
    """Specification for a single UI parameter."""

    name: str
    widget: str          # "select"|"select_column"|"bool"|"int"|"multiselect_column"
    default: Any
    label: str = ""
    options: list = field(default_factory=list)
    dtype_filter: str | None = None  # "categorical"|"numeric"|"datetime"|None


@dataclass
class AnalyticsFunction:
    """Descriptor for a registered analytics function."""

    id: str
    label: str
    scope: str                     # "generic" | "diabetes" | ...
    description: str
    params: list[ParamSpec]
    fn: Callable
    has_chart: bool
    chart_type: str | None         # "bar"|"heatmap"|"hist"|"scatter"|None
    requires_enrichment: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, AnalyticsFunction] = {
    # ── Generic ──────────────────────────────────────────────────────────────
    "generic.describe": AnalyticsFunction(
        id="generic.describe",
        label="Descriptive Statistics",
        scope="generic",
        description="pandas .describe(include='all') for all columns — counts, mean, std, min/max.",
        params=[],
        fn=run_describe,
        has_chart=False,
        chart_type=None,
    ),
    "generic.correlation": AnalyticsFunction(
        id="generic.correlation",
        label="Correlation Matrix",
        scope="generic",
        description="Pearson / Spearman / Kendall correlation heatmap (numeric columns only).",
        params=[
            ParamSpec(
                name="method",
                widget="select",
                default="pearson",
                label="Correlation method",
                options=["pearson", "spearman", "kendall"],
            ),
        ],
        fn=run_correlation,
        has_chart=True,
        chart_type="heatmap",
    ),
    "generic.value_counts": AnalyticsFunction(
        id="generic.value_counts",
        label="Value Counts",
        scope="generic",
        description="Frequency table for a categorical column.",
        params=[
            ParamSpec(
                name="column",
                widget="select_column",
                default="",
                label="Column",
                dtype_filter="categorical",
            ),
            ParamSpec(
                name="top_n",
                widget="int",
                default=20,
                label="Top N values",
            ),
        ],
        fn=run_value_counts,
        has_chart=True,
        chart_type="bar",
    ),
    "generic.groupby": AnalyticsFunction(
        id="generic.groupby",
        label="Group By / Aggregate",
        scope="generic",
        description="Group by a categorical column and aggregate a numeric column.",
        params=[
            ParamSpec(
                name="group_col",
                widget="select_column",
                default="",
                label="Group by",
                dtype_filter="categorical",
            ),
            ParamSpec(
                name="agg_col",
                widget="select_column",
                default="",
                label="Aggregate column",
                dtype_filter="numeric",
            ),
            ParamSpec(
                name="agg_func",
                widget="select",
                default="mean",
                label="Aggregation function",
                options=["mean", "sum", "count", "min", "max", "median"],
            ),
        ],
        fn=run_groupby,
        has_chart=True,
        chart_type="bar",
    ),
    "generic.crosstab": AnalyticsFunction(
        id="generic.crosstab",
        label="Cross-tabulation",
        scope="generic",
        description="Pivot table of two categorical columns (counts or aggregated numeric).",
        params=[
            ParamSpec(
                name="row_col",
                widget="select_column",
                default="",
                label="Row column",
                dtype_filter="categorical",
            ),
            ParamSpec(
                name="col_col",
                widget="select_column",
                default="",
                label="Column column",
                dtype_filter="categorical",
            ),
        ],
        fn=run_crosstab,
        has_chart=True,
        chart_type="heatmap",
    ),
    "generic.distribution": AnalyticsFunction(
        id="generic.distribution",
        label="Distribution",
        scope="generic",
        description="Histogram with KDE overlay for a numeric column.",
        params=[
            ParamSpec(
                name="column",
                widget="select_column",
                default="",
                label="Column",
                dtype_filter="numeric",
            ),
            ParamSpec(
                name="bins",
                widget="int",
                default=30,
                label="Number of bins",
            ),
        ],
        fn=run_distribution,
        has_chart=True,
        chart_type="hist",
    ),
    "generic.null_analysis": AnalyticsFunction(
        id="generic.null_analysis",
        label="Null Analysis",
        scope="generic",
        description="Bar chart of null percentage per column.",
        params=[],
        fn=run_null_analysis,
        has_chart=True,
        chart_type="bar",
    ),
    "generic.dtypes": AnalyticsFunction(
        id="generic.dtypes",
        label="Data Types",
        scope="generic",
        description="Column type summary: dtype, cardinality, null counts.",
        params=[],
        fn=run_dtypes,
        has_chart=True,
        chart_type="bar",
    ),

    # ── Diabetes ─────────────────────────────────────────────────────────────
    "diabetes.readmission_by_group": AnalyticsFunction(
        id="diabetes.readmission_by_group",
        label="Readmission Rate by Group",
        scope="diabetes",
        description="Readmission percentage broken down by any categorical column.",
        params=[
            ParamSpec(
                name="group_by",
                widget="select_column",
                default="age",
                label="Group by",
                dtype_filter="categorical",
            ),
            ParamSpec(
                name="readmission_binary",
                widget="bool",
                default=True,
                label="Binary readmission (<30 and >30 = positive)",
            ),
        ],
        fn=run_readmission_by_group,
        has_chart=True,
        chart_type="bar",
        requires_enrichment=False,
    ),
    "diabetes.hba1c_vs_readmission": AnalyticsFunction(
        id="diabetes.hba1c_vs_readmission",
        label="HbA1c vs Readmission",
        scope="diabetes",
        description="Readmission rate by HbA1c test result (A1Cresult column).",
        params=[
            ParamSpec(
                name="readmission_binary",
                widget="bool",
                default=True,
                label="Binary readmission",
            ),
        ],
        fn=run_hba1c_vs_readmission,
        has_chart=True,
        chart_type="bar",
        requires_enrichment=False,
    ),
    "diabetes.top_diagnoses": AnalyticsFunction(
        id="diabetes.top_diagnoses",
        label="Top Diagnoses by Readmission",
        scope="diabetes",
        description="Top N primary ICD-9 diagnoses ranked by readmission rate. Requires enrichment.",
        params=[
            ParamSpec(
                name="top_n",
                widget="int",
                default=10,
                label="Top N diagnoses",
            ),
        ],
        fn=run_top_diagnoses,
        has_chart=True,
        chart_type="bar",
        requires_enrichment=True,
    ),
    "diabetes.medication_frequency": AnalyticsFunction(
        id="diabetes.medication_frequency",
        label="Medication Frequency",
        scope="diabetes",
        description="Most prescribed medications from unpivoted medication data. Requires enrichment.",
        params=[
            ParamSpec(
                name="top_n",
                widget="int",
                default=15,
                label="Top N medications",
            ),
        ],
        fn=run_medication_frequency,
        has_chart=True,
        chart_type="bar",
        requires_enrichment=True,
    ),
    "diabetes.los_by_readmission": AnalyticsFunction(
        id="diabetes.los_by_readmission",
        label="Length of Stay by Readmission",
        scope="diabetes",
        description="Mean, median, min, max length of stay broken down by readmission class.",
        params=[],
        fn=run_los_by_readmission,
        has_chart=True,
        chart_type="bar",
        requires_enrichment=False,
    ),
    "diabetes.medications_vs_los": AnalyticsFunction(
        id="diabetes.medications_vs_los",
        label="Medications vs LOS",
        scope="diabetes",
        description="Scatter plot of number of medications vs mean length of stay.",
        params=[],
        fn=run_medications_vs_los,
        has_chart=True,
        chart_type="scatter",
        requires_enrichment=False,
    ),
}


def get_functions_for(dataset_type: str) -> list[AnalyticsFunction]:
    """
    Return all functions applicable to the given dataset type.

    Generic functions come first, then specialized functions for the type.
    """
    generic = [fn for fn in REGISTRY.values() if fn.scope == "generic"]
    specialized = [fn for fn in REGISTRY.values() if fn.scope == dataset_type]
    return generic + specialized
