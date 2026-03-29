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


# ---------------------------------------------------------------------------
# 9. Principal Component Analysis
# ---------------------------------------------------------------------------

def run_pca(
    df: pd.DataFrame,
    meta: list[dict],
    n_components: int = 2,
    scale: bool = True,
    x_component: int = 1,
    y_component: int = 2,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """PCA on numeric columns. Returns variance summary + biplot scatter."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise RuntimeError(
            "scikit-learn is required for PCA. Install it with: pip install scikit-learn"
        )

    num_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    num_df = num_df.loc[num_df.dropna().index]  # rows with no nulls across numeric cols

    if num_df.shape[1] < 2:
        raise ValueError("PCA requires at least 2 numeric columns with non-null values.")

    n_components = min(n_components, num_df.shape[1], num_df.shape[0])

    # Clamp component indices to valid range
    xi = max(1, min(int(x_component), n_components)) - 1  # 0-based
    yi = max(1, min(int(y_component), n_components)) - 1
    if xi == yi:
        yi = (xi + 1) % n_components  # ensure different axes

    X = num_df.values
    if scale:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)  # shape: (n_samples, n_components)

    # Variance summary table (all components)
    cumulative = 0.0
    summary_rows = []
    for i, (var, ratio) in enumerate(
        zip(pca.explained_variance_, pca.explained_variance_ratio_), start=1
    ):
        cumulative += ratio
        summary_rows.append(
            {
                "component": f"PC{i}",
                "explained_variance": round(float(var), 4),
                "explained_variance_ratio": round(float(ratio), 4),
                "cumulative_ratio": round(cumulative, 4),
            }
        )
    result_df = pd.DataFrame(summary_rows)

    # Biplot: scores scatter + loading arrows
    fig = None
    if n_components >= 2:
        pc_x_label = f"PC{xi + 1}"
        pc_y_label = f"PC{yi + 1}"
        x_var = pca.explained_variance_ratio_[xi] * 100
        y_var = pca.explained_variance_ratio_[yi] * 100
        total_var = x_var + y_var

        scatter_df = pd.DataFrame(
            {pc_x_label: scores[:, xi], pc_y_label: scores[:, yi]}
        )

        # Colour by first categorical column of the original frame if available
        cat_cols = _categorical_cols(df.loc[num_df.index])
        color_col = cat_cols[0] if cat_cols else None
        if color_col:
            scatter_df[color_col] = df.loc[num_df.index, color_col].values

        fig = px.scatter(
            scatter_df,
            x=pc_x_label,
            y=pc_y_label,
            color=color_col,
            opacity=0.5,
            title=f"PCA Biplot — {pc_x_label} vs {pc_y_label} ({total_var:.1f}% variance explained)",
            labels={
                pc_x_label: f"{pc_x_label} ({x_var:.1f}%)",
                pc_y_label: f"{pc_y_label} ({y_var:.1f}%)",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker_size=4, selector=dict(mode="markers"))

        # Loading arrows (biplot)
        loadings_x = pca.components_[xi]  # shape: (n_features,)
        loadings_y = pca.components_[yi]

        # Scale arrows to fit nicely in score space
        score_scale = max(
            float(np.abs(scores[:, xi]).max()),
            float(np.abs(scores[:, yi]).max()),
        ) if len(scores) > 0 else 1.0
        loading_scale = max(
            float(np.abs(loadings_x).max()),
            float(np.abs(loadings_y).max()),
        ) if len(loadings_x) > 0 else 1.0
        arrow_scale = score_scale / loading_scale * 0.7  # 0.7 keeps arrows inside cloud

        feature_names = list(num_df.columns)
        for feat, lx, ly in zip(feature_names, loadings_x, loadings_y):
            tip_x = float(lx) * arrow_scale
            tip_y = float(ly) * arrow_scale
            # Arrow line
            fig.add_trace(
                go.Scatter(
                    x=[0, tip_x],
                    y=[0, tip_y],
                    mode="lines",
                    line=dict(color="rgba(220,50,50,0.7)", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Arrow tip label
            fig.add_trace(
                go.Scatter(
                    x=[tip_x],
                    y=[tip_y],
                    mode="text",
                    text=[feat],
                    textposition="top center",
                    textfont=dict(size=9, color="rgba(180,0,0,0.9)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Origin marker
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode="markers",
                marker=dict(symbol="cross", size=8, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return result_df, fig


# ---------------------------------------------------------------------------
# 10. Outlier Detection (Z-score & IQR)
# ---------------------------------------------------------------------------

def run_outlier_detection(
    df: pd.DataFrame,
    meta: list[dict],
    column: str = "",
    method: str = "zscore",
    threshold: int = 3,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Flag outliers in a numeric column using Z-score or IQR method."""
    from scipy import stats as scipy_stats

    num_cols = _numeric_cols(df)
    if not num_cols:
        return pd.DataFrame({"error": ["No numeric columns found."]}), None
    if not column or column not in df.columns:
        column = num_cols[0]

    series = df[column].dropna()
    if len(series) < 4:
        return pd.DataFrame({"error": ["Too few non-null values for outlier detection."]}), None

    if method == "zscore":
        z = np.abs(scipy_stats.zscore(series))
        mask = z > threshold
    else:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        fence_lo = q1 - threshold * iqr
        fence_hi = q3 + threshold * iqr
        mask = (series < fence_lo) | (series > fence_hi)

    outlier_vals = series[mask]
    result_df = pd.DataFrame([{
        "column": column,
        "method": method,
        "threshold": threshold,
        "total_values": len(series),
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(mask.mean() * 100, 2),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
    }])

    plot_df = df[[column]].copy().dropna()
    plot_df["_outlier"] = mask.reindex(plot_df.index, fill_value=False)
    fig = px.box(
        series.reset_index(drop=True),
        points="all",
        title=f"Outlier Detection — {column} ({method}, threshold={threshold})",
        labels={"value": column},
    )
    if not outlier_vals.empty:
        fig.add_scatter(
            x=[0] * len(outlier_vals),
            y=outlier_vals.values,
            mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name=f"Outliers ({len(outlier_vals)})",
        )
    return result_df, fig


# ---------------------------------------------------------------------------
# 11. Chi-Square Test of Independence
# ---------------------------------------------------------------------------

def run_chi_square(
    df: pd.DataFrame,
    meta: list[dict],
    column_a: str = "",
    column_b: str = "",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Chi-square test of independence between two categorical columns."""
    from scipy import stats as scipy_stats

    # Fall back to low-cardinality columns when no true categoricals
    cat_cols = _categorical_cols(df)
    if len(cat_cols) < 2:
        low_card = [c for c in df.columns if df[c].nunique() <= 20]
        cat_cols = low_card if len(low_card) >= 2 else list(df.columns)

    if len(cat_cols) < 2:
        return pd.DataFrame({"error": ["Need at least 2 categorical columns."]}), None

    if not column_a or column_a not in df.columns:
        column_a = cat_cols[0]
    if not column_b or column_b not in df.columns:
        column_b = cat_cols[1] if cat_cols[1] != column_a else (cat_cols[2] if len(cat_cols) > 2 else cat_cols[0])

    subset = df[[column_a, column_b]].dropna()
    if len(subset) < 5:
        return pd.DataFrame({"error": ["Too few rows after dropping nulls."]}), None

    ct = pd.crosstab(subset[column_a], subset[column_b])
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)

    n = len(subset)
    cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else 0.0

    result_df = pd.DataFrame([{
        "column_a": column_a,
        "column_b": column_b,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p, 6),
        "degrees_of_freedom": dof,
        "cramers_v": round(cramers_v, 4),
        "significant_at_0.05": p < 0.05,
        "n": n,
    }])

    fig = px.imshow(
        ct,
        text_auto=True,
        title=f"Chi-Square — {column_a} × {column_b} (χ²={chi2:.2f}, p={p:.4f})",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    return result_df, fig


# ---------------------------------------------------------------------------
# 12. Two-Group Comparison (T-Test / Mann-Whitney U)
# ---------------------------------------------------------------------------

def run_two_group_test(
    df: pd.DataFrame,
    meta: list[dict],
    numeric_col: str = "",
    group_col: str = "",
    test_type: str = "t-test",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Compare a numeric column across two groups using t-test or Mann-Whitney U."""
    from scipy import stats as scipy_stats

    num_cols = _numeric_cols(df)
    if not num_cols:
        return pd.DataFrame({"error": ["No numeric columns found."]}), None

    if not numeric_col or numeric_col not in df.columns:
        numeric_col = num_cols[0]

    # Find a usable group column (low cardinality, ≥2 groups)
    if not group_col or group_col not in df.columns:
        candidates = [c for c in df.columns if c != numeric_col and df[c].nunique() == 2]
        if not candidates:
            candidates = [c for c in df.columns if c != numeric_col and 2 <= df[c].nunique() <= 10]
        group_col = candidates[0] if candidates else (num_cols[1] if len(num_cols) > 1 else num_cols[0])

    top2 = df[group_col].value_counts().nlargest(2).index.tolist()
    if len(top2) < 2:
        return pd.DataFrame({"error": [f"Column '{group_col}' has fewer than 2 distinct values."]}), None

    g1 = df.loc[df[group_col] == top2[0], numeric_col].dropna()
    g2 = df.loc[df[group_col] == top2[1], numeric_col].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return pd.DataFrame({"error": ["At least one group has too few observations."]}), None

    if test_type == "mann-whitney":
        stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U"
    else:
        stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=False)
        test_name = "Welch t-test"

    # Cohen's d
    pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
    cohens_d = float((g1.mean() - g2.mean()) / pooled_std) if pooled_std > 0 else 0.0

    result_df = pd.DataFrame([
        {
            "group": str(top2[0]), "n": len(g1),
            "mean": round(float(g1.mean()), 4), "std": round(float(g1.std()), 4),
            "median": round(float(g1.median()), 4),
        },
        {
            "group": str(top2[1]), "n": len(g2),
            "mean": round(float(g2.mean()), 4), "std": round(float(g2.std()), 4),
            "median": round(float(g2.median()), 4),
        },
        {
            "group": f"TEST ({test_name})",
            "n": len(g1) + len(g2),
            "mean": round(stat, 4),
            "std": round(p, 6),
            "median": round(cohens_d, 4),
        },
    ])

    subset = df[df[group_col].isin(top2)][[group_col, numeric_col]].dropna()
    subset[group_col] = subset[group_col].astype(str)
    fig = px.box(
        subset,
        x=group_col,
        y=numeric_col,
        points="all",
        title=f"{test_name}: {numeric_col} by {group_col} (p={p:.4f}, d={cohens_d:.3f})",
    )
    return result_df, fig


# ---------------------------------------------------------------------------
# 13. Multi-Group Comparison (ANOVA / Kruskal-Wallis)
# ---------------------------------------------------------------------------

def run_multi_group_test(
    df: pd.DataFrame,
    meta: list[dict],
    numeric_col: str = "",
    group_col: str = "",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """ANOVA or Kruskal-Wallis test across multiple groups."""
    from scipy import stats as scipy_stats

    num_cols = _numeric_cols(df)
    if not num_cols:
        return pd.DataFrame({"error": ["No numeric columns found."]}), None

    if not numeric_col or numeric_col not in df.columns:
        numeric_col = num_cols[0]

    if not group_col or group_col not in df.columns:
        candidates = [c for c in df.columns if c != numeric_col and 2 <= df[c].nunique() <= 15]
        group_col = candidates[0] if candidates else (num_cols[1] if len(num_cols) > 1 else num_cols[0])

    subset = df[[group_col, numeric_col]].dropna()
    groups = [g[numeric_col].values for _, g in subset.groupby(group_col) if len(g) >= 2]
    if len(groups) < 2:
        return pd.DataFrame({"error": ["Need at least 2 groups with ≥ 2 observations each."]}), None

    # Decide test: if any group fails Shapiro-Wilk normality → Kruskal
    use_kruskal = False
    for g in groups:
        if len(g) >= 3 and len(g) <= 5000:
            _, sp = scipy_stats.shapiro(g[:5000])
            if sp < 0.05:
                use_kruskal = True
                break

    if use_kruskal:
        stat, p = scipy_stats.kruskal(*groups)
        test_name = "Kruskal-Wallis H"
    else:
        stat, p = scipy_stats.f_oneway(*groups)
        test_name = "One-way ANOVA F"

    per_group = subset.groupby(group_col)[numeric_col].agg(
        n="count", mean="mean", std="std", median="median"
    ).round(4).reset_index()
    per_group.columns = ["group", "n", "mean", "std", "median"]
    summary = pd.concat([
        per_group,
        pd.DataFrame([{"group": f"TEST ({test_name})", "n": len(subset),
                       "mean": round(stat, 4), "std": round(p, 6), "median": float("nan")}]),
    ], ignore_index=True)

    subset[group_col] = subset[group_col].astype(str)
    fig = px.box(
        subset,
        x=group_col,
        y=numeric_col,
        points="outliers",
        title=f"{test_name}: {numeric_col} by {group_col} (stat={stat:.3f}, p={p:.4f})",
    )
    fig.update_layout(xaxis_tickangle=-30)
    return summary, fig


# ---------------------------------------------------------------------------
# 14. Normality Test
# ---------------------------------------------------------------------------

def run_normality_test(
    df: pd.DataFrame,
    meta: list[dict],
    column: str = "",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Shapiro-Wilk (n≤5000) or KS test plus skewness/kurtosis."""
    from scipy import stats as scipy_stats

    num_cols = _numeric_cols(df)
    if not num_cols:
        return pd.DataFrame({"error": ["No numeric columns found."]}), None
    if not column or column not in df.columns:
        column = num_cols[0]

    series = df[column].dropna()
    if len(series) < 3:
        return pd.DataFrame({"error": ["Need at least 3 non-null values."]}), None

    n = len(series)
    if n <= 5000:
        stat, p = scipy_stats.shapiro(series)
        test_name = "Shapiro-Wilk"
    else:
        norm_series = (series - series.mean()) / series.std()
        stat, p = scipy_stats.kstest(norm_series, "norm")
        test_name = "Kolmogorov-Smirnov"

    skew = float(scipy_stats.skew(series))
    kurt = float(scipy_stats.kurtosis(series))

    result_df = pd.DataFrame([{
        "column": column,
        "test": test_name,
        "n": n,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "is_normal_p005": p > 0.05,
    }])

    # Histogram with normal curve overlay
    x_range = np.linspace(series.min(), series.max(), 200)
    mu, sigma = series.mean(), series.std()
    normal_y = scipy_stats.norm.pdf(x_range, mu, sigma)
    # Scale pdf to histogram counts
    bin_width = (series.max() - series.min()) / 30
    normal_y_scaled = normal_y * len(series) * bin_width

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series, nbinsx=30, name="Data", opacity=0.7))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_y_scaled,
        mode="lines", name="Normal curve",
        line=dict(color="red", width=2),
    ))
    fig.update_layout(
        title=f"Normality — {column} ({test_name}: p={p:.4f}, {'normal' if p > 0.05 else 'not normal'})",
        xaxis_title=column,
        yaxis_title="Count",
        barmode="overlay",
    )
    return result_df, fig


# ---------------------------------------------------------------------------
# 15. K-Means Clustering
# ---------------------------------------------------------------------------

def run_kmeans(
    df: pd.DataFrame,
    meta: list[dict],
    n_clusters: int = 3,
    scale: bool = True,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """K-Means clustering on numeric columns with silhouette score and 2D projection."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    num_df = df.select_dtypes(include="number").dropna()
    if num_df.shape[1] < 1:
        return pd.DataFrame({"error": ["No numeric columns found."]}), None
    if len(num_df) < n_clusters + 1:
        return pd.DataFrame({"error": [f"Need at least {n_clusters + 1} rows."]}), None

    X = num_df.values
    if scale:
        X = StandardScaler().fit_transform(X)

    n_clusters = min(n_clusters, len(num_df) - 1)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    sil = float(silhouette_score(X, labels)) if n_clusters > 1 else float("nan")

    counts = pd.Series(labels).value_counts().sort_index()
    summary = pd.DataFrame({
        "cluster": counts.index,
        "count": counts.values,
        "pct": (counts.values / len(labels) * 100).round(2),
        "silhouette_score": [round(sil, 4)] + [float("nan")] * (len(counts) - 1),
    })

    # Elbow plot k=2..min(10, n//10)
    max_k = min(10, len(num_df) // 10)
    ks = list(range(2, max(3, max_k + 1)))
    inertias = []
    for k in ks:
        _km = KMeans(n_clusters=k, random_state=42, n_init=10)
        _km.fit(X)
        inertias.append(_km.inertia_)
    elbow_fig = px.line(
        x=ks, y=inertias,
        markers=True,
        title="Elbow Plot — Inertia vs Number of Clusters",
        labels={"x": "k (clusters)", "y": "Inertia"},
    )

    # 2D scatter (PCA projection if >2 dims)
    if X.shape[1] >= 2:
        coords = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X[:, :2]
        scatter_df = pd.DataFrame(coords, columns=["Dim1", "Dim2"])
        scatter_df["Cluster"] = labels.astype(str)
        fig = px.scatter(
            scatter_df, x="Dim1", y="Dim2", color="Cluster",
            opacity=0.7,
            title=f"K-Means (k={n_clusters}, silhouette={sil:.3f})",
        )
        fig.update_traces(marker_size=5)
    else:
        fig = elbow_fig

    return summary, fig


# ---------------------------------------------------------------------------
# 16. Feature Importance (Random Forest)
# ---------------------------------------------------------------------------

def run_feature_importance(
    df: pd.DataFrame,
    meta: list[dict],
    target_col: str = "",
    max_features: int = 20,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Random Forest feature importance for any target column."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]

    feature_cols = [c for c in _numeric_cols(df) if c != target_col]
    if not feature_cols:
        return pd.DataFrame({"error": ["No numeric feature columns found."]}), None

    subset = df[feature_cols + [target_col]].dropna()
    if len(subset) < 10:
        return pd.DataFrame({"error": ["Too few rows after dropping nulls."]}), None

    # Sample for speed
    if len(subset) > 10000:
        subset = subset.sample(10000, random_state=42)

    X = subset[feature_cols].values
    y_raw = subset[target_col]
    is_categorical = y_raw.dtype == object or y_raw.nunique() <= 20

    if is_categorical:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_type = "classifier"
    else:
        y = y_raw.values
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_type = "regressor"

    model.fit(X, y)
    importances = model.feature_importances_

    result_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(max_features).reset_index(drop=True)
    result_df["rank"] = range(1, len(result_df) + 1)
    result_df["importance"] = result_df["importance"].round(6)

    fig = px.bar(
        result_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature Importance — target: {target_col} ({model_type})",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return result_df, fig


# ---------------------------------------------------------------------------
# 17. Time Series Trend
# ---------------------------------------------------------------------------

def run_time_series(
    df: pd.DataFrame,
    meta: list[dict],
    date_col: str = "",
    value_col: str = "",
    window: int = 7,
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Line chart with rolling mean overlay for a date + numeric column pair."""
    # Find date column
    if not date_col or date_col not in df.columns:
        # Try to auto-detect a date-like column
        for col in df.columns:
            if any(kw in col.lower() for kw in ("date", "time", "year", "month", "week", "day")):
                date_col = col
                break
    if not date_col or date_col not in df.columns:
        return pd.DataFrame({"error": [
            "No date column found. Select a column containing dates/times."
        ]}), None

    num_cols = _numeric_cols(df)
    if not value_col or value_col not in df.columns:
        value_col = num_cols[0] if num_cols else None
    if not value_col:
        return pd.DataFrame({"error": ["No numeric value column found."]}), None

    ts = df[[date_col, value_col]].dropna().copy()
    if len(ts) < 2:
        return pd.DataFrame({"error": ["Too few rows for time series."]}), None

    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col)

    if len(ts) < 2:
        return pd.DataFrame({"error": [f"Could not parse '{date_col}' as dates."]}), None

    window = min(window, max(2, len(ts) // 10))
    ts["_rolling_mean"] = ts[value_col].rolling(window=window, min_periods=1).mean()

    result_df = pd.DataFrame([{
        "date_col": date_col,
        "value_col": value_col,
        "n": len(ts),
        "date_min": str(ts[date_col].min().date()),
        "date_max": str(ts[date_col].max().date()),
        "value_mean": round(float(ts[value_col].mean()), 4),
        "value_std": round(float(ts[value_col].std()), 4),
        "rolling_window": window,
    }])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts[date_col], y=ts[value_col],
        mode="lines",
        name=value_col,
        line=dict(width=1, color="steelblue"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=ts[date_col], y=ts["_rolling_mean"],
        mode="lines",
        name=f"Rolling mean ({window})",
        line=dict(width=2, color="red"),
    ))
    fig.update_layout(
        title=f"Time Series: {value_col} over {date_col} (window={window})",
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    return result_df, fig


# ---------------------------------------------------------------------------
# 18. Geographic Summary (Choropleth)
# ---------------------------------------------------------------------------

def run_geo_summary(
    df: pd.DataFrame,
    meta: list[dict],
    location_col: str = "",
    value_col: str = "",
    agg: str = "mean",
    **params: Any,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Aggregate by a location column and render a choropleth world map."""
    all_cols = list(df.columns)

    # Auto-select location column: prefer columns with 'country'/'geo'/'location' in name
    if not location_col or location_col not in df.columns:
        for col in all_cols:
            if any(kw in col.lower() for kw in ("country", "geo", "location", "nation", "region")):
                location_col = col
                break
        if not location_col or location_col not in df.columns:
            location_col = all_cols[0] if all_cols else None

    if not location_col:
        return pd.DataFrame({"error": ["No location column found."]}), None

    num_cols = _numeric_cols(df)
    if not value_col or value_col not in df.columns:
        value_col = num_cols[0] if num_cols else None
    if not value_col:
        return pd.DataFrame({"error": ["No numeric value column found."]}), None

    if location_col == value_col:
        return pd.DataFrame({"error": ["Location column and value column must be different."]}), None

    subset = df[[location_col, value_col]].dropna()
    if subset.empty:
        return pd.DataFrame({"error": ["No data after dropping nulls."]}), None

    fn_map = {"mean": "mean", "sum": "sum", "count": "count",
              "min": "min", "max": "max", "median": "median"}
    fn = fn_map.get(agg, "mean")
    if fn == "count":
        agg_df = subset.groupby(location_col).size().reset_index(name=value_col)
    else:
        agg_df = subset.groupby(location_col)[value_col].agg(fn).reset_index()

    agg_df = agg_df.sort_values(value_col, ascending=False).reset_index(drop=True)
    agg_df[value_col] = agg_df[value_col].round(4)

    # Auto-detect location mode
    sample = agg_df[location_col].dropna().astype(str).head(50)
    avg_len = sample.str.len().mean() if not sample.empty else 10
    pct_upper = sample.str.isupper().mean() if not sample.empty else 0
    location_mode = "ISO-3" if (avg_len <= 3.5 and pct_upper >= 0.7) else "country names"

    fig = px.choropleth(
        agg_df,
        locations=location_col,
        color=value_col,
        locationmode=location_mode,
        color_continuous_scale="Viridis",
        title=f"Geographic Summary: {agg}({value_col}) by {location_col}",
        projection="natural earth",
    )
    fig.update_layout(
        coloraxis_colorbar={"title": f"{agg}({value_col})"},
        margin={"r": 0, "l": 0, "t": 40, "b": 0},
        height=500,
    )
    return agg_df, fig
