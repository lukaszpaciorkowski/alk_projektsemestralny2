"""
tests/test_integration.py — End-to-end integration tests for the app/core pipeline.

Tests:
  1. Pipeline end-to-end: import CSV → _datasets → columns populated
  2. list_datasets() returns complete metadata including decoded columns
  3. Generic analytics functions with real DB data
  4. Diabetes-specific analytics functions with real DB data
  5. Query layer: fetch_table, fetch_column_stats, fetch_distinct_values
  6. Chart builder: build_chart() with real data
  7. Page file AST validation (no syntax errors)
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

from app.core.pipeline import (
    DB_PATH,
    get_engine,
    import_csv,
    list_datasets,
    get_dataset_meta,
)
from app.core.query import (
    Filter,
    fetch_column_stats,
    fetch_distinct_values,
    fetch_table,
    row_count,
)
from app.core.registry import REGISTRY, get_functions_for
from app.components.chart_builder import build_chart

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEART_CSV = Path("data/raw/heart_disease_cleveland.csv")
PIMA_CSV = Path("data/raw/pima_indians_diabetes.csv")
DIABETES_CSV = Path("data/raw/diabetic_data.csv")

BOOTSTRAP_SQL = Path("database/bootstrap.sql")


def _make_fresh_engine():
    """In-memory SQLite engine with bootstrap schema."""
    engine = create_engine("sqlite:///:memory:")
    sql = BOOTSTRAP_SQL.read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))
    return engine


@pytest.fixture(scope="module")
def fresh_engine():
    return _make_fresh_engine()


@pytest.fixture(scope="module")
def prod_engine():
    """Engine pointing at the real data.db."""
    return get_engine(DB_PATH)


@pytest.fixture(scope="module")
def prod_datasets(prod_engine):
    return list_datasets(prod_engine)


@pytest.fixture(scope="module")
def heart_table(prod_datasets):
    ds = next((d for d in prod_datasets if "heart" in d["table_name"]), None)
    assert ds is not None, "Heart disease dataset not found in data.db"
    return ds


@pytest.fixture(scope="module")
def pima_table(prod_datasets):
    ds = next((d for d in prod_datasets if "pima" in d["table_name"]), None)
    assert ds is not None, "Pima dataset not found in data.db"
    return ds


@pytest.fixture(scope="module")
def diabetes_table(prod_datasets):
    ds = next((d for d in prod_datasets if "diabetic_data" in d["table_name"]), None)
    assert ds is not None, "Diabetes dataset not found in data.db"
    return ds


@pytest.fixture(scope="module")
def heart_df(heart_table, prod_engine):
    with prod_engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM [{heart_table['table_name']}]"), conn)


@pytest.fixture(scope="module")
def pima_df(pima_table, prod_engine):
    with prod_engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM [{pima_table['table_name']}]"), conn)


@pytest.fixture(scope="module")
def diabetes_df(diabetes_table, prod_engine):
    with prod_engine.connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM [{diabetes_table['table_name']}] LIMIT 5000"), conn
        )


# ---------------------------------------------------------------------------
# 1. Pipeline end-to-end
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_import_csv_creates_table(self, fresh_engine):
        result = import_csv(HEART_CSV, fresh_engine)
        assert result.row_count == 303
        assert result.col_count == 14
        with fresh_engine.connect() as conn:
            rows = conn.execute(text(f"SELECT COUNT(*) FROM [{result.table_name}]")).scalar()
        assert rows == 303

    def test_import_csv_registers_in_datasets(self, fresh_engine):
        datasets = list_datasets(fresh_engine)
        assert len(datasets) >= 1
        names = [d["table_name"] for d in datasets]
        assert any("heart" in n for n in names)

    def test_import_csv_columns_metadata_populated(self, fresh_engine):
        datasets = list_datasets(fresh_engine)
        ds = next(d for d in datasets if "heart" in d["table_name"])
        cols = ds["columns"]
        assert isinstance(cols, list), f"Expected list, got {type(cols)}"
        assert len(cols) == 14
        col_names = [c["name"] for c in cols]
        assert "age" in col_names
        assert "target" in col_names

    def test_import_csv_columns_have_dtype(self, fresh_engine):
        datasets = list_datasets(fresh_engine)
        ds = next(d for d in datasets if "heart" in d["table_name"])
        for col in ds["columns"]:
            assert "name" in col
            assert "dtype" in col

    def test_import_pima_generic_type(self, fresh_engine):
        result = import_csv(PIMA_CSV, fresh_engine)
        assert result.dataset_type == "generic"
        assert result.row_count == 768

    def test_import_detects_diabetes_type(self):
        engine = _make_fresh_engine()
        result = import_csv(DIABETES_CSV, engine)
        assert result.dataset_type == "diabetes"
        assert result.row_count > 50_000

    def test_duplicate_import_raises(self, fresh_engine):
        with pytest.raises(ValueError, match="already imported"):
            import_csv(HEART_CSV, fresh_engine)


# ---------------------------------------------------------------------------
# 2. list_datasets() returns complete metadata
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_returns_list(self, prod_engine):
        datasets = list_datasets(prod_engine)
        assert isinstance(datasets, list)
        assert len(datasets) >= 3

    def test_columns_is_list_not_none(self, prod_datasets):
        for ds in prod_datasets:
            assert ds["columns"] is not None, f"{ds['table_name']} has None columns"
            assert isinstance(ds["columns"], list), \
                f"{ds['table_name']} columns is {type(ds['columns'])}"

    def test_columns_contain_name_and_dtype(self, prod_datasets):
        for ds in prod_datasets:
            for col in ds["columns"]:
                assert "name" in col, f"Missing 'name' in {col}"
                assert "dtype" in col, f"Missing 'dtype' in {col}"

    def test_heart_has_correct_columns(self, heart_table):
        cols = [c["name"] for c in heart_table["columns"]]
        assert "age" in cols
        assert "target" in cols
        assert len(cols) == 14

    def test_pima_has_correct_columns(self, pima_table):
        cols = [c["name"] for c in pima_table["columns"]]
        assert "Glucose" in cols
        assert "Outcome" in cols
        assert len(cols) == 9

    def test_row_col_counts_positive(self, prod_datasets):
        for ds in prod_datasets:
            assert ds["row_count"] > 0
            assert ds["col_count"] > 0

    def test_enrichment_status_field_present(self, prod_datasets):
        for ds in prod_datasets:
            assert "enrichment_status" in ds
            assert ds["enrichment_status"] in ("none", "pending", "done")

    def test_get_dataset_meta_matches_list(self, prod_engine, heart_table):
        meta = get_dataset_meta(heart_table["table_name"], prod_engine)
        assert meta is not None
        assert meta["table_name"] == heart_table["table_name"]
        assert isinstance(meta["columns"], list)


# ---------------------------------------------------------------------------
# 3. Generic analytics functions
# ---------------------------------------------------------------------------

class TestGenericAnalytics:
    """Each function must return (DataFrame, Figure|None) without raising."""

    def _meta(self, df: pd.DataFrame) -> list[dict]:
        return [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    def test_describe(self, heart_df):
        fn = REGISTRY["generic.describe"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df))
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 14

    def test_correlation(self, heart_df):
        fn = REGISTRY["generic.correlation"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df))
        assert isinstance(result_df, pd.DataFrame)
        assert fig is not None

    def test_value_counts(self, heart_df):
        fn = REGISTRY["generic.value_counts"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), column="sex")
        assert isinstance(result_df, pd.DataFrame)
        assert "count" in result_df.columns

    def test_groupby(self, heart_df):
        fn = REGISTRY["generic.groupby"]
        result_df, fig = fn.fn(
            heart_df, self._meta(heart_df), group_col="sex", agg_col="age", agg_func="mean"
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0

    def test_crosstab(self, heart_df):
        fn = REGISTRY["generic.crosstab"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df))
        assert isinstance(result_df, pd.DataFrame)

    def test_distribution(self, heart_df):
        fn = REGISTRY["generic.distribution"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), column="age")
        assert isinstance(result_df, pd.DataFrame)
        assert fig is not None

    def test_null_analysis(self, heart_df):
        fn = REGISTRY["generic.null_analysis"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df))
        assert isinstance(result_df, pd.DataFrame)
        assert "column" in result_df.columns or len(result_df.columns) > 0

    def test_dtypes(self, heart_df):
        fn = REGISTRY["generic.dtypes"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df))
        assert isinstance(result_df, pd.DataFrame)

    def test_pca_returns_variance_table_and_scatter(self, heart_df):
        fn = REGISTRY["generic.pca"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), n_components=2, scale=True)
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == [
            "component", "explained_variance", "explained_variance_ratio", "cumulative_ratio"
        ]
        assert len(result_df) == 2
        assert result_df["cumulative_ratio"].iloc[-1] <= 1.0
        assert fig is not None

    def test_pca_cumulative_ratio_monotone(self, heart_df):
        fn = REGISTRY["generic.pca"]
        result_df, _ = fn.fn(heart_df, self._meta(heart_df), n_components=3, scale=True)
        ratios = result_df["cumulative_ratio"].tolist()
        assert ratios == sorted(ratios), "cumulative_ratio must be non-decreasing"

    def test_pca_no_scale(self, heart_df):
        fn = REGISTRY["generic.pca"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), n_components=2, scale=False)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2

    def test_outlier_detection_zscore(self, heart_df):
        fn = REGISTRY["generic.outlier_detection"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), column="chol", method="zscore", threshold=3)
        assert isinstance(result_df, pd.DataFrame)
        assert "outlier_count" in result_df.columns
        assert fig is not None

    def test_outlier_detection_iqr(self, heart_df):
        fn = REGISTRY["generic.outlier_detection"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), column="age", method="iqr", threshold=1)
        assert int(result_df["outlier_count"].iloc[0]) >= 0

    def test_chi_square_returns_statistic(self, heart_df):
        fn = REGISTRY["generic.chi_square"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), column_a="sex", column_b="target")
        assert isinstance(result_df, pd.DataFrame)
        assert "chi2_statistic" in result_df.columns
        assert "p_value" in result_df.columns
        assert fig is not None

    def test_chi_square_cramers_v_range(self, heart_df):
        fn = REGISTRY["generic.chi_square"]
        result_df, _ = fn.fn(heart_df, self._meta(heart_df), column_a="sex", column_b="cp")
        assert 0.0 <= float(result_df["cramers_v"].iloc[0]) <= 1.0

    def test_two_group_test_ttest(self, heart_df):
        fn = REGISTRY["generic.two_group_test"]
        result_df, fig = fn.fn(
            heart_df, self._meta(heart_df),
            numeric_col="chol", group_col="sex", test_type="t-test"
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3  # group1, group2, TEST row
        assert fig is not None

    def test_two_group_test_mann_whitney(self, heart_df):
        fn = REGISTRY["generic.two_group_test"]
        result_df, _ = fn.fn(
            heart_df, self._meta(heart_df),
            numeric_col="age", group_col="target", test_type="mann-whitney"
        )
        assert "TEST" in result_df["group"].iloc[-1]

    def test_multi_group_test_returns_per_group_stats(self, heart_df):
        fn = REGISTRY["generic.multi_group_test"]
        result_df, fig = fn.fn(
            heart_df, self._meta(heart_df), numeric_col="chol", group_col="cp"
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= 3  # at least 2 groups + TEST row
        assert fig is not None

    def test_normality_test_shapiro(self, heart_df):
        fn = REGISTRY["generic.normality_test"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), column="age")
        assert isinstance(result_df, pd.DataFrame)
        assert "p_value" in result_df.columns
        assert "is_normal_p005" in result_df.columns
        assert fig is not None

    def test_normality_test_skewness_is_float(self, heart_df):
        fn = REGISTRY["generic.normality_test"]
        result_df, _ = fn.fn(heart_df, self._meta(heart_df), column="chol")
        assert isinstance(float(result_df["skewness"].iloc[0]), float)

    def test_kmeans_summary_shape(self, heart_df):
        fn = REGISTRY["generic.kmeans"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), n_clusters=3, scale=True)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert result_df["count"].sum() == len(heart_df.dropna())
        assert fig is not None

    def test_kmeans_pct_sums_to_100(self, heart_df):
        fn = REGISTRY["generic.kmeans"]
        result_df, _ = fn.fn(heart_df, self._meta(heart_df), n_clusters=4, scale=False)
        assert abs(result_df["pct"].sum() - 100.0) < 0.1

    def test_feature_importance_sorted_desc(self, heart_df):
        fn = REGISTRY["generic.feature_importance"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df), target_col="target", max_features=5)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) <= 5
        importances = result_df["importance"].tolist()
        assert importances == sorted(importances, reverse=True)
        assert fig is not None

    def test_feature_importance_rank_column(self, heart_df):
        fn = REGISTRY["generic.feature_importance"]
        result_df, _ = fn.fn(heart_df, self._meta(heart_df), target_col="target", max_features=10)
        assert list(result_df["rank"]) == list(range(1, len(result_df) + 1))

    def test_time_series_no_date_col_returns_error_df(self, heart_df):
        """heart_df has no date column — should return error DataFrame, not raise."""
        fn = REGISTRY["generic.time_series"]
        result_df, fig = fn.fn(heart_df, self._meta(heart_df))
        assert isinstance(result_df, pd.DataFrame)
        assert "error" in result_df.columns

    def test_time_series_with_date_col(self):
        """Synthetic time series data."""
        import pandas as pd
        fn = REGISTRY["generic.time_series"]
        ts_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=30, freq="D").astype(str),
            "value": range(30),
        })
        meta = [{"name": c, "dtype": str(ts_df[c].dtype)} for c in ts_df.columns]
        result_df, fig = fn.fn(ts_df, meta, date_col="date", value_col="value", window=5)
        assert isinstance(result_df, pd.DataFrame)
        assert "error" not in result_df.columns
        assert fig is not None

    def test_all_generic_functions_callable(self, heart_df):
        """Smoke test: every generic fn can be called with defaults."""
        fns = get_functions_for("generic")
        meta = self._meta(heart_df)
        for fn in fns:
            result_df, _ = fn.fn(heart_df, meta)
            assert isinstance(result_df, pd.DataFrame), f"{fn.id} did not return DataFrame"


# ---------------------------------------------------------------------------
# 4. Diabetes-specific analytics functions
# ---------------------------------------------------------------------------

class TestDiabetesAnalytics:
    def _meta(self, df: pd.DataFrame) -> list[dict]:
        return [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    def test_readmission_by_group(self, diabetes_df):
        fn = REGISTRY["diabetes.readmission_by_group"]
        result_df, fig = fn.fn(diabetes_df, self._meta(diabetes_df))
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert fig is not None

    def test_hba1c_vs_readmission(self, diabetes_df):
        fn = REGISTRY["diabetes.hba1c_vs_readmission"]
        result_df, fig = fn.fn(diabetes_df, self._meta(diabetes_df))
        assert isinstance(result_df, pd.DataFrame)

    def test_los_by_readmission(self, diabetes_df):
        fn = REGISTRY["diabetes.los_by_readmission"]
        result_df, fig = fn.fn(diabetes_df, self._meta(diabetes_df))
        assert isinstance(result_df, pd.DataFrame)

    def test_medications_vs_los(self, diabetes_df):
        fn = REGISTRY["diabetes.medications_vs_los"]
        result_df, fig = fn.fn(diabetes_df, self._meta(diabetes_df))
        assert isinstance(result_df, pd.DataFrame)

    def test_top_diagnoses_with_enrichment(self, diabetes_table, diabetes_df, prod_engine):
        fn = REGISTRY["diabetes.top_diagnoses"]
        assert fn.requires_enrichment
        result_df, fig = fn.fn(
            diabetes_df,
            self._meta(diabetes_df),
            con=prod_engine,
            table_name=diabetes_table["table_name"],
            enrichment_status="done",
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0

    def test_medication_frequency_with_enrichment(self, diabetes_table, diabetes_df, prod_engine):
        fn = REGISTRY["diabetes.medication_frequency"]
        assert fn.requires_enrichment
        result_df, fig = fn.fn(
            diabetes_df,
            self._meta(diabetes_df),
            con=prod_engine,
            table_name=diabetes_table["table_name"],
            enrichment_status="done",
        )
        assert isinstance(result_df, pd.DataFrame)

    def test_enrichment_required_raises_without_con(self, diabetes_df):
        from app.core.pipeline import EnrichmentRequiredError
        fn = REGISTRY["diabetes.top_diagnoses"]
        with pytest.raises((EnrichmentRequiredError, Exception)):
            fn.fn(diabetes_df, [], enrichment_status="none")


# ---------------------------------------------------------------------------
# 5. Query layer
# ---------------------------------------------------------------------------

class TestQueryLayer:
    def test_fetch_table_returns_dataframe(self, heart_table, prod_engine):
        df = fetch_table(heart_table["table_name"], prod_engine, limit=500)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 303

    def test_fetch_table_limit(self, heart_table, prod_engine):
        df = fetch_table(heart_table["table_name"], prod_engine, limit=10)
        assert len(df) == 10

    def test_fetch_table_offset(self, heart_table, prod_engine):
        df_all = fetch_table(heart_table["table_name"], prod_engine)
        df_offset = fetch_table(heart_table["table_name"], prod_engine, offset=100, limit=10)
        assert len(df_offset) == 10
        # offset rows differ from start
        assert not df_offset.equals(df_all.iloc[:10])

    def test_fetch_table_order_by(self, heart_table, prod_engine):
        df = fetch_table(heart_table["table_name"], prod_engine, order_by="age", ascending=True)
        ages = df["age"].dropna().tolist()
        assert ages == sorted(ages)

    def test_fetch_table_order_by_desc(self, heart_table, prod_engine):
        df = fetch_table(heart_table["table_name"], prod_engine, order_by="age", ascending=False)
        ages = df["age"].dropna().tolist()
        assert ages == sorted(ages, reverse=True)

    def test_fetch_table_filter_eq(self, heart_table, prod_engine):
        df = fetch_table(
            heart_table["table_name"],
            prod_engine,
            filters=[Filter("sex", "eq", 1.0)],
        )
        assert len(df) > 0
        assert all(df["sex"] == 1.0)

    def test_fetch_table_filter_gt(self, heart_table, prod_engine):
        df = fetch_table(
            heart_table["table_name"],
            prod_engine,
            filters=[Filter("age", "gt", 60.0)],
        )
        assert len(df) > 0
        assert all(df["age"] > 60.0)

    def test_row_count_no_filter(self, heart_table, prod_engine):
        count = row_count(heart_table["table_name"], prod_engine)
        assert count == 303

    def test_row_count_with_filter(self, heart_table, prod_engine):
        count = row_count(
            heart_table["table_name"],
            prod_engine,
            filters=[Filter("sex", "eq", 1.0)],
        )
        assert 0 < count < 303

    def test_fetch_distinct_values(self, heart_table, prod_engine):
        vals = fetch_distinct_values(heart_table["table_name"], "sex", prod_engine)
        assert isinstance(vals, list)
        assert len(vals) == 2  # 0.0 and 1.0

    def test_fetch_distinct_values_limit(self, heart_table, prod_engine):
        vals = fetch_distinct_values(heart_table["table_name"], "age", prod_engine, limit=5)
        assert len(vals) <= 5

    def test_fetch_column_stats_numeric(self, heart_table, prod_engine):
        stats = fetch_column_stats(heart_table["table_name"], "age", "float64", prod_engine)
        assert stats["total"] == 303
        assert stats["non_null_count"] > 0
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert stats["min"] >= 20
        assert stats["max"] <= 80

    def test_fetch_column_stats_null_pct(self, heart_table, prod_engine):
        stats = fetch_column_stats(heart_table["table_name"], "age", "float64", prod_engine)
        assert 0.0 <= stats["null_pct"] <= 100.0

    def test_fetch_column_stats_categorical(self, heart_table, prod_engine):
        # sex is numeric but let's use a text-ish dtype check
        stats = fetch_column_stats(heart_table["table_name"], "sex", "object", prod_engine)
        assert stats["total"] == 303
        assert "sample_values" in stats


# ---------------------------------------------------------------------------
# 6. Chart builder
# ---------------------------------------------------------------------------

class TestChartBuilder:
    def test_bar_chart(self, heart_table, prod_engine):
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Bar",
            x_col="sex",
            y_col="age",
            agg_func="mean",
        )
        assert isinstance(fig, go.Figure)

    def test_histogram(self, heart_table, prod_engine):
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Histogram",
            x_col="age",
            bins=20,
        )
        assert isinstance(fig, go.Figure)

    def test_scatter(self, heart_table, prod_engine):
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Scatter",
            x_col="age",
            y_col="chol",
        )
        assert isinstance(fig, go.Figure)

    def test_box(self, heart_table, prod_engine):
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Box",
            x_col="sex",
            y_col="age",
        )
        assert isinstance(fig, go.Figure)

    def test_line_chart(self, heart_table, prod_engine):
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Line",
            x_col="age",
            y_col="chol",
        )
        assert isinstance(fig, go.Figure)

    def test_invalid_chart_type_returns_figure(self, heart_table, prod_engine):
        """build_chart never raises — returns annotated figure on error."""
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="NonExistentType",
            x_col="age",
        )
        assert isinstance(fig, go.Figure)

    def test_chart_with_color(self, heart_table, prod_engine):
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Scatter",
            x_col="age",
            y_col="chol",
            color_col="sex",
        )
        assert isinstance(fig, go.Figure)

    def test_choropleth_with_country_names(self, prod_engine, prod_datasets):
        """Choropleth using OWID COVID data which has a 'location' country column."""
        import plotly.graph_objects as go
        owid = next((d for d in prod_datasets if "owid_covid" in d["table_name"]), None)
        if owid is None:
            pytest.skip("OWID COVID dataset not loaded")
        fig = build_chart(
            table_name=owid["table_name"],
            engine=prod_engine,
            chart_type="Choropleth Map",
            x_col="location",
            y_col="total_cases",
            agg_func="max",
            location_mode="country names",
        )
        assert isinstance(fig, go.Figure)

    def test_choropleth_bad_columns_returns_error_figure(self, heart_table, prod_engine):
        """build_chart never raises even for bad choropleth config."""
        import plotly.graph_objects as go
        fig = build_chart(
            table_name=heart_table["table_name"],
            engine=prod_engine,
            chart_type="Choropleth Map",
            x_col="nonexistent_location",
            y_col="age",
            location_mode="country names",
        )
        assert isinstance(fig, go.Figure)

    def test_geo_summary_with_country_data(self, prod_engine, prod_datasets):
        """run_geo_summary with OWID COVID produces aggregated table + figure."""
        owid = next((d for d in prod_datasets if "owid_covid" in d["table_name"]), None)
        if owid is None:
            pytest.skip("OWID COVID dataset not loaded")
        import pandas as pd
        from sqlalchemy import text as sql_text
        with prod_engine.connect() as conn:
            df = pd.read_sql(sql_text(f"SELECT * FROM [{owid['table_name']}] LIMIT 5000"), conn)
        fn = REGISTRY["generic.geo_summary"]
        meta = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
        result_df, fig = fn.fn(df, meta, location_col="location", value_col="total_cases", agg="max")
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= 1  # at least one country aggregated
        assert "location" in result_df.columns
        assert fig is not None

    def test_geo_summary_no_geo_col_returns_error_df(self, heart_df):
        """heart_df has no country column — geo_summary returns error DataFrame."""
        import pandas as pd
        fn = REGISTRY["generic.geo_summary"]
        meta = [{"name": c, "dtype": str(heart_df[c].dtype)} for c in heart_df.columns]
        result_df, fig = fn.fn(heart_df, meta)
        assert isinstance(result_df, pd.DataFrame)
        # Should either return error or gracefully use first column
        assert "error" in result_df.columns or len(result_df) > 0


# ---------------------------------------------------------------------------
# 7. Page file AST validation
# ---------------------------------------------------------------------------

class TestPageFileAST:
    VIEW_DIR = Path("app/views")
    COMPONENT_DIR = Path("app/components")

    def _parse_ok(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"SyntaxError in {path}: {exc}")

    def test_data_sources_page(self):
        self._parse_ok(self.VIEW_DIR / "1_data_sources.py")

    def test_exploration_page(self):
        self._parse_ok(self.VIEW_DIR / "2_exploration.py")

    def test_adhoc_charts_page(self):
        self._parse_ok(self.VIEW_DIR / "3_adhoc_charts.py")

    def test_analytics_page(self):
        self._parse_ok(self.VIEW_DIR / "4_analytics.py")

    def test_reports_page(self):
        self._parse_ok(self.VIEW_DIR / "5_reports.py")

    def test_architecture_page(self):
        self._parse_ok(self.VIEW_DIR / "6_architecture.py")

    def test_sidebar_component(self):
        self._parse_ok(self.COMPONENT_DIR / "sidebar.py")

    def test_chart_builder_component(self):
        self._parse_ok(self.COMPONENT_DIR / "chart_builder.py")

    def test_main_entry_point(self):
        self._parse_ok(Path("app/main.py"))


# ---------------------------------------------------------------------------
# 8. Live HTTP smoke test
# ---------------------------------------------------------------------------

class TestLiveApp:
    BASE = "http://localhost:8501"

    def _get(self, path: str) -> int:
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.BASE}{path}", timeout=5) as r:
                return r.status
        except Exception:
            return 0

    def test_healthz(self):
        status = self._get("/healthz")
        assert status == 200, "Streamlit healthz endpoint not reachable"

    def test_root_returns_200(self):
        status = self._get("/")
        assert status == 200

    def test_data_sources_url(self):
        status = self._get("/data_sources")
        # Streamlit redirects non-default pages — accept 200 or 3xx
        assert status in (200, 301, 302, 303, 307, 308)

    def test_adhoc_charts_url(self):
        # We set url_path="adhoc_charts" explicitly
        status = self._get("/adhoc_charts")
        assert status in (200, 301, 302, 303, 307, 308)
