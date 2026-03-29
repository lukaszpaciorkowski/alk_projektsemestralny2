"""
tests/test_registry.py — Tests for app/core/registry.py

Verifies:
- REGISTRY structure and completeness
- get_functions_for() ordering and filtering
- Each function follows the (df, meta, **params) -> (DataFrame, Figure|None) contract
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.registry import REGISTRY, AnalyticsFunction, ParamSpec, get_functions_for


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generic_df():
    """Minimal DataFrame with both numeric and categorical columns."""
    return pd.DataFrame({
        "id": range(20),
        "value": [float(i * 1.5) for i in range(20)],
        "score": range(20, 40),
        "category": ["A", "B", "C", "D"] * 5,
        "label": ["X", "Y"] * 10,
    })


@pytest.fixture
def generic_meta(generic_df):
    """Column metadata list matching generic_df."""
    meta = []
    for col in generic_df.columns:
        dtype = str(generic_df[col].dtype)
        meta.append({
            "name": col,
            "dtype": dtype,
            "sql_type": "INTEGER" if "int" in dtype else ("REAL" if "float" in dtype else "TEXT"),
            "nullable": False,
            "unique_count": generic_df[col].nunique(),
        })
    return meta


DIABETES_REQUIRED = [
    "patient_nbr", "readmitted", "metformin",
    "time_in_hospital", "num_medications", "diag_1",
    "age", "race", "A1Cresult", "encounter_id",
    "diag_2", "diag_3",
]


@pytest.fixture
def diabetes_df():
    n = 60  # divisible by 2, 3, 4 for clean list multiplications
    df = pd.DataFrame({
        "patient_nbr": range(n),
        "encounter_id": range(1000, 1000 + n),
        "readmitted": (["NO", "<30", ">30"] * (n // 3 + 1))[:n],
        "metformin": ["Steady"] * n,
        "time_in_hospital": [i % 14 + 1 for i in range(n)],
        "num_medications": [i % 20 + 1 for i in range(n)],
        "diag_1": (["250.00", "401.9", "276.6"] * (n // 3 + 1))[:n],
        "diag_2": ["401.9"] * n,
        "diag_3": ["276.6"] * n,
        "age": (["[70-80)", "[60-70)", "[50-60)"] * (n // 3 + 1))[:n],
        "race": (["Caucasian", "AfricanAmerican", "Other"] * (n // 3 + 1))[:n],
        "A1Cresult": (["None", ">7", ">8", "Norm"] * (n // 4 + 1))[:n],
        "insulin": (["No", "Steady", "Up", "Down"] * (n // 4 + 1))[:n],
        "glyburide": (["No", "Steady"] * (n // 2 + 1))[:n],
    })
    return df


@pytest.fixture
def diabetes_meta(diabetes_df):
    meta = []
    for col in diabetes_df.columns:
        dtype = str(diabetes_df[col].dtype)
        meta.append({
            "name": col,
            "dtype": dtype,
            "sql_type": "INTEGER" if "int" in dtype else ("REAL" if "float" in dtype else "TEXT"),
            "nullable": False,
            "unique_count": diabetes_df[col].nunique(),
        })
    return meta


# ---------------------------------------------------------------------------
# REGISTRY structure
# ---------------------------------------------------------------------------

class TestRegistryStructure:
    def test_registry_is_not_empty(self):
        assert len(REGISTRY) > 0

    def test_all_entries_are_analytics_function(self):
        for key, fn in REGISTRY.items():
            assert isinstance(fn, AnalyticsFunction), f"{key} is not an AnalyticsFunction"

    def test_generic_functions_exist(self):
        generic = [fn for fn in REGISTRY.values() if fn.scope == "generic"]
        assert len(generic) >= 8

    def test_diabetes_functions_exist(self):
        diabetes = [fn for fn in REGISTRY.values() if fn.scope == "diabetes"]
        assert len(diabetes) >= 6

    def test_all_functions_have_callable_fn(self):
        for fn in REGISTRY.values():
            assert callable(fn.fn)

    def test_all_params_are_paramspec(self):
        for fn in REGISTRY.values():
            for p in fn.params:
                assert isinstance(p, ParamSpec)

    def test_ids_match_keys(self):
        for key, fn in REGISTRY.items():
            assert fn.id == key, f"ID mismatch: key={key!r}, fn.id={fn.id!r}"


# ---------------------------------------------------------------------------
# get_functions_for
# ---------------------------------------------------------------------------

class TestGetFunctionsFor:
    def test_generic_only_returns_generic(self):
        fns = get_functions_for("generic")
        scopes = {fn.scope for fn in fns}
        assert scopes == {"generic"}

    def test_diabetes_returns_generic_and_diabetes(self):
        fns = get_functions_for("diabetes")
        scopes = {fn.scope for fn in fns}
        assert "generic" in scopes
        assert "diabetes" in scopes

    def test_generic_functions_come_first(self):
        fns = get_functions_for("diabetes")
        # All generic fns should appear before any diabetes fn
        found_specialized = False
        for fn in fns:
            if fn.scope == "diabetes":
                found_specialized = True
            if found_specialized and fn.scope == "generic":
                pytest.fail("Generic function found after specialized function")

    def test_unknown_type_returns_only_generic(self):
        fns = get_functions_for("unknown_dataset_type")
        assert all(fn.scope == "generic" for fn in fns)


# ---------------------------------------------------------------------------
# Function contract: (df, meta, **params) -> (DataFrame, Figure|None)
# ---------------------------------------------------------------------------

class TestFunctionContracts:
    """
    Verify that each generic function returns (DataFrame, Figure|None)
    when called with a valid DataFrame and metadata.
    """

    def _assert_contract(self, fn_id: str, df: pd.DataFrame, meta: list[dict], **params):
        fn = REGISTRY[fn_id]
        result = fn.fn(df, meta, **params)
        assert isinstance(result, tuple), f"{fn_id}: should return tuple"
        assert len(result) == 2, f"{fn_id}: tuple should have 2 elements"
        result_df, fig = result
        assert isinstance(result_df, pd.DataFrame), f"{fn_id}: first element should be DataFrame"
        assert fig is None or isinstance(fig, go.Figure), f"{fn_id}: second element should be Figure or None"

    def test_describe(self, generic_df, generic_meta):
        self._assert_contract("generic.describe", generic_df, generic_meta)

    def test_correlation(self, generic_df, generic_meta):
        self._assert_contract("generic.correlation", generic_df, generic_meta, method="pearson")

    def test_value_counts(self, generic_df, generic_meta):
        self._assert_contract("generic.value_counts", generic_df, generic_meta, column="category")

    def test_groupby(self, generic_df, generic_meta):
        self._assert_contract("generic.groupby", generic_df, generic_meta,
                               group_col="category", agg_col="value", agg_func="mean")

    def test_crosstab(self, generic_df, generic_meta):
        self._assert_contract("generic.crosstab", generic_df, generic_meta,
                               row_col="category", col_col="label")

    def test_distribution(self, generic_df, generic_meta):
        self._assert_contract("generic.distribution", generic_df, generic_meta,
                               column="value", bins=10)

    def test_null_analysis(self, generic_df, generic_meta):
        self._assert_contract("generic.null_analysis", generic_df, generic_meta)

    def test_dtypes(self, generic_df, generic_meta):
        self._assert_contract("generic.dtypes", generic_df, generic_meta)

    def test_diabetes_readmission_by_group(self, diabetes_df, diabetes_meta):
        self._assert_contract("diabetes.readmission_by_group", diabetes_df, diabetes_meta,
                               group_by="age", readmission_binary=True)

    def test_diabetes_hba1c_vs_readmission(self, diabetes_df, diabetes_meta):
        self._assert_contract("diabetes.hba1c_vs_readmission", diabetes_df, diabetes_meta,
                               readmission_binary=True)

    def test_diabetes_los_by_readmission(self, diabetes_df, diabetes_meta):
        self._assert_contract("diabetes.los_by_readmission", diabetes_df, diabetes_meta)

    def test_diabetes_medications_vs_los(self, diabetes_df, diabetes_meta):
        self._assert_contract("diabetes.medications_vs_los", diabetes_df, diabetes_meta)

    def test_enrichment_functions_raise_without_enrichment(self, diabetes_df, diabetes_meta):
        from app.core.pipeline import EnrichmentRequiredError
        fn = REGISTRY["diabetes.top_diagnoses"]
        with pytest.raises(EnrichmentRequiredError):
            fn.fn(diabetes_df, diabetes_meta, enrichment_status="none")
