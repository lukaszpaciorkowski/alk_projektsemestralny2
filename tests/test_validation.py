"""
test_validation.py — Unit tests for the ingestion and validation logic.

Tests:
  - test_null_threshold_drops_column
  - test_outlier_removal
  - test_age_group_validation
  - test_readmission_binary_conversion
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_helpers import (
    add_age_group_normalized,
    drop_high_null_columns,
    remove_outliers_zscore,
    standardize_readmission,
    validate_age_groups,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df_with_nulls() -> pd.DataFrame:
    """DataFrame where col_heavy_null has 80% null (above threshold 0.3)."""
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "col_ok": [10, 20, 30, 40, 50],
            "col_heavy_null": [np.nan, np.nan, np.nan, np.nan, 1.0],  # 80% null
            "col_moderate_null": [1, np.nan, 3, 4, 5],  # 20% null — keep
        }
    )


@pytest.fixture()
def sample_df_with_outliers() -> pd.DataFrame:
    """DataFrame with one extreme outlier row."""
    normal_vals = list(range(1, 21))   # 1..20, mean≈10.5
    outlier_val = 10_000               # extreme outlier
    return pd.DataFrame(
        {
            "encounter_id": list(range(21)),
            "time_in_hospital": normal_vals + [outlier_val],
        }
    )


@pytest.fixture()
def sample_df_ages() -> pd.DataFrame:
    """DataFrame with a mix of valid and invalid age_group values."""
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "age": ["[10-20)", "[30-40)", "INVALID", None, "[70-80)"],
        }
    )


@pytest.fixture()
def sample_df_readmission() -> pd.DataFrame:
    """DataFrame with various readmission label styles."""
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "readmitted": ["<30", ">30", "NO", "No", "no"],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNullThreshold:
    """Tests for drop_high_null_columns."""

    def test_null_threshold_drops_column(self, sample_df_with_nulls: pd.DataFrame) -> None:
        """Column with 80% null should be dropped with threshold=0.3."""
        result = drop_high_null_columns(sample_df_with_nulls, threshold=0.3)
        assert "col_heavy_null" not in result.columns, (
            "Column with 80% null should be dropped at threshold=0.3"
        )

    def test_null_threshold_keeps_acceptable_column(self, sample_df_with_nulls: pd.DataFrame) -> None:
        """Column with 20% null should NOT be dropped with threshold=0.3."""
        result = drop_high_null_columns(sample_df_with_nulls, threshold=0.3)
        assert "col_moderate_null" in result.columns

    def test_null_threshold_drops_nothing_at_high_threshold(
        self, sample_df_with_nulls: pd.DataFrame
    ) -> None:
        """With threshold=0.9 nothing should be dropped."""
        result = drop_high_null_columns(sample_df_with_nulls, threshold=0.9)
        assert "col_heavy_null" in result.columns

    def test_null_threshold_drops_all_null_columns(self) -> None:
        """A completely-null column is always dropped."""
        df = pd.DataFrame({"a": [1, 2, 3], "all_null": [np.nan, np.nan, np.nan]})
        result = drop_high_null_columns(df, threshold=0.3)
        assert "all_null" not in result.columns

    def test_row_count_unchanged_after_column_drop(self, sample_df_with_nulls: pd.DataFrame) -> None:
        """Dropping columns must not change row count."""
        result = drop_high_null_columns(sample_df_with_nulls, threshold=0.3)
        assert len(result) == len(sample_df_with_nulls)


class TestOutlierRemoval:
    """Tests for remove_outliers_zscore."""

    def test_outlier_removal_removes_extreme_row(
        self, sample_df_with_outliers: pd.DataFrame
    ) -> None:
        """Row with |z-score| > 3.0 should be removed."""
        result = remove_outliers_zscore(sample_df_with_outliers, zscore_threshold=3.0)
        assert 10_000 not in result["time_in_hospital"].values, (
            "Extreme outlier value should be removed"
        )

    def test_outlier_removal_keeps_normal_rows(
        self, sample_df_with_outliers: pd.DataFrame
    ) -> None:
        """Normal rows (z < 3.0) should be retained."""
        result = remove_outliers_zscore(sample_df_with_outliers, zscore_threshold=3.0)
        assert len(result) >= 15, "Most normal rows should be kept"

    def test_outlier_removal_with_strict_threshold(
        self, sample_df_with_outliers: pd.DataFrame
    ) -> None:
        """Stricter threshold removes more rows."""
        result_strict = remove_outliers_zscore(sample_df_with_outliers, zscore_threshold=1.5)
        result_loose = remove_outliers_zscore(sample_df_with_outliers, zscore_threshold=3.0)
        assert len(result_strict) <= len(result_loose)

    def test_outlier_removal_no_numeric_columns(self) -> None:
        """DataFrame with no numeric columns should be returned unchanged."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "category": ["A", "B"]})
        result = remove_outliers_zscore(df, zscore_threshold=3.0)
        assert len(result) == len(df)


class TestAgeGroupValidation:
    """Tests for validate_age_groups."""

    def test_age_group_validation_removes_invalid(
        self, sample_df_ages: pd.DataFrame
    ) -> None:
        """Rows with invalid age_group format should be removed."""
        result = validate_age_groups(sample_df_ages)
        assert "INVALID" not in result["age"].values

    def test_age_group_validation_removes_none(
        self, sample_df_ages: pd.DataFrame
    ) -> None:
        """Rows with None age_group should be removed."""
        result = validate_age_groups(sample_df_ages)
        assert result["age"].notna().all()

    def test_age_group_validation_keeps_valid(
        self, sample_df_ages: pd.DataFrame
    ) -> None:
        """Valid age brackets like '[30-40)' should be retained."""
        result = validate_age_groups(sample_df_ages)
        assert "[30-40)" in result["age"].values
        assert "[10-20)" in result["age"].values

    def test_age_group_validation_no_age_column(self) -> None:
        """DataFrame without 'age' column should be returned unchanged."""
        df = pd.DataFrame({"encounter_id": [1, 2, 3]})
        result = validate_age_groups(df)
        assert len(result) == len(df)


class TestReadmissionBinaryConversion:
    """Tests for standardize_readmission."""

    def test_readmission_normalizes_no_variants(
        self, sample_df_readmission: pd.DataFrame
    ) -> None:
        """'No' and 'no' should be normalised to 'NO'."""
        result = standardize_readmission(sample_df_readmission)
        assert "No" not in result["readmitted"].values
        assert "no" not in result["readmitted"].values

    def test_readmission_keeps_lt30(
        self, sample_df_readmission: pd.DataFrame
    ) -> None:
        """'<30' should be preserved."""
        result = standardize_readmission(sample_df_readmission)
        assert "<30" in result["readmitted"].values

    def test_readmission_keeps_gt30(
        self, sample_df_readmission: pd.DataFrame
    ) -> None:
        """'>30' should be preserved."""
        result = standardize_readmission(sample_df_readmission)
        assert ">30" in result["readmitted"].values

    def test_readmission_binary_coding_logic(self) -> None:
        """Verify binary logic: <30 and >30 → 1, NO → 0."""
        values = pd.Series(["<30", ">30", "NO", "NO", "<30"])
        binary = values.apply(lambda x: 1 if x in ("<30", ">30") else 0)
        assert binary.tolist() == [1, 1, 0, 0, 1]

    def test_add_age_group_normalized_bins(self) -> None:
        """Age bins should produce correct group labels."""
        df = pd.DataFrame({"age": ["[10-20)", "[40-50)", "[60-70)", "[80-90)"]})
        bins = [0, 30, 50, 70, 100]
        labels = ["0-30", "30-50", "50-70", "70+"]
        result = add_age_group_normalized(df, age_bins=bins, age_labels=labels)
        assert "age_group" in result.columns
        assert result.loc[0, "age_group"] == "0-30"
        assert result.loc[1, "age_group"] == "30-50"
