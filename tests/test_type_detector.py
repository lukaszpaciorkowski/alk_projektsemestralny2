"""
tests/test_type_detector.py — Unit tests for app/core/type_detector.py
"""

from __future__ import annotations

import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.type_detector import detect_dataset_type, dataset_type_label, dataset_type_icon


# Minimum required columns to trigger diabetes detection
DIABETES_REQUIRED = {
    "patient_nbr", "readmitted", "metformin",
    "time_in_hospital", "num_medications", "diag_1",
}


def _make_df(columns: list[str]) -> pd.DataFrame:
    """Create a minimal DataFrame with the given column names."""
    return pd.DataFrame({c: [1] for c in columns})


class TestDetectDatasetType:
    def test_diabetes_all_required_columns(self):
        df = _make_df(list(DIABETES_REQUIRED))
        assert detect_dataset_type(df) == "diabetes"

    def test_diabetes_with_extra_columns(self):
        cols = list(DIABETES_REQUIRED) + ["race", "gender", "age", "insulin"]
        df = _make_df(cols)
        assert detect_dataset_type(df) == "diabetes"

    def test_generic_empty_df(self):
        df = pd.DataFrame()
        assert detect_dataset_type(df) == "generic"

    def test_generic_random_columns(self):
        df = _make_df(["product_id", "sales", "region", "quarter"])
        assert detect_dataset_type(df) == "generic"

    def test_generic_partial_diabetes_columns(self):
        # Only some required columns present — should NOT trigger diabetes
        partial = list(DIABETES_REQUIRED)[:3]
        df = _make_df(partial)
        assert detect_dataset_type(df) == "generic"

    def test_diabetes_column_names_case_insensitive(self):
        # Uppercase column names should still match
        cols = [c.upper() for c in DIABETES_REQUIRED]
        df = _make_df(cols)
        assert detect_dataset_type(df) == "diabetes"

    def test_diabetes_column_names_with_whitespace(self):
        # Trailing/leading spaces should be stripped
        cols = [f" {c} " for c in DIABETES_REQUIRED]
        df = _make_df(cols)
        assert detect_dataset_type(df) == "diabetes"


class TestHelpers:
    def test_label_diabetes(self):
        assert "Diabetes" in dataset_type_label("diabetes")

    def test_label_generic(self):
        assert "Generic" in dataset_type_label("generic")

    def test_label_unknown(self):
        result = dataset_type_label("unknown_type")
        assert isinstance(result, str)

    def test_icon_diabetes(self):
        assert dataset_type_icon("diabetes") == "🧬"

    def test_icon_generic(self):
        assert isinstance(dataset_type_icon("generic"), str)
