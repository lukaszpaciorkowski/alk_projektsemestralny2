"""
test_query.py — Unit tests for query_helpers.py using in-memory SQLite.

Tests:
  - test_readmission_by_group_returns_dataframe
  - test_top_diagnoses_returns_top_n
  - test_summary_stats_has_expected_keys
  - test_los_by_readmission_returns_dataframe
  - test_hba1c_vs_readmission_returns_dataframe
  - test_medication_counts_respects_top_n
  - test_medications_vs_los_returns_dataframe
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.query_helpers import (
    hba1c_vs_readmission,
    los_by_readmission,
    medication_counts,
    medications_vs_los,
    readmission_by_group,
    summary_stats,
    top_diagnoses_by_readmission,
)


# ---------------------------------------------------------------------------
# Fixtures — in-memory SQLite with minimal schema and seed data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> Engine:
    """Create an in-memory SQLite engine with test schema and data."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    with eng.begin() as conn:
        # Create tables
        conn.execute(text("""
            CREATE TABLE patients (
                patient_id INTEGER PRIMARY KEY,
                race       TEXT,
                gender     TEXT,
                age_group  TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE admission_types (
                id          INTEGER PRIMARY KEY,
                description TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE discharge_types (
                id          INTEGER PRIMARY KEY,
                description TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE admissions (
                encounter_id        INTEGER PRIMARY KEY,
                patient_id          INTEGER,
                admission_type_id   INTEGER,
                discharge_type_id   INTEGER,
                time_in_hospital    INTEGER,
                num_lab_procedures  INTEGER,
                num_procedures      INTEGER,
                num_medications     INTEGER,
                num_diagnoses       INTEGER,
                hba1c_result        TEXT,
                change_medications  TEXT,
                diabetes_medication TEXT,
                readmission         TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE diagnoses_lookup (
                icd9_code   TEXT PRIMARY KEY,
                description TEXT,
                category    TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE diagnosis_encounters (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                encounter_id       INTEGER,
                icd9_code          TEXT,
                diagnosis_position INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE medications (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                encounter_id     INTEGER,
                drug_name        TEXT,
                prescribed       TEXT,
                change_indicator TEXT
            )
        """))

        # Seed patients
        conn.execute(text("""
            INSERT INTO patients VALUES
            (1, 'Caucasian', 'Male',   '30-50'),
            (2, 'Hispanic',  'Female', '50-70'),
            (3, 'Caucasian', 'Female', '70+'),
            (4, 'Asian',     'Male',   '0-30'),
            (5, 'Caucasian', 'Male',   '50-70')
        """))

        # Seed admission types
        conn.execute(text("""
            INSERT INTO admission_types VALUES (1, 'Emergency'), (2, 'Urgent'), (3, 'Elective')
        """))

        # Seed admissions
        conn.execute(text("""
            INSERT INTO admissions
            (encounter_id, patient_id, admission_type_id, discharge_type_id,
             time_in_hospital, num_lab_procedures, num_procedures, num_medications,
             num_diagnoses, hba1c_result, change_medications, diabetes_medication, readmission)
            VALUES
            (101, 1, 1, 1, 3, 41, 0, 18, 9, 'None', 'Ch', 'Yes', '<30'),
            (102, 2, 1, 1, 5, 59, 0, 23, 6, '>7',   'No', 'Yes', '>30'),
            (103, 3, 2, 2, 2, 11, 5, 13, 7, 'None', 'No', 'No',  'NO'),
            (104, 4, 1, 1, 7, 44, 1, 28, 9, 'Norm', 'Ch', 'Yes', '<30'),
            (105, 5, 3, 3, 4, 33, 0, 15, 8, '>7',   'No', 'Yes', 'NO'),
            (106, 1, 2, 2, 6, 55, 2, 21, 5, 'None', 'Ch', 'Yes', '>30'),
            (107, 2, 1, 1, 1, 12, 0, 10, 3, 'Norm', 'No', 'Yes', 'NO'),
            (108, 3, 1, 1, 8, 67, 3, 30, 9, '>7',   'Ch', 'Yes', '<30'),
            (109, 4, 2, 2, 3, 38, 1, 16, 6, 'None', 'No', 'No',  '>30'),
            (110, 5, 3, 3, 5, 50, 0, 20, 7, 'Norm', 'No', 'Yes', 'NO')
        """))

        # Seed diagnoses
        conn.execute(text("""
            INSERT INTO diagnoses_lookup VALUES
            ('250', 'Diabetes mellitus', 'Endocrine'),
            ('401', 'Essential hypertension', 'Circulatory'),
            ('428', 'Heart failure', 'Circulatory'),
            ('272', 'Disorders of lipoid metabolism', 'Endocrine'),
            ('414', 'Coronary artery disease', 'Circulatory')
        """))
        conn.execute(text("""
            INSERT INTO diagnosis_encounters (encounter_id, icd9_code, diagnosis_position)
            VALUES
            (101, '250', 1), (101, '401', 2),
            (102, '250', 1), (102, '428', 2),
            (103, '401', 1), (103, '272', 2),
            (104, '250', 1), (104, '414', 2),
            (105, '428', 1), (105, '250', 2),
            (106, '250', 1), (107, '401', 1),
            (108, '250', 1), (109, '272', 1),
            (110, '414', 1)
        """))

        # Seed medications
        conn.execute(text("""
            INSERT INTO medications (encounter_id, drug_name, prescribed, change_indicator)
            VALUES
            (101, 'insulin',   'Up',     'Ch'),
            (101, 'metformin', 'Steady', 'No'),
            (102, 'insulin',   'Steady', 'No'),
            (103, 'glipizide', 'Up',     'Ch'),
            (104, 'insulin',   'Down',   'Ch'),
            (105, 'metformin', 'Steady', 'No'),
            (106, 'insulin',   'Up',     'Ch'),
            (107, 'glipizide', 'Steady', 'No'),
            (108, 'insulin',   'Up',     'Ch'),
            (109, 'metformin', 'Down',   'Ch'),
            (110, 'insulin',   'Steady', 'No')
        """))

    return eng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadmissionByGroup:
    """Tests for readmission_by_group()."""

    def test_readmission_by_group_returns_dataframe(self, engine: Engine) -> None:
        """Function should return a non-empty DataFrame."""
        df = readmission_by_group(engine, group_col="age_group")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_readmission_by_group_has_required_columns(self, engine: Engine) -> None:
        """DataFrame must contain group_value, readmission, count, rate."""
        df = readmission_by_group(engine, group_col="age_group")
        assert "group_value" in df.columns
        assert "readmission" in df.columns
        assert "count" in df.columns
        assert "rate" in df.columns

    def test_readmission_by_group_rate_between_0_and_1(self, engine: Engine) -> None:
        """Rate values must be in [0, 1]."""
        df = readmission_by_group(engine, group_col="age_group")
        assert (df["rate"] >= 0).all()
        assert (df["rate"] <= 1).all()

    def test_readmission_by_group_binary_mode(self, engine: Engine) -> None:
        """Binary mode should return only values 0 and 1 in readmission column."""
        df = readmission_by_group(engine, group_col="age_group", binary=True)
        assert set(df["readmission"].unique()).issubset({0, 1})

    def test_readmission_by_group_race(self, engine: Engine) -> None:
        """Should work with group_col='race' from patients table."""
        df = readmission_by_group(engine, group_col="race")
        assert not df.empty


class TestTopDiagnoses:
    """Tests for top_diagnoses_by_readmission()."""

    def test_top_diagnoses_returns_dataframe(self, engine: Engine) -> None:
        """Function should return a DataFrame."""
        df = top_diagnoses_by_readmission(engine, top_n=3)
        assert isinstance(df, pd.DataFrame)

    def test_top_diagnoses_returns_top_n(self, engine: Engine) -> None:
        """Result should contain at most top_n distinct ICD-9 codes."""
        top_n = 3
        df = top_diagnoses_by_readmission(engine, top_n=top_n)
        if not df.empty:
            assert df["icd9_code"].nunique() <= top_n

    def test_top_diagnoses_has_required_columns(self, engine: Engine) -> None:
        """DataFrame must include icd9_code, readmission, count, rate."""
        df = top_diagnoses_by_readmission(engine, top_n=5)
        if not df.empty:
            for col in ["icd9_code", "readmission", "count", "rate"]:
                assert col in df.columns

    def test_top_diagnoses_rate_in_range(self, engine: Engine) -> None:
        """Rate values must be in [0, 1]."""
        df = top_diagnoses_by_readmission(engine, top_n=5)
        if not df.empty:
            assert (df["rate"] >= 0).all()
            assert (df["rate"] <= 1).all()


class TestSummaryStats:
    """Tests for summary_stats()."""

    def test_summary_stats_has_expected_keys(self, engine: Engine) -> None:
        """summary_stats must return a dict with the five expected keys."""
        result = summary_stats(engine)
        assert isinstance(result, dict)
        expected_keys = {"row_counts", "readmission_dist", "age_dist", "los_stats", "hba1c_dist"}
        assert expected_keys.issubset(set(result.keys()))

    def test_summary_stats_row_counts_is_dataframe(self, engine: Engine) -> None:
        """row_counts entry should be a DataFrame."""
        result = summary_stats(engine)
        assert isinstance(result["row_counts"], pd.DataFrame)

    def test_summary_stats_row_counts_positive(self, engine: Engine) -> None:
        """admissions table should report > 0 rows."""
        result = summary_stats(engine)
        admissions_row = result["row_counts"][result["row_counts"]["table"] == "admissions"]
        assert not admissions_row.empty
        assert admissions_row["rows"].iloc[0] > 0

    def test_summary_stats_readmission_dist_not_empty(self, engine: Engine) -> None:
        """readmission_dist should not be empty."""
        result = summary_stats(engine)
        assert not result["readmission_dist"].empty


class TestLosQuery:
    """Tests for los_by_readmission()."""

    def test_los_by_readmission_returns_dataframe(self, engine: Engine) -> None:
        """Function should return a non-empty DataFrame."""
        df = los_by_readmission(engine)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_los_has_mean_los_column(self, engine: Engine) -> None:
        """DataFrame should contain a mean_los column."""
        df = los_by_readmission(engine)
        assert "mean_los" in df.columns

    def test_los_mean_positive(self, engine: Engine) -> None:
        """Mean LOS values should be positive."""
        df = los_by_readmission(engine)
        assert (df["mean_los"] > 0).all()


class TestHba1cQuery:
    """Tests for hba1c_vs_readmission()."""

    def test_hba1c_vs_readmission_returns_dataframe(self, engine: Engine) -> None:
        """Function should return a DataFrame."""
        df = hba1c_vs_readmission(engine)
        assert isinstance(df, pd.DataFrame)

    def test_hba1c_has_required_columns(self, engine: Engine) -> None:
        """DataFrame should have hba1c_result, readmission, count, rate."""
        df = hba1c_vs_readmission(engine)
        if not df.empty:
            for col in ["hba1c_result", "readmission", "count", "rate"]:
                assert col in df.columns


class TestMedicationCounts:
    """Tests for medication_counts()."""

    def test_medication_counts_returns_dataframe(self, engine: Engine) -> None:
        """Function should return a DataFrame."""
        df = medication_counts(engine, top_n=5)
        assert isinstance(df, pd.DataFrame)

    def test_medication_counts_respects_top_n(self, engine: Engine) -> None:
        """Result should have at most top_n rows."""
        top_n = 2
        df = medication_counts(engine, top_n=top_n)
        assert len(df) <= top_n

    def test_medication_counts_sorted_descending(self, engine: Engine) -> None:
        """Results should be sorted by count descending."""
        df = medication_counts(engine, top_n=10)
        if len(df) > 1:
            assert (df["count"].diff().dropna() <= 0).all()


class TestMedicationsVsLos:
    """Tests for medications_vs_los()."""

    def test_medications_vs_los_returns_dataframe(self, engine: Engine) -> None:
        """Function should return a DataFrame."""
        df = medications_vs_los(engine)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_medications_vs_los_has_required_columns(self, engine: Engine) -> None:
        """DataFrame should have num_medications, mean_los, count."""
        df = medications_vs_los(engine)
        for col in ["num_medications", "mean_los", "count"]:
            assert col in df.columns
