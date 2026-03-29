"""
tests/test_pipeline.py — Unit tests for app/core/pipeline.py

Uses in-memory SQLite databases and temporary CSV files.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import (
    ColumnMeta,
    ImportResult,
    detect_column_types,
    drop_dataset,
    enrich_dataset,
    get_dataset_meta,
    import_csv,
    list_datasets,
    make_table_name,
    validate_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_engine():
    """In-memory SQLite engine with the bootstrap _datasets table."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE _datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                dataset_type TEXT NOT NULL DEFAULT 'generic',
                enrichment_status TEXT NOT NULL DEFAULT 'none',
                row_count INTEGER,
                col_count INTEGER,
                columns TEXT,
                checksum TEXT,
                uploaded_at TEXT NOT NULL
            )
        """))
    return engine


DIABETES_REQUIRED = [
    "patient_nbr", "readmitted", "metformin",
    "time_in_hospital", "num_medications", "diag_1",
]


def _make_diabetes_csv(rows: int = 5) -> io.BytesIO:
    """Return a BytesIO CSV file that looks like the diabetes dataset."""
    all_cols = DIABETES_REQUIRED + [
        "race", "gender", "age", "encounter_id",
        "diag_2", "diag_3", "insulin", "glyburide",
    ]
    data = {c: [f"val_{i}" for i in range(rows)] for c in all_cols}
    data["time_in_hospital"] = list(range(1, rows + 1))
    data["num_medications"] = list(range(5, 5 + rows))
    data["patient_nbr"] = list(range(100, 100 + rows))
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    buf.name = "diabetic_data.csv"
    return buf


def _make_generic_csv(rows: int = 5) -> io.BytesIO:
    df = pd.DataFrame({
        "product_id": range(rows),
        "sales": [100.0 * i for i in range(rows)],
        "region": ["North"] * rows,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    buf.name = "sales.csv"
    return buf


# ---------------------------------------------------------------------------
# make_table_name
# ---------------------------------------------------------------------------

class TestMakeTableName:
    def test_starts_with_ds(self):
        assert make_table_name("myfile.csv").startswith("ds_")

    def test_max_length(self):
        long_name = "a" * 200 + ".csv"
        result = make_table_name(long_name)
        assert len(result) <= 60

    def test_no_spaces_or_special_chars(self):
        result = make_table_name("my file (2024).csv")
        assert " " not in result
        assert "(" not in result
        assert ")" not in result

    def test_deterministic(self):
        a = make_table_name("test.csv")
        b = make_table_name("test.csv")
        assert a == b

    def test_different_names_produce_different_tables(self):
        a = make_table_name("file_a.csv")
        b = make_table_name("file_b.csv")
        assert a != b


# ---------------------------------------------------------------------------
# validate_csv
# ---------------------------------------------------------------------------

class TestValidateCsv:
    def test_valid_df(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        report = validate_csv(df)
        assert report.is_valid is True
        assert report.duplicate_rows == 0

    def test_empty_df_invalid(self):
        df = pd.DataFrame()
        report = validate_csv(df)
        assert report.is_valid is False

    def test_null_counts_tracked(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
        report = validate_csv(df)
        assert report.null_counts["a"] == 1
        assert report.null_counts["b"] == 0

    def test_duplicate_rows_counted(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        report = validate_csv(df)
        assert report.duplicate_rows == 1


# ---------------------------------------------------------------------------
# detect_column_types
# ---------------------------------------------------------------------------

class TestDetectColumnTypes:
    def test_integer_col(self):
        df = pd.DataFrame({"count": [1, 2, 3]})
        meta = detect_column_types(df)
        assert meta[0].name == "count"
        assert meta[0].sql_type == "INTEGER"

    def test_float_col(self):
        df = pd.DataFrame({"price": [1.1, 2.2]})
        meta = detect_column_types(df)
        assert meta[0].sql_type == "REAL"

    def test_text_col(self):
        df = pd.DataFrame({"name": ["a", "b"]})
        meta = detect_column_types(df)
        assert meta[0].sql_type == "TEXT"

    def test_nullable_detection(self):
        df = pd.DataFrame({"a": [1, None]})
        meta = detect_column_types(df)
        assert meta[0].nullable is True


# ---------------------------------------------------------------------------
# import_csv
# ---------------------------------------------------------------------------

class TestImportCsv:
    def test_generic_import(self, mem_engine):
        buf = _make_generic_csv(rows=10)
        result = import_csv(buf, con=mem_engine)
        assert isinstance(result, ImportResult)
        assert result.dataset_type == "generic"
        assert result.row_count == 10

    def test_diabetes_import_detects_type(self, mem_engine):
        buf = _make_diabetes_csv(rows=10)
        result = import_csv(buf, con=mem_engine)
        assert result.dataset_type == "diabetes"

    def test_table_created_in_db(self, mem_engine):
        buf = _make_generic_csv(rows=5)
        result = import_csv(buf, con=mem_engine)
        with mem_engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT COUNT(*) FROM [{result.table_name}]")
            ).fetchone()
        assert rows[0] == 5

    def test_registered_in_datasets(self, mem_engine):
        buf = _make_generic_csv(rows=5)
        result = import_csv(buf, con=mem_engine)
        datasets = list_datasets(mem_engine)
        assert any(d["table_name"] == result.table_name for d in datasets)

    def test_duplicate_checksum_raises(self, mem_engine):
        buf1 = _make_generic_csv(rows=5)
        content = buf1.read()
        buf1.seek(0)

        import_csv(buf1, con=mem_engine)

        buf2 = io.BytesIO(content)
        buf2.name = "sales.csv"
        with pytest.raises(ValueError, match="already imported"):
            import_csv(buf2, con=mem_engine)

    def test_empty_csv_raises(self, mem_engine):
        buf = io.BytesIO(b"col_a,col_b\n")
        buf.name = "empty.csv"
        with pytest.raises(ValueError, match="empty"):
            import_csv(buf, con=mem_engine)


# ---------------------------------------------------------------------------
# drop_dataset
# ---------------------------------------------------------------------------

class TestDropDataset:
    def test_drops_table_and_registry(self, mem_engine):
        buf = _make_generic_csv(rows=5)
        result = import_csv(buf, con=mem_engine)

        drop_dataset(result.table_name, mem_engine)

        datasets = list_datasets(mem_engine)
        assert not any(d["table_name"] == result.table_name for d in datasets)

        # Table should no longer exist
        with mem_engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
                {"n": result.table_name},
            ).fetchall()
        assert len(tables) == 0


# ---------------------------------------------------------------------------
# list_datasets / get_dataset_meta
# ---------------------------------------------------------------------------

class TestListAndMeta:
    def test_list_returns_all(self, mem_engine):
        for i in range(3):
            buf = _make_generic_csv(rows=5)
            buf.seek(0)
            # Need different content for each to avoid checksum clash
            df = pd.DataFrame({"val": [i * 10 + j for j in range(5)], "idx": range(5)})
            inner = io.BytesIO()
            df.to_csv(inner, index=False)
            inner.seek(0)
            inner.name = f"data_{i}.csv"
            import_csv(inner, con=mem_engine)

        datasets = list_datasets(mem_engine)
        assert len(datasets) == 3

    def test_meta_not_found_returns_none(self, mem_engine):
        meta = get_dataset_meta("nonexistent_table", mem_engine)
        assert meta is None
