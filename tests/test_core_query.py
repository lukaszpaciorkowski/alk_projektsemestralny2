"""
tests/test_core_query.py — Tests for app/core/query.py

Tests each filter op against a known table in an in-memory SQLite DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.query import Filter, fetch_column_stats, fetch_distinct_values, fetch_table, row_count


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def test_engine():
    """In-memory SQLite with a known test_data table."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Dave", None],
        "score": [90.0, 75.0, 85.0, None, 60.0],
        "category": ["A", "B", "A", "C", "B"],
    })
    df.to_sql("test_data", engine, if_exists="replace", index=False)
    return engine


# ---------------------------------------------------------------------------
# fetch_table
# ---------------------------------------------------------------------------

class TestFetchTable:
    def test_no_filters_returns_all(self, test_engine):
        df = fetch_table("test_data", test_engine, limit=100)
        assert len(df) == 5

    def test_limit(self, test_engine):
        df = fetch_table("test_data", test_engine, limit=2)
        assert len(df) == 2

    def test_offset(self, test_engine):
        df_all = fetch_table("test_data", test_engine, limit=100)
        df_offset = fetch_table("test_data", test_engine, limit=100, offset=2)
        assert len(df_offset) == len(df_all) - 2

    def test_order_by_ascending(self, test_engine):
        df = fetch_table("test_data", test_engine, order_by="id", ascending=True, limit=100)
        ids = df["id"].tolist()
        assert ids == sorted(ids)

    def test_order_by_descending(self, test_engine):
        df = fetch_table("test_data", test_engine, order_by="id", ascending=False, limit=100)
        ids = df["id"].tolist()
        assert ids == sorted(ids, reverse=True)


# ---------------------------------------------------------------------------
# Filter ops
# ---------------------------------------------------------------------------

class TestFilterOps:
    def test_eq(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("category", "eq", "A")])
        assert len(df) == 2
        assert all(df["category"] == "A")

    def test_neq(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("category", "neq", "A")])
        assert all(df["category"] != "A")

    def test_gte(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("score", "gte", 85.0)])
        assert all(df["score"] >= 85.0)

    def test_lte(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("score", "lte", 75.0)])
        assert all(df["score"] <= 75.0)

    def test_gt(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("id", "gt", 3)])
        assert all(df["id"] > 3)

    def test_lt(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("id", "lt", 3)])
        assert all(df["id"] < 3)

    def test_in(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("category", "in", ["A", "C"])])
        assert set(df["category"].unique()).issubset({"A", "C"})

    def test_nin(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("category", "nin", ["A"])])
        assert "A" not in df["category"].values

    def test_isnull(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("name", "isnull", None)])
        assert len(df) == 1

    def test_notnull(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("name", "notnull", None)])
        assert len(df) == 4
        assert df["name"].notna().all()

    def test_like(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("name", "like", "%li%")])
        assert len(df) >= 1

    def test_multiple_filters(self, test_engine):
        filters = [
            Filter("category", "eq", "A"),
            Filter("score", "gte", 85.0),
        ]
        df = fetch_table("test_data", test_engine, filters=filters)
        assert all(df["category"] == "A")
        assert all(df["score"] >= 85.0)

    def test_in_empty_list_returns_nothing(self, test_engine):
        df = fetch_table("test_data", test_engine, filters=[Filter("category", "in", [])])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# row_count
# ---------------------------------------------------------------------------

class TestRowCount:
    def test_no_filters(self, test_engine):
        assert row_count("test_data", test_engine) == 5

    def test_with_filter(self, test_engine):
        assert row_count("test_data", test_engine, [Filter("category", "eq", "A")]) == 2


# ---------------------------------------------------------------------------
# fetch_distinct_values
# ---------------------------------------------------------------------------

class TestFetchDistinctValues:
    def test_returns_distinct(self, test_engine):
        vals = fetch_distinct_values("test_data", "category", test_engine)
        assert sorted(vals) == ["A", "B", "C"]

    def test_limit(self, test_engine):
        vals = fetch_distinct_values("test_data", "category", test_engine, limit=2)
        assert len(vals) <= 2

    def test_null_excluded(self, test_engine):
        vals = fetch_distinct_values("test_data", "name", test_engine)
        assert None not in vals


# ---------------------------------------------------------------------------
# fetch_column_stats
# ---------------------------------------------------------------------------

class TestFetchColumnStats:
    def test_numeric_stats(self, test_engine):
        stats = fetch_column_stats("test_data", "score", "float64", test_engine)
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert stats["null_count"] == 1
        assert stats["non_null_count"] == 4

    def test_text_stats(self, test_engine):
        stats = fetch_column_stats("test_data", "category", "object", test_engine)
        assert "unique_count" in stats
        assert stats["unique_count"] == 3
        assert "sample_values" in stats

    def test_null_pct_calculation(self, test_engine):
        stats = fetch_column_stats("test_data", "name", "object", test_engine)
        assert stats["null_count"] == 1
        assert stats["null_pct"] == 20.0
