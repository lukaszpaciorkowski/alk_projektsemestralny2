"""
query.py — Generic SQL builder for the Exploration and Ad Hoc Charts pages.

Never exposes raw SQL to the UI. All WHERE clauses use bound parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass
class Filter:
    column: str
    op: str   # "eq"|"neq"|"gte"|"lte"|"gt"|"lt"|"in"|"nin"|"notnull"|"isnull"|"like"
    value: Any = None  # None for notnull / isnull


_OP_MAP: dict[str, str] = {
    "eq":      "= :val",
    "neq":     "!= :val",
    "gte":     ">= :val",
    "lte":     "<= :val",
    "gt":      "> :val",
    "lt":      "< :val",
    "like":    "LIKE :val",
    "notnull": "IS NOT NULL",
    "isnull":  "IS NULL",
}


def _build_where(filters: list[Filter], params: dict) -> str:
    """
    Build a WHERE clause string from a list of Filter objects.

    Multi-value ops (in / nin) expand to parameterised IN (...) lists.
    Bound params are injected into `params` dict in-place.
    """
    clauses: list[str] = []
    for i, f in enumerate(filters):
        col = f"[{f.column}]"
        if f.op in ("notnull", "isnull"):
            clauses.append(f"{col} {_OP_MAP[f.op]}")
        elif f.op in ("in", "nin"):
            values = list(f.value) if f.value else []
            if not values:
                # empty IN → always false / true
                clauses.append("1=0" if f.op == "in" else "1=1")
                continue
            placeholders = ", ".join(f":in_{i}_{j}" for j in range(len(values)))
            for j, v in enumerate(values):
                params[f"in_{i}_{j}"] = v
            neg = "NOT " if f.op == "nin" else ""
            clauses.append(f"{col} {neg}IN ({placeholders})")
        else:
            param_key = f"p_{i}"
            params[param_key] = f.value
            op_sql = _OP_MAP.get(f.op, "= :val").replace(":val", f":{param_key}")
            clauses.append(f"{col} {op_sql}")

    return "WHERE " + " AND ".join(clauses) if clauses else ""


def fetch_table(
    table_name: str,
    con: Engine,
    filters: list[Filter] | None = None,
    order_by: str | None = None,
    ascending: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Fetch paginated rows from a table with optional filtering and ordering.

    Args:
        table_name: SQLite table name (no interpolation — used as identifier).
        con: SQLAlchemy engine.
        filters: Optional list of Filter objects.
        order_by: Column name to sort by.
        ascending: Sort direction.
        limit: Max rows to return.
        offset: Row offset for pagination.

    Returns:
        DataFrame with the requested rows.
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    where = _build_where(filters or [], params)
    order = ""
    if order_by:
        direction = "ASC" if ascending else "DESC"
        order = f"ORDER BY [{order_by}] {direction}"

    sql = text(
        f"SELECT * FROM [{table_name}] {where} {order} LIMIT :limit OFFSET :offset"
    )
    with con.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def row_count(
    table_name: str,
    con: Engine,
    filters: list[Filter] | None = None,
) -> int:
    """Return the number of rows matching the given filters."""
    params: dict[str, Any] = {}
    where = _build_where(filters or [], params)
    sql = text(f"SELECT COUNT(*) FROM [{table_name}] {where}")
    with con.connect() as conn:
        result = conn.execute(sql, params).fetchone()
    return int(result[0]) if result else 0


def fetch_distinct_values(
    table_name: str,
    col: str,
    con: Engine,
    limit: int = 200,
) -> list:
    """Return up to `limit` distinct non-null values from a column (for filter widgets)."""
    sql = text(
        f"SELECT DISTINCT [{col}] FROM [{table_name}] "
        f"WHERE [{col}] IS NOT NULL ORDER BY [{col}] LIMIT :lim"
    )
    with con.connect() as conn:
        rows = conn.execute(sql, {"lim": limit}).fetchall()
    return [r[0] for r in rows]


def fetch_column_stats(
    table_name: str,
    col: str,
    dtype: str,
    con: Engine,
) -> dict:
    """
    Return basic statistics for a single column.

    For numeric columns: min, max, mean, std, null_count, unique_count.
    For text columns: null_count, unique_count, sample_values.
    """
    with con.connect() as conn:
        null_count_row = conn.execute(
            text(f"SELECT COUNT(*) FROM [{table_name}] WHERE [{col}] IS NULL")
        ).fetchone()
        null_count = int(null_count_row[0]) if null_count_row else 0

        total_row = conn.execute(
            text(f"SELECT COUNT(*) FROM [{table_name}]")
        ).fetchone()
        total = int(total_row[0]) if total_row else 0

        unique_row = conn.execute(
            text(f"SELECT COUNT(DISTINCT [{col}]) FROM [{table_name}]")
        ).fetchone()
        unique_count = int(unique_row[0]) if unique_row else 0

        stats: dict[str, Any] = {
            "null_count": null_count,
            "non_null_count": total - null_count,
            "null_pct": round(null_count / total * 100, 1) if total else 0.0,
            "unique_count": unique_count,
            "total": total,
        }

        if "int" in dtype or "float" in dtype:
            num_row = conn.execute(
                text(
                    f"SELECT MIN(CAST([{col}] AS REAL)), MAX(CAST([{col}] AS REAL)), "
                    f"AVG(CAST([{col}] AS REAL)) FROM [{table_name}] WHERE [{col}] IS NOT NULL"
                )
            ).fetchone()
            if num_row:
                stats["min"] = num_row[0]
                stats["max"] = num_row[1]
                stats["mean"] = round(num_row[2], 4) if num_row[2] is not None else None

            # std via pandas (SQLite lacks STDDEV)
            df_col = pd.read_sql(
                text(f"SELECT [{col}] FROM [{table_name}] WHERE [{col}] IS NOT NULL LIMIT 50000"),
                conn,
            )
            if not df_col.empty:
                stats["std"] = round(float(df_col.iloc[:, 0].std()), 4)
                stats["median"] = round(float(df_col.iloc[:, 0].median()), 4)
        else:
            sample_rows = conn.execute(
                text(
                    f"SELECT DISTINCT [{col}] FROM [{table_name}] "
                    f"WHERE [{col}] IS NOT NULL LIMIT 10"
                )
            ).fetchall()
            stats["sample_values"] = [r[0] for r in sample_rows]

    return stats
