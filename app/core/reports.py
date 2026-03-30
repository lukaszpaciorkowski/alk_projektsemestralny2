"""
reports.py — DB persistence for report items.

Each report item stores a Plotly figure as JSON so it survives page refreshes.
"""

from __future__ import annotations

import json

import plotly.io as pio
from sqlalchemy import text
from sqlalchemy.engine import Engine


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS _reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    fig_json     TEXT    NOT NULL,
    filters      TEXT    NOT NULL DEFAULT '[]',
    dataset_name TEXT    NOT NULL DEFAULT '',
    row_count    INTEGER,
    total_rows   INTEGER,
    sort_order   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""


def ensure_table(engine: Engine) -> None:
    """Create _reports table if it does not exist."""
    with engine.begin() as conn:
        conn.execute(text(_CREATE_TABLE))


def save_report_item(
    engine: Engine,
    title: str,
    fig,
    filters: list[dict],
    dataset_name: str = "",
    row_count: int | None = None,
    total_rows: int | None = None,
) -> int:
    """Persist one report item. Returns the new row id."""
    ensure_table(engine)
    fig_json = pio.to_json(fig)
    filters_json = json.dumps(filters)
    with engine.begin() as conn:
        # sort_order = max(existing) + 1
        row = conn.execute(text("SELECT COALESCE(MAX(sort_order), -1) FROM _reports")).fetchone()
        next_order = (row[0] if row else -1) + 1
        result = conn.execute(
            text(
                "INSERT INTO _reports (title, fig_json, filters, dataset_name, "
                "row_count, total_rows, sort_order) "
                "VALUES (:title, :fig_json, :filters, :dataset_name, "
                ":row_count, :total_rows, :sort_order)"
            ),
            {
                "title": title,
                "fig_json": fig_json,
                "filters": filters_json,
                "dataset_name": dataset_name,
                "row_count": row_count,
                "total_rows": total_rows,
                "sort_order": next_order,
            },
        )
        return result.lastrowid


def list_report_items(engine: Engine) -> list[dict]:
    """Return all report items ordered by sort_order, with deserialized figures."""
    ensure_table(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, title, fig_json, filters, dataset_name, row_count, total_rows "
                "FROM _reports ORDER BY sort_order ASC"
            )
        ).fetchall()
    items = []
    for row in rows:
        try:
            fig = pio.from_json(row[2])
        except Exception:
            continue  # skip corrupted rows
        try:
            filters = json.loads(row[3])
        except Exception:
            filters = []
        items.append(
            {
                "id": row[0],
                "title": row[1],
                "fig": fig,
                "filters": filters,
                "dataset_name": row[4] or "",
                "row_count": row[5],
                "total_rows": row[6],
            }
        )
    return items


def delete_report_item(engine: Engine, item_id: int) -> None:
    """Delete a report item by DB id."""
    ensure_table(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM _reports WHERE id = :id"), {"id": item_id})


def swap_report_items(engine: Engine, id_a: int, id_b: int) -> None:
    """Swap the sort_order of two report items (for up/down reordering)."""
    ensure_table(engine)
    with engine.begin() as conn:
        row_a = conn.execute(
            text("SELECT sort_order FROM _reports WHERE id = :id"), {"id": id_a}
        ).fetchone()
        row_b = conn.execute(
            text("SELECT sort_order FROM _reports WHERE id = :id"), {"id": id_b}
        ).fetchone()
        if row_a is None or row_b is None:
            return
        conn.execute(
            text("UPDATE _reports SET sort_order = :o WHERE id = :id"),
            {"o": row_b[0], "id": id_a},
        )
        conn.execute(
            text("UPDATE _reports SET sort_order = :o WHERE id = :id"),
            {"o": row_a[0], "id": id_b},
        )


def clear_all_reports(engine: Engine) -> None:
    """Delete all report items."""
    ensure_table(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM _reports"))


# ---------------------------------------------------------------------------
# Saved report configurations
# ---------------------------------------------------------------------------

_CREATE_SAVED_REPORTS = """
CREATE TABLE IF NOT EXISTS _saved_reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    config     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def ensure_saved_reports_table(engine: Engine) -> None:
    """Create _saved_reports table if it does not exist."""
    with engine.begin() as conn:
        conn.execute(text(_CREATE_SAVED_REPORTS))


def save_report_config(
    engine: Engine,
    name: str,
    title: str,
    author: str,
    sections: dict,
    items: list[dict],
) -> int:
    """
    Persist the current report configuration (metadata + figures) as a named snapshot.

    Figures are serialised to JSON so the snapshot is self-contained.
    Returns the new row id.
    """
    ensure_saved_reports_table(engine)
    serialized_items = []
    for item in items:
        fig = item.get("fig")
        fig_json = pio.to_json(fig) if fig is not None else ""
        serialized_items.append({
            "title": item.get("title", ""),
            "fig_json": fig_json,
            "filters": item.get("filters", []),
            "dataset_name": item.get("dataset_name", ""),
            "row_count": item.get("row_count"),
            "total_rows": item.get("total_rows"),
        })
    config = json.dumps({
        "title": title,
        "author": author,
        "sections": sections,
        "items": serialized_items,
    })
    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO _saved_reports (name, config) VALUES (:name, :config)"),
            {"name": name, "config": config},
        )
        return result.lastrowid


def list_saved_reports(engine: Engine) -> list[dict]:
    """Return all saved report snapshots (id, name, created_at) newest-first."""
    ensure_saved_reports_table(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, created_at FROM _saved_reports ORDER BY id DESC")
        ).fetchall()
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def load_saved_report(engine: Engine, report_id: int) -> dict:
    """
    Load a saved report config by id.

    Returns a dict with keys: title, author, sections, items.
    Each item has: title, fig_json, filters, dataset_name, row_count, total_rows.
    Raises ValueError if not found.
    """
    ensure_saved_reports_table(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT config FROM _saved_reports WHERE id = :id"),
            {"id": report_id},
        ).fetchone()
    if row is None:
        raise ValueError(f"Saved report {report_id!r} not found.")
    return json.loads(row[0])


def delete_saved_report(engine: Engine, report_id: int) -> None:
    """Delete a saved report snapshot by id."""
    ensure_saved_reports_table(engine)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM _saved_reports WHERE id = :id"),
            {"id": report_id},
        )
