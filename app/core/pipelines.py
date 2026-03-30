"""
pipelines.py — Custom analytical pipeline storage and execution engine.

Public API:
    ensure_pipelines_tables(engine)
    create_pipeline(engine, name, description, dataset_type, steps) -> int
    list_pipelines(engine) -> list[dict]
    get_pipeline(engine, pipeline_id) -> dict | None
    update_pipeline(engine, pipeline_id, *, name, description, dataset_type, steps)
    delete_pipeline(engine, pipeline_id)
    clone_pipeline(engine, pipeline_id, new_name) -> int
    export_pipeline_json(engine, pipeline_id) -> str
    import_pipeline_json(engine, json_str, name_override) -> int
    execute_pipeline_step(engine, step, dataset_table, meta, enrichment_status) -> dict
    start_pipeline_run(engine, pipeline_id, dataset_table) -> int
    save_pipeline_run(engine, run_id, results, status)
    clear_pipeline_runs(engine, pipeline_id)
    list_runs_for_pipeline(engine, pipeline_id) -> list[dict]
    get_run_results(engine, run_id) -> dict | None
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import plotly.io as pio
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.registry import REGISTRY

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_PIPELINES_SQL = """
CREATE TABLE IF NOT EXISTS _pipelines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    dataset_type TEXT NOT NULL DEFAULT 'generic',
    steps        TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS _pipeline_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id   INTEGER NOT NULL REFERENCES _pipelines(id),
    dataset_table TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running',
    results       TEXT,
    started_at    TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at  TEXT
)
"""

# ---------------------------------------------------------------------------
# Pre-built templates (seeded on first table creation)
# ---------------------------------------------------------------------------

_TEMPLATES = [
    {
        "name": "Quick EDA",
        "description": "Rapid EDA: shape overview, missing data, correlations, and distribution.",
        "dataset_type": "generic",
        "steps": [
            {"step_id": "step_1", "function_id": "generic.describe",     "label": "Overview Statistics", "params": {},                       "filters": [], "add_to_report": True},
            {"step_id": "step_2", "function_id": "generic.null_analysis", "label": "Missing Data",        "params": {},                       "filters": [], "add_to_report": True},
            {"step_id": "step_3", "function_id": "generic.correlation",   "label": "Correlation Matrix",  "params": {"method": "pearson"},     "filters": [], "add_to_report": True},
            {"step_id": "step_4", "function_id": "generic.distribution",  "label": "Distribution",        "params": {"bins": 30},              "filters": [], "add_to_report": True},
        ],
    },
    {
        "name": "Statistical Testing Suite",
        "description": "Normality, outlier detection, two-group and multi-group comparisons.",
        "dataset_type": "generic",
        "steps": [
            {"step_id": "step_1", "function_id": "generic.normality_test",    "label": "Normality Check",       "params": {},                                   "filters": [], "add_to_report": True},
            {"step_id": "step_2", "function_id": "generic.outlier_detection", "label": "Outlier Detection",     "params": {"method": "zscore", "threshold": 3}, "filters": [], "add_to_report": True},
            {"step_id": "step_3", "function_id": "generic.two_group_test",    "label": "Two-Group Comparison",  "params": {"test_type": "t-test"},              "filters": [], "add_to_report": True},
            {"step_id": "step_4", "function_id": "generic.multi_group_test",  "label": "Multi-Group (ANOVA)",   "params": {},                                   "filters": [], "add_to_report": True},
        ],
    },
    {
        "name": "Full Diabetes Analysis",
        "description": "Complete analysis of the UCI Diabetes 130-US Hospitals dataset.",
        "dataset_type": "diabetes",
        "steps": [
            {"step_id": "step_1", "function_id": "generic.describe",                  "label": "Overview Statistics",           "params": {},                                          "filters": [], "add_to_report": True},
            {"step_id": "step_2", "function_id": "diabetes.readmission_by_group",     "label": "Readmission by Age",            "params": {"group_by": "age", "readmission_binary": True}, "filters": [], "add_to_report": True},
            {"step_id": "step_3", "function_id": "diabetes.hba1c_vs_readmission",     "label": "HbA1c vs Readmission",          "params": {"readmission_binary": True},                "filters": [], "add_to_report": True},
            {"step_id": "step_4", "function_id": "diabetes.top_diagnoses",            "label": "Top 10 Diagnoses",              "params": {"top_n": 10},                               "filters": [], "add_to_report": True},
            {"step_id": "step_5", "function_id": "diabetes.medication_frequency",     "label": "Medication Frequency",          "params": {"top_n": 15},                               "filters": [], "add_to_report": True},
            {"step_id": "step_6", "function_id": "diabetes.los_by_readmission",       "label": "Length of Stay by Readmission", "params": {},                                          "filters": [], "add_to_report": True},
        ],
    },
    {
        "name": "Dimensionality Reduction",
        "description": "PCA projection, K-Means clustering, and random forest feature importance.",
        "dataset_type": "generic",
        "steps": [
            {"step_id": "step_1", "function_id": "generic.pca",                "label": "PCA (3 components)",  "params": {"n_components": 3, "scale": True, "x_component": 1, "y_component": 2}, "filters": [], "add_to_report": True},
            {"step_id": "step_2", "function_id": "generic.kmeans",             "label": "K-Means (3 clusters)", "params": {"n_clusters": 3, "scale": True},                                       "filters": [], "add_to_report": True},
            {"step_id": "step_3", "function_id": "generic.feature_importance", "label": "Feature Importance",  "params": {"max_features": 20},                                                   "filters": [], "add_to_report": True},
        ],
    },
]


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------

def ensure_pipelines_tables(engine: Engine) -> None:
    """Create tables and upsert built-in templates by name.

    Templates are matched by name; if a row with that name already exists its
    steps are updated to the current definition so stale DB records are always
    corrected when the app starts.  User-created pipelines (names not in
    _TEMPLATES) are never touched.
    """
    with engine.begin() as conn:
        conn.execute(text(_CREATE_PIPELINES_SQL))
        conn.execute(text(_CREATE_RUNS_SQL))

        existing_names: set[str] = {
            row[0]
            for row in conn.execute(text("SELECT name FROM _pipelines")).fetchall()
        }

        for t in _TEMPLATES:
            params = {
                "name": t["name"],
                "description": t["description"],
                "dataset_type": t["dataset_type"],
                "steps": json.dumps(t["steps"]),
            }
            if t["name"] in existing_names:
                # Update steps/description in case the template changed
                conn.execute(
                    text(
                        "UPDATE _pipelines "
                        "SET description=:description, dataset_type=:dataset_type, steps=:steps "
                        "WHERE name=:name"
                    ),
                    params,
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO _pipelines (name, description, dataset_type, steps) "
                        "VALUES (:name, :description, :dataset_type, :steps)"
                    ),
                    params,
                )


# ---------------------------------------------------------------------------
# Pipeline CRUD
# ---------------------------------------------------------------------------

def create_pipeline(
    engine: Engine,
    name: str,
    description: str,
    dataset_type: str,
    steps: list[dict],
) -> int:
    """Insert a new pipeline; returns the new id."""
    ensure_pipelines_tables(engine)
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "INSERT INTO _pipelines (name, description, dataset_type, steps) "
                "VALUES (:name, :description, :dataset_type, :steps)"
            ),
            {
                "name": name,
                "description": description,
                "dataset_type": dataset_type,
                "steps": json.dumps(steps),
            },
        )
        return result.lastrowid


def _parse_pipeline_row(row) -> dict:
    try:
        steps = json.loads(row[4])
    except Exception:
        steps = []
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "dataset_type": row[3],
        "steps": steps,
        "step_count": len(steps),
        "created_at": row[5],
        "updated_at": row[6],
    }


def list_pipelines(engine: Engine) -> list[dict]:
    """Return all pipelines ordered by id."""
    ensure_pipelines_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, name, description, dataset_type, steps, created_at, updated_at "
                "FROM _pipelines ORDER BY id ASC"
            )
        ).fetchall()
    return [_parse_pipeline_row(r) for r in rows]


def get_pipeline(engine: Engine, pipeline_id: int) -> dict | None:
    """Return a single pipeline by id, or None."""
    ensure_pipelines_tables(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, name, description, dataset_type, steps, created_at, updated_at "
                "FROM _pipelines WHERE id = :id"
            ),
            {"id": pipeline_id},
        ).fetchone()
    return _parse_pipeline_row(row) if row else None


def update_pipeline(
    engine: Engine,
    pipeline_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    dataset_type: str | None = None,
    steps: list[dict] | None = None,
) -> None:
    """Update non-None fields; always refreshes updated_at."""
    existing = get_pipeline(engine, pipeline_id)
    if existing is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE _pipelines SET name=:name, description=:description, "
                "dataset_type=:dataset_type, steps=:steps, updated_at=datetime('now') "
                "WHERE id=:id"
            ),
            {
                "name":         name         if name         is not None else existing["name"],
                "description":  description  if description  is not None else existing["description"],
                "dataset_type": dataset_type if dataset_type is not None else existing["dataset_type"],
                "steps":        json.dumps(steps) if steps is not None else json.dumps(existing["steps"]),
                "id":           pipeline_id,
            },
        )


def delete_pipeline(engine: Engine, pipeline_id: int) -> None:
    """Delete pipeline and all its run records."""
    ensure_pipelines_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM _pipeline_runs WHERE pipeline_id = :id"), {"id": pipeline_id})
        conn.execute(text("DELETE FROM _pipelines WHERE id = :id"),             {"id": pipeline_id})


def clone_pipeline(engine: Engine, pipeline_id: int, new_name: str) -> int:
    """Clone a pipeline under a new name; returns new id."""
    pl = get_pipeline(engine, pipeline_id)
    if pl is None:
        raise ValueError(f"Pipeline {pipeline_id} not found")
    return create_pipeline(
        engine,
        name=new_name,
        description=pl["description"],
        dataset_type=pl["dataset_type"],
        steps=pl["steps"],
    )


def export_pipeline_json(engine: Engine, pipeline_id: int) -> str:
    """Return portable JSON (no id/timestamps) suitable for sharing."""
    pl = get_pipeline(engine, pipeline_id)
    if pl is None:
        raise ValueError(f"Pipeline {pipeline_id} not found")
    return json.dumps(
        {
            "name":         pl["name"],
            "description":  pl["description"],
            "dataset_type": pl["dataset_type"],
            "steps":        pl["steps"],
        },
        indent=2,
    )


def import_pipeline_json(
    engine: Engine,
    json_str: str,
    name_override: str | None = None,
) -> int:
    """Parse, validate function_ids, and create a pipeline. Returns new id."""
    data = json.loads(json_str)
    steps = data.get("steps", [])
    for step in steps:
        fn_id = step.get("function_id", "")
        if fn_id not in REGISTRY:
            raise ValueError(f"Unknown function_id: '{fn_id}'")
    return create_pipeline(
        engine,
        name=name_override or data.get("name", "Imported Pipeline"),
        description=data.get("description", ""),
        dataset_type=data.get("dataset_type", "generic"),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def execute_pipeline_step(
    engine: Engine,
    step: dict,
    dataset_table: str,
    meta: list[dict],
    enrichment_status: str,
) -> dict:
    """
    Execute one pipeline step against dataset_table.

    Never raises — errors are captured in the returned dict.
    Returns:
        {step_id, label, status, result_summary, result_df_json,
         fig_json, error, duration_s}
    """
    from app.core.pipeline import EnrichmentRequiredError
    from app.core.query import Filter, fetch_table

    fn_id = step.get("function_id", "")
    fn = REGISTRY.get(fn_id)

    result: dict = {
        "step_id":        step.get("step_id", ""),
        "label":          step.get("label", fn_id),
        "status":         "failed",
        "result_summary": "",
        "result_df_json": None,
        "fig_json":       None,
        "error":          None,
        "duration_s":     0.0,
    }

    if fn is None:
        result["error"] = f"Unknown function: '{fn_id}'"
        return result

    filters = [
        Filter(f["column"], f["op"], f.get("value"))
        for f in (step.get("filters") or [])
        if f.get("column") and f.get("op")
    ]

    t0 = time.perf_counter()
    try:
        df = fetch_table(dataset_table, engine, filters=filters, limit=500_000)
        call_params = dict(step.get("params") or {})
        if fn.requires_enrichment:
            call_params["con"] = engine
            call_params["table_name"] = dataset_table
            call_params["enrichment_status"] = enrichment_status

        result_df, fig = fn.fn(df, meta, **call_params)

        result["status"]         = "completed"
        result["result_summary"] = f"shape: {result_df.shape[0]}×{result_df.shape[1]}"
        result["result_df_json"] = result_df.head(2000).to_json(orient="records")
        if fig is not None:
            result["fig_json"] = pio.to_json(fig)

    except EnrichmentRequiredError as exc:
        result["error"] = f"Enrichment required: {exc}"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["duration_s"] = round(time.perf_counter() - t0, 2)

    return result


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------

def start_pipeline_run(engine: Engine, pipeline_id: int, dataset_table: str) -> int:
    """Insert a run record with status='running'; returns new run id."""
    ensure_pipelines_tables(engine)
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "INSERT INTO _pipeline_runs (pipeline_id, dataset_table, status) "
                "VALUES (:pid, :table, 'running')"
            ),
            {"pid": pipeline_id, "table": dataset_table},
        )
        return result.lastrowid


def save_pipeline_run(
    engine: Engine,
    run_id: int,
    results: list[dict],
    status: str,
) -> None:
    """Update run record with results and final status."""
    ensure_pipelines_tables(engine)
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE _pipeline_runs SET status=:status, results=:results, "
                "completed_at=:completed_at WHERE id=:id"
            ),
            {
                "status":       status,
                "results":      json.dumps(results),
                "completed_at": completed_at,
                "id":           run_id,
            },
        )


def clear_pipeline_runs(engine: Engine, pipeline_id: int) -> None:
    """Delete all run records for a pipeline."""
    ensure_pipelines_tables(engine)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM _pipeline_runs WHERE pipeline_id = :id"),
            {"id": pipeline_id},
        )


def list_runs_for_pipeline(engine: Engine, pipeline_id: int) -> list[dict]:
    """Return all runs for a pipeline, newest first."""
    ensure_pipelines_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, dataset_table, status, started_at, completed_at "
                "FROM _pipeline_runs WHERE pipeline_id = :pid ORDER BY id DESC"
            ),
            {"pid": pipeline_id},
        ).fetchall()
    return [
        {
            "id":            row[0],
            "dataset_table": row[1],
            "status":        row[2],
            "started_at":    row[3],
            "completed_at":  row[4],
        }
        for row in rows
    ]


def get_run_results(engine: Engine, run_id: int) -> dict | None:
    """Return full run record with parsed results list, or None."""
    ensure_pipelines_tables(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, pipeline_id, dataset_table, status, results, started_at, completed_at "
                "FROM _pipeline_runs WHERE id = :id"
            ),
            {"id": run_id},
        ).fetchone()
    if row is None:
        return None
    try:
        results = json.loads(row[4]) if row[4] else []
    except Exception:
        results = []
    return {
        "id":            row[0],
        "pipeline_id":   row[1],
        "dataset_table": row[2],
        "status":        row[3],
        "results":       results,
        "started_at":    row[5],
        "completed_at":  row[6],
    }
