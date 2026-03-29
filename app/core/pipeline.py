"""
pipeline.py — Generic data import pipeline for patient datasets.

Public API:
    import_csv(csv_path, config_path) -> dict
        Detect dataset type, validate, load into SQLite.
        Returns metadata dict with dataset_type, row_count, etc.

    enrich_dataset(dataset_name, config_path) -> dict
        Run type-specific enrichment (statistics, correlations, domain analytics).
        Returns dict with enrichment results.

Dataset type detection:
    "diabetes"  — CSV contains the UCI 130-US Hospitals signature columns
    "generic"   — All other CSVs; stored as-is in a per-dataset SQLite table
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# ── Diabetes column signature ────────────────────────────────────────────────
_DIABETES_SIGNATURE: frozenset[str] = frozenset(
    {"encounter_id", "patient_nbr", "readmitted", "A1Cresult", "diag_1", "admission_type_id"}
)
_DIABETES_MATCH_THRESHOLD = 4  # need at least 4 of the 6 signature cols


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_engine(config: dict):
    db_path = config["database"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def _sanitize_name(csv_path: str) -> str:
    """Convert a file path to a safe SQLite table / dataset name."""
    stem = Path(csv_path).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Return 'diabetes' or 'generic' based on column signature."""
    overlap = _DIABETES_SIGNATURE & set(df.columns)
    if len(overlap) >= _DIABETES_MATCH_THRESHOLD:
        return "diabetes"
    return "generic"


# ── Datasets registry ────────────────────────────────────────────────────────

def _ensure_registry(engine) -> None:
    """Create the datasets_registry table if it does not exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS datasets_registry (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT    NOT NULL UNIQUE,
                dataset_type TEXT    NOT NULL,
                source_path  TEXT,
                row_count    INTEGER,
                col_count    INTEGER,
                imported_at  TEXT
            )
        """))


def _register_dataset(
    engine,
    dataset_name: str,
    dataset_type: str,
    source_path: str,
    row_count: int,
    col_count: int,
) -> None:
    _ensure_registry(engine)
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO datasets_registry (dataset_name, dataset_type, source_path,
                                           row_count, col_count, imported_at)
            VALUES (:name, :dtype, :src, :rows, :cols, :ts)
            ON CONFLICT(dataset_name) DO UPDATE SET
                dataset_type = excluded.dataset_type,
                source_path  = excluded.source_path,
                row_count    = excluded.row_count,
                col_count    = excluded.col_count,
                imported_at  = excluded.imported_at
        """), {"name": dataset_name, "dtype": dataset_type, "src": source_path,
               "rows": row_count, "cols": col_count, "ts": now})
    logger.info("Registered dataset '%s' (%s) in registry.", dataset_name, dataset_type)


# ── Diabetes import ──────────────────────────────────────────────────────────

def _import_diabetes(df: pd.DataFrame, config: dict, engine) -> int:
    """
    Run the full diabetes ETL: validate → normalise → load 3NF schema.
    Reuses the logic from scripts/01_ingest.py and scripts/02_load.py.
    """
    from scripts.ingest_helpers import (
        add_age_group_normalized,
        drop_high_null_columns,
        remove_outliers_zscore,
        replace_question_marks,
        standardize_readmission,
        validate_age_groups,
    )
    from scripts.query_helpers import get_engine as _qe  # noqa: F401  (not used)
    import scripts.load_helpers as lh  # type: ignore  # loaded lazily

    pipeline = config["pipeline"]
    df = replace_question_marks(df)
    df = drop_high_null_columns(df, threshold=pipeline["null_threshold"])
    df = remove_outliers_zscore(df, zscore_threshold=pipeline["outlier_zscore"])
    df = validate_age_groups(df)
    df = standardize_readmission(df)
    df = add_age_group_normalized(
        df,
        age_bins=pipeline["age_bins"],
        age_labels=pipeline["age_labels"],
    )

    # Save processed CSV so 02_load.py can also be run standalone
    processed_csv = config["data"]["processed_csv"]
    Path(processed_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    logger.info("Diabetes processed CSV saved: %s (%d rows)", processed_csv, len(df))

    # Load into relational schema using load_helpers
    lh.ensure_schema(engine)
    lh.load_admission_types(df, engine)
    lh.load_discharge_types(df, engine)
    lh.load_patients(df, engine)
    lh.load_admissions(df, engine)
    lh.unpivot_medications(df, engine)
    lh.load_diagnoses_lookup(df, engine)
    lh.load_diagnosis_encounters(df, engine)

    logger.info("Diabetes ETL complete: %d rows loaded.", len(df))
    return len(df)


# ── Generic import ───────────────────────────────────────────────────────────

def _import_generic(df: pd.DataFrame, dataset_name: str, engine) -> int:
    """Store a generic CSV as a flat table in SQLite."""
    # Replace '?' with NaN
    df = df.replace("?", np.nan)
    # Drop fully-empty columns
    df = df.dropna(axis=1, how="all")

    table_name = f"generic_{dataset_name}"
    df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=5000)
    logger.info("Generic dataset stored as table '%s' (%d rows).", table_name, len(df))
    return len(df)


# ── Public API ───────────────────────────────────────────────────────────────

def import_csv(csv_path: str, config_path: str = "config.json") -> dict[str, Any]:
    """
    Import a CSV file through the appropriate pipeline.

    Auto-detects dataset type:
      - "diabetes" if the file contains the UCI 130-US Hospitals column signature
      - "generic"  otherwise

    Args:
        csv_path:    Path to the CSV file to import.
        config_path: Path to config.json (default: "config.json").

    Returns:
        dict with keys:
            dataset_name  str   — sanitized name derived from the filename
            dataset_type  str   — "diabetes" | "generic"
            row_count     int   — rows after cleaning
            col_count     int   — columns in the raw CSV
            table_name    str   — SQLite table(s) where data lives
    """
    config = _load_config(config_path)
    engine = _get_engine(config)
    dataset_name = _sanitize_name(csv_path)

    logger.info("Importing CSV: %s → dataset_name='%s'", csv_path, dataset_name)

    df_raw = pd.read_csv(csv_path, low_memory=False)
    col_count = len(df_raw.columns)
    dataset_type = detect_dataset_type(df_raw)
    logger.info("Auto-detected dataset type: %s", dataset_type)

    if dataset_type == "diabetes":
        row_count = _import_diabetes(df_raw, config, engine)
        table_name = "admissions"
    else:
        row_count = _import_generic(df_raw, dataset_name, engine)
        table_name = f"generic_{dataset_name}"

    _register_dataset(engine, dataset_name, dataset_type, csv_path, row_count, col_count)

    return {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "row_count": row_count,
        "col_count": col_count,
        "table_name": table_name,
    }


def enrich_dataset(dataset_name: str, config_path: str = "config.json") -> dict[str, Any]:
    """
    Run type-specific enrichment for a registered dataset.

    For "diabetes": runs readmission / HbA1c / LOS summary analytics and
                    re-generates matplotlib PNGs to outputs/figures/.
    For "generic":  computes descriptive statistics and a correlation matrix,
                    saves figures to outputs/figures/<dataset_name>_*.png.

    Args:
        dataset_name: The sanitized dataset name (from import_csv return value).
        config_path:  Path to config.json.

    Returns:
        dict with enrichment metadata (figures_saved, stats_summary, etc.)
    """
    config = _load_config(config_path)
    engine = _get_engine(config)
    _ensure_registry(engine)

    # Lookup dataset type from registry
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT dataset_type, row_count FROM datasets_registry WHERE dataset_name = :n"),
            {"n": dataset_name},
        ).fetchone()

    if row is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry. Run import_csv first.")

    dataset_type, row_count = row[0], row[1]
    logger.info("Enriching '%s' (type=%s, rows=%d)", dataset_name, dataset_type, row_count)

    if dataset_type == "diabetes":
        return _enrich_diabetes(config, engine)
    else:
        return _enrich_generic(dataset_name, config, engine)


# ── Enrichment: diabetes ─────────────────────────────────────────────────────

def _enrich_diabetes(config: dict, engine) -> dict[str, Any]:
    """Re-generate all 6 diabetes figures and return summary stats."""
    import matplotlib
    matplotlib.use("Agg")

    from scripts.visualize_helpers import generate_all_figures  # type: ignore
    from scripts.query_helpers import summary_stats

    figures_dir = config["output"]["figures_dir"]
    dpi = config["output"]["dpi"]
    palette = config["pipeline"]["palette"]
    top_n = config["pipeline"]["top_n_diagnoses"]

    figs = generate_all_figures(engine, figures_dir, dpi=dpi, palette=palette, top_n=top_n)
    figures_saved = list(figs.keys())

    stats = summary_stats(engine)
    readmission_dist = stats.get("readmission_dist", pd.DataFrame()).to_dict(orient="records")

    logger.info("Diabetes enrichment complete: %d figures generated.", len(figures_saved))
    return {
        "dataset_type": "diabetes",
        "figures_saved": figures_saved,
        "readmission_dist": readmission_dist,
    }


# ── Enrichment: generic ──────────────────────────────────────────────────────

def _enrich_generic(dataset_name: str, config: dict, engine) -> dict[str, Any]:
    """Compute descriptive statistics and correlation matrix; save PNGs."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    table_name = f"generic_{dataset_name}"
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)

    figures_dir = config["output"]["figures_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    dpi = config["output"]["dpi"]

    numeric_df = df.select_dtypes(include="number")
    figures_saved: list[str] = []

    # ── Descriptive statistics figure ──────────────────────────────────────
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_df.columns)), 4))
        ax.axis("off")
        tbl = ax.table(
            cellText=desc.round(2).values,
            rowLabels=desc.index.tolist(),
            colLabels=desc.columns.tolist(),
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.4)
        ax.set_title(f"{dataset_name} — Descriptive Statistics", pad=12, fontsize=10)
        path = f"{figures_dir}/{dataset_name}_describe.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures_saved.append(path)
        logger.info("Saved describe figure: %s", path)

    # ── Correlation heatmap ─────────────────────────────────────────────────
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        n = corr.shape[0]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        sns.heatmap(
            corr,
            annot=n <= 12,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"{dataset_name} — Correlation Matrix")
        path = f"{figures_dir}/{dataset_name}_correlation.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures_saved.append(path)
        logger.info("Saved correlation figure: %s", path)

    # ── Distribution plots for first 6 numeric columns ─────────────────────
    cols_to_plot = numeric_df.columns[:6].tolist()
    if cols_to_plot:
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]
        for i, col in enumerate(cols_to_plot):
            axes_flat[i].hist(numeric_df[col].dropna(), bins=30, edgecolor="white")
            axes_flat[i].set_title(col, fontsize=9)
            axes_flat[i].set_xlabel(col)
            axes_flat[i].set_ylabel("Count")
        for j in range(len(cols_to_plot), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{dataset_name} — Feature Distributions", fontsize=11)
        fig.tight_layout()
        path = f"{figures_dir}/{dataset_name}_distributions.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures_saved.append(path)
        logger.info("Saved distributions figure: %s", path)

    stats_summary = {}
    if not numeric_df.empty:
        stats_summary = {
            col: {
                "mean": float(numeric_df[col].mean()) if not numeric_df[col].isna().all() else None,
                "std": float(numeric_df[col].std()) if not numeric_df[col].isna().all() else None,
                "min": float(numeric_df[col].min()) if not numeric_df[col].isna().all() else None,
                "max": float(numeric_df[col].max()) if not numeric_df[col].isna().all() else None,
            }
            for col in numeric_df.columns[:20]
        }

    return {
        "dataset_type": "generic",
        "dataset_name": dataset_name,
        "figures_saved": figures_saved,
        "stats_summary": stats_summary,
    }
