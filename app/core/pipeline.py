"""
pipeline.py — Single import pipeline for all CSV datasets.

Flow: upload → validate → detect type → create table → register → optional enrich
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.type_detector import detect_dataset_type

logger = logging.getLogger(__name__)

DB_PATH = "data/data.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnMeta:
    name: str
    dtype: str          # pandas dtype string, e.g. "int64", "object", "float64"
    sql_type: str       # SQLite type: "INTEGER", "REAL", "TEXT"
    nullable: bool
    unique_count: int


@dataclass
class ValidationReport:
    null_counts: dict[str, int]
    rows_dropped: int
    type_coercion_warnings: list[str]
    duplicate_rows: int
    is_valid: bool


@dataclass
class ImportResult:
    table_name: str
    display_name: str
    dataset_type: str
    row_count: int
    col_count: int
    column_meta: list[ColumnMeta]
    checksum: str
    validation: ValidationReport


@dataclass
class EnrichmentResult:
    meds_rows: int
    diag_rows: int


class EnrichmentRequiredError(Exception):
    """Raised when an analytics function requires enrichment that has not been run."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_engine(db_path: str = DB_PATH) -> Engine:
    """Return a SQLAlchemy engine for the app database."""
    from pathlib import Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def make_table_name(filename: str) -> str:
    """
    Produce a valid SQLite identifier from a filename.

    Format: ds_<sanitised_stem>_<6-char md5>
    Max length: 60 chars total.
    """
    stem = re.sub(r"[^a-z0-9]+", "_", filename.lower().rsplit(".", 1)[0])
    stem = stem.strip("_")[:40]
    suffix = hashlib.md5(filename.encode()).hexdigest()[:6]
    return f"ds_{stem}_{suffix}"


def _pandas_to_sql_type(dtype: str) -> str:
    """Map pandas dtype string to SQLite column type."""
    if "int" in dtype:
        return "INTEGER"
    if "float" in dtype:
        return "REAL"
    return "TEXT"


def detect_column_types(df: pd.DataFrame) -> list[ColumnMeta]:
    """Build ColumnMeta list from a DataFrame."""
    meta: list[ColumnMeta] = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        meta.append(
            ColumnMeta(
                name=col,
                dtype=dtype_str,
                sql_type=_pandas_to_sql_type(dtype_str),
                nullable=bool(df[col].isna().any()),
                unique_count=int(df[col].nunique(dropna=True)),
            )
        )
    return meta


def validate_csv(df: pd.DataFrame) -> ValidationReport:
    """
    Run basic validation checks on a freshly-read DataFrame.

    Returns a ValidationReport — does NOT modify the DataFrame.
    """
    null_counts = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    warnings: list[str] = []

    # Warn on columns that are all-null
    all_null = [c for c, n in null_counts.items() if n == len(df)]
    for col in all_null:
        warnings.append(f"Column '{col}' is entirely null.")

    is_valid = len(df) > 0

    return ValidationReport(
        null_counts={k: int(v) for k, v in null_counts.items()},
        rows_dropped=0,
        type_coercion_warnings=warnings,
        duplicate_rows=duplicate_rows,
        is_valid=is_valid,
    )


def _file_checksum(raw_bytes: bytes) -> str:
    return hashlib.md5(raw_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------

def import_csv(
    file,
    con: Engine,
    config: dict | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    description: str = "",
) -> ImportResult:
    """
    Full import flow: read → validate → detect → create table → register.

    Args:
        file: File-like object (Streamlit UploadedFile or path str).
        con: SQLAlchemy engine pointing at data.db.
        config: Optional pipeline config dict.
        delimiter: CSV delimiter (default ',').
        encoding: File encoding (default 'utf-8').

    Returns:
        ImportResult with all metadata.

    Raises:
        ValueError: If the CSV cannot be parsed or is empty.
    """
    # Read raw bytes for checksum
    if hasattr(file, "read"):
        raw_bytes = file.read()
        file_name = getattr(file, "name", "upload.csv")
    else:
        with open(file, "rb") as fh:
            raw_bytes = fh.read()
        file_name = str(file)

    checksum = _file_checksum(raw_bytes)

    # Check for duplicate by checksum
    with con.connect() as conn:
        existing = conn.execute(
            text("SELECT table_name FROM _datasets WHERE checksum = :cs"),
            {"cs": checksum},
        ).fetchone()
        if existing:
            raise ValueError(
                f"This file was already imported as table '{existing[0]}'. "
                "Delete the existing dataset first if you want to re-import."
            )

    # Parse CSV
    try:
        df = pd.read_csv(
            BytesIO(raw_bytes),
            delimiter=delimiter,
            encoding=encoding,
            low_memory=False,
        )
    except Exception as exc:
        raise ValueError(f"Cannot parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV file is empty.")

    # Clean up column names: strip whitespace
    df.columns = df.columns.str.strip()

    # Validate
    validation = validate_csv(df)

    # Detect type
    dataset_type = detect_dataset_type(df)

    # Build identifiers
    table_name = make_table_name(file_name)
    display_name = file_name
    col_meta = detect_column_types(df)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    # Write to DB atomically
    columns_json = json.dumps(
        [
            {
                "name": m.name,
                "dtype": m.dtype,
                "sql_type": m.sql_type,
                "nullable": m.nullable,
                "unique_count": m.unique_count,
            }
            for m in col_meta
        ]
    )

    with con.begin() as txn:
        # Create the dynamic table
        df.to_sql(table_name, con=txn, if_exists="fail", index=False, chunksize=5000)

        # Register in _datasets
        txn.execute(
            text(
                """
                INSERT INTO _datasets
                    (table_name, display_name, dataset_type, enrichment_status,
                     row_count, col_count, columns, checksum, uploaded_at, description)
                VALUES
                    (:table_name, :display_name, :dataset_type, 'none',
                     :row_count, :col_count, :columns, :checksum, :uploaded_at, :description)
                """
            ),
            {
                "table_name": table_name,
                "display_name": display_name,
                "dataset_type": dataset_type,
                "row_count": len(df),
                "col_count": len(df.columns),
                "columns": columns_json,
                "checksum": checksum,
                "uploaded_at": uploaded_at,
                "description": description,
            },
        )

    logger.info(
        "Imported '%s' → %s (%d rows, %d cols, type=%s)",
        display_name,
        table_name,
        len(df),
        len(df.columns),
        dataset_type,
    )

    return ImportResult(
        table_name=table_name,
        display_name=display_name,
        dataset_type=dataset_type,
        row_count=len(df),
        col_count=len(df.columns),
        column_meta=col_meta,
        checksum=checksum,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# Enrichment (diabetes only)
# ---------------------------------------------------------------------------

_DIABETES_MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

_ICD9_RANGES: list[tuple[str, str, str]] = [
    ("001", "139", "Infectious and Parasitic Diseases"),
    ("140", "239", "Neoplasms"),
    ("240", "279", "Endocrine, Nutritional, Metabolic"),
    ("280", "289", "Blood Diseases"),
    ("290", "319", "Mental Disorders"),
    ("320", "389", "Nervous System"),
    ("390", "459", "Circulatory System"),
    ("460", "519", "Respiratory System"),
    ("520", "579", "Digestive System"),
    ("580", "629", "Genitourinary System"),
    ("630", "679", "Pregnancy/Childbirth"),
    ("680", "709", "Skin Diseases"),
    ("710", "739", "Musculoskeletal System"),
    ("740", "759", "Congenital Anomalies"),
    ("760", "779", "Perinatal Conditions"),
    ("780", "799", "Symptoms and Signs"),
    ("800", "999", "Injury and Poisoning"),
]


def _classify_icd9(code: str | None) -> str:
    """Return the ICD-9 chapter name for a given code string."""
    if not code or not isinstance(code, str):
        return "Unknown"
    code = code.strip().upper()
    # V-codes and E-codes
    if code.startswith("V"):
        return "V Codes (Supplementary)"
    if code.startswith("E"):
        return "E Codes (External Causes)"
    try:
        num = int(code.split(".")[0])
    except ValueError:
        return "Unknown"
    for lo, hi, label in _ICD9_RANGES:
        if int(lo) <= num <= int(hi):
            return label
    return "Unknown"


def enrich_dataset(
    table_name: str,
    dataset_type: str,
    con: Engine,
) -> EnrichmentResult:
    """
    Run enrichment for a recognized dataset type.

    Currently only 'diabetes' is supported. Creates:
        <table_name>_meds  — unpivoted medication rows
        <table_name>_diag  — ICD-9 diagnosis rows

    Args:
        table_name: Base table name (e.g. 'ds_diabetic_a3f9c1').
        dataset_type: Must be 'diabetes' for enrichment to run.
        con: SQLAlchemy engine for data.db.

    Returns:
        EnrichmentResult with row counts for derived tables.

    Raises:
        ValueError: If dataset_type is not 'diabetes'.
    """
    if dataset_type != "diabetes":
        raise ValueError(f"Enrichment not supported for dataset type '{dataset_type}'.")

    # Mark as pending
    with con.begin() as txn:
        txn.execute(
            text("UPDATE _datasets SET enrichment_status='pending' WHERE table_name=:tn"),
            {"tn": table_name},
        )

    # Load base table
    with con.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM [{table_name}]"), conn)

    # ---- Medications unpivot ----
    med_cols_present = [c for c in _DIABETES_MED_COLS if c in df.columns]
    id_col = "encounter_id" if "encounter_id" in df.columns else df.columns[0]

    if med_cols_present:
        meds_df = df[[id_col] + med_cols_present].copy()
        meds_df = meds_df.melt(
            id_vars=[id_col],
            value_vars=med_cols_present,
            var_name="drug",
            value_name="dosage_change",
        )
        meds_df = meds_df[meds_df["dosage_change"].notna()].copy()
        meds_df.columns = ["encounter_id", "drug", "dosage_change"]
    else:
        meds_df = pd.DataFrame(columns=["encounter_id", "drug", "dosage_change"])

    meds_table = f"{table_name}_meds"
    with con.begin() as txn:
        txn.execute(text(f"DROP TABLE IF EXISTS [{meds_table}]"))
    meds_df.to_sql(meds_table, con=con, if_exists="replace", index=False, chunksize=5000)

    # ---- Diagnoses decode ----
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    diag_rows: list[dict] = []
    for _, row in df.iterrows():
        enc_id = row.get(id_col)
        for pos, dc in enumerate(diag_cols, start=1):
            code = row.get(dc)
            if pd.notna(code):
                diag_rows.append(
                    {
                        "encounter_id": enc_id,
                        "icd9_code": str(code),
                        "diagnosis_position": pos,
                        "icd9_chapter": _classify_icd9(str(code)),
                    }
                )

    diag_df = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame(
        columns=["encounter_id", "icd9_code", "diagnosis_position", "icd9_chapter"]
    )
    diag_table = f"{table_name}_diag"
    with con.begin() as txn:
        txn.execute(text(f"DROP TABLE IF EXISTS [{diag_table}]"))
    diag_df.to_sql(diag_table, con=con, if_exists="replace", index=False, chunksize=5000)

    # Mark as done
    with con.begin() as txn:
        txn.execute(
            text("UPDATE _datasets SET enrichment_status='done' WHERE table_name=:tn"),
            {"tn": table_name},
        )

    logger.info(
        "Enrichment done for %s: meds=%d rows, diag=%d rows",
        table_name,
        len(meds_df),
        len(diag_df),
    )
    return EnrichmentResult(meds_rows=len(meds_df), diag_rows=len(diag_df))


# ---------------------------------------------------------------------------
# Drop dataset
# ---------------------------------------------------------------------------

def drop_dataset(table_name: str, con: Engine) -> None:
    """
    Drop the base table plus any derived enrichment tables, then remove from registry.

    Args:
        table_name: Base table name registered in _datasets.
        con: SQLAlchemy engine for data.db.
    """
    derived = [f"{table_name}_meds", f"{table_name}_diag"]
    with con.begin() as txn:
        for t in [table_name] + derived:
            txn.execute(text(f"DROP TABLE IF EXISTS [{t}]"))
        txn.execute(
            text("DELETE FROM _datasets WHERE table_name = :tn"),
            {"tn": table_name},
        )
    logger.info("Dropped dataset and derived tables for: %s", table_name)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def list_datasets(con: Engine) -> list[dict]:
    """Return all registered datasets as a list of dicts (columns JSON decoded)."""
    with con.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, table_name, display_name, dataset_type, "
                "enrichment_status, row_count, col_count, columns, uploaded_at, "
                "COALESCE(description, '') "
                "FROM _datasets ORDER BY id DESC"
            )
        ).fetchall()
    result = []
    for r in rows:
        columns_val = r[7]
        if isinstance(columns_val, str):
            try:
                columns_val = json.loads(columns_val)
            except (ValueError, TypeError):
                columns_val = []
        result.append({
            "id": r[0],
            "table_name": r[1],
            "display_name": r[2],
            "dataset_type": r[3],
            "enrichment_status": r[4],
            "row_count": r[5],
            "col_count": r[6],
            "columns": columns_val,
            "uploaded_at": r[8],
            "description": r[9],
        })
    return result


def update_description(table_name: str, description: str, con: Engine) -> None:
    """Update the human-readable description for a registered dataset."""
    with con.begin() as txn:
        txn.execute(
            text("UPDATE _datasets SET description = :desc WHERE table_name = :tn"),
            {"desc": description, "tn": table_name},
        )


def get_dataset_meta(table_name: str, con: Engine) -> dict | None:
    """Return full metadata dict for one dataset, or None if not found."""
    with con.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM _datasets WHERE table_name = :tn"),
            {"tn": table_name},
        ).fetchone()
    if row is None:
        return None
    keys = ["id", "table_name", "display_name", "dataset_type", "enrichment_status",
            "row_count", "col_count", "columns", "checksum", "uploaded_at"]
    result = dict(zip(keys, row))
    if result.get("columns"):
        result["columns"] = json.loads(result["columns"])
    return result
