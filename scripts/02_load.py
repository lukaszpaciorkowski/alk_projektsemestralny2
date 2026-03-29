"""
02_load.py — Load cleaned CSV data into the SQLite database (3NF schema).

Steps:
  1. Read cleaned CSV from data.processed_csv
  2. Load admission_types and discharge_types lookups
  3. Deduplicate and load patients table
  4. Load admissions (core encounters)
  5. Unpivot 24 medication columns → medications table (skip 'No' values)
  6. Parse diag_1/diag_2/diag_3 → diagnosis_encounters table

Usage:
    python scripts/02_load.py [--config config.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MEDICATION_COLS: list[str] = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def load_config(config_path: str) -> dict:
    """Load and return pipeline configuration."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_engine(config: dict):
    """Create SQLAlchemy engine from config."""
    db_path = config["database"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    logger.info("Connecting to database: %s", url)
    return create_engine(url, echo=False)


def ensure_schema(engine) -> None:
    """Create tables if they don't exist by running schema.sql."""
    schema_path = Path(__file__).parent.parent / "database" / "schema.sql"
    if schema_path.exists():
        sql_text = schema_path.read_text(encoding="utf-8")
        statements = [s.strip() for s in sql_text.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
        logger.info("Schema verified/created.")
    else:
        logger.warning("schema.sql not found at %s, assuming tables exist.", schema_path)


def load_admission_types(df: pd.DataFrame, engine) -> dict[str, int]:
    """Extract unique admission types and insert into admission_types table."""
    if "admission_type_id" not in df.columns:
        return {}

    unique_ids = df["admission_type_id"].dropna().unique()
    records = [{"id": int(i), "description": f"Admission Type {int(i)}"} for i in unique_ids]
    records_df = pd.DataFrame(records)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM admission_types"))
    records_df.to_sql("admission_types", engine, if_exists="append", index=False)
    logger.info("Loaded %d admission types.", len(records))
    return {r["id"]: r["description"] for r in records}


def load_discharge_types(df: pd.DataFrame, engine) -> dict[str, int]:
    """Extract unique discharge types and insert into discharge_types table."""
    if "discharge_disposition_id" not in df.columns:
        return {}

    unique_ids = df["discharge_disposition_id"].dropna().unique()
    records = [{"id": int(i), "description": f"Discharge Type {int(i)}"} for i in unique_ids]
    records_df = pd.DataFrame(records)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM discharge_types"))
    records_df.to_sql("discharge_types", engine, if_exists="append", index=False)
    logger.info("Loaded %d discharge types.", len(records))
    return {}


def load_patients(df: pd.DataFrame, engine) -> None:
    """Deduplicate by patient_nbr and load into patients table."""
    patient_col = "patient_nbr" if "patient_nbr" in df.columns else None
    if patient_col is None:
        logger.warning("Column 'patient_nbr' not found; skipping patients load.")
        return

    patients_df = (
        df[["patient_nbr", "race", "gender", "age_group"]]
        .drop_duplicates(subset=["patient_nbr"])
        .copy()
    )
    patients_df = patients_df.rename(columns={"patient_nbr": "patient_id"})
    patients_df["patient_id"] = patients_df["patient_id"].astype(int)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM patients"))
    patients_df.to_sql("patients", engine, if_exists="append", index=False)
    logger.info("Loaded %d unique patients.", len(patients_df))


def load_admissions(df: pd.DataFrame, engine) -> None:
    """Load core admission/encounter rows."""
    col_map = {
        "encounter_id": "encounter_id",
        "patient_nbr": "patient_id",
        "admission_type_id": "admission_type_id",
        "discharge_disposition_id": "discharge_type_id",
        "time_in_hospital": "time_in_hospital",
        "num_lab_procedures": "num_lab_procedures",
        "num_procedures": "num_procedures",
        "num_medications": "num_medications",
        "number_diagnoses": "num_diagnoses",
        "A1Cresult": "hba1c_result",
        "change": "change_medications",
        "diabetesMed": "diabetes_medication",
        "readmitted": "readmission",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    admissions_df = df[list(available.keys())].rename(columns=available).copy()

    for col in ["encounter_id", "patient_id"]:
        if col in admissions_df.columns:
            admissions_df[col] = admissions_df[col].astype(int)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM admissions"))
    admissions_df.to_sql("admissions", engine, if_exists="append", index=False, chunksize=5000)
    logger.info("Loaded %d admissions.", len(admissions_df))


def unpivot_medications(df: pd.DataFrame, engine) -> None:
    """Unpivot 24 medication columns into long-form medications table."""
    present_med_cols = [c for c in MEDICATION_COLS if c in df.columns]
    if not present_med_cols:
        logger.warning("No medication columns found in DataFrame.")
        return

    id_vars = ["encounter_id"]
    melt_df = df[id_vars + present_med_cols].melt(
        id_vars=id_vars,
        value_vars=present_med_cols,
        var_name="drug_name",
        value_name="prescribed",
    )

    # Only keep rows where a drug was actually prescribed (not 'No')
    melt_df = melt_df[melt_df["prescribed"].notna() & (melt_df["prescribed"] != "No")].copy()

    # Derive change indicator: 'Ch' for Up/Down, 'No' for Steady
    melt_df["change_indicator"] = melt_df["prescribed"].apply(
        lambda x: "Ch" if x in ("Up", "Down") else "No"
    )
    melt_df["encounter_id"] = melt_df["encounter_id"].astype(int)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM medications"))
    melt_df.to_sql("medications", engine, if_exists="append", index=False, chunksize=5000)
    logger.info("Loaded %d medication rows (long format).", len(melt_df))


def _icd9_category(code: str) -> str:
    """Return a broad ICD-9 category description for a code string."""
    c = code.strip().upper()
    if c.startswith("E"):
        return "External causes of injury"
    if c.startswith("V"):
        return "Supplementary classification"
    try:
        n = float(c)
    except ValueError:
        return "Other / unclassified"
    if n < 140:
        return "Infectious and parasitic diseases"
    if n < 240:
        return "Neoplasms"
    if n < 280:
        return "Endocrine, nutritional, metabolic, immunity"
    if n < 290:
        return "Diseases of blood"
    if n < 320:
        return "Mental disorders"
    if n < 390:
        return "Nervous system and sense organs"
    if n < 460:
        return "Circulatory system"
    if n < 520:
        return "Respiratory system"
    if n < 580:
        return "Digestive system"
    if n < 630:
        return "Genitourinary system"
    if n < 680:
        return "Complications of pregnancy / childbirth"
    if n < 710:
        return "Skin and subcutaneous tissue"
    if n < 740:
        return "Musculoskeletal and connective tissue"
    if n < 760:
        return "Congenital anomalies"
    if n < 780:
        return "Perinatal conditions"
    if n < 800:
        return "Symptoms, signs, ill-defined conditions"
    return "Injury and poisoning"


def load_diagnoses_lookup(df: pd.DataFrame, engine) -> None:
    """Populate diagnoses_lookup with all unique ICD-9 codes found in the data."""
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    codes = (
        pd.concat([df[c] for c in diag_cols])
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
    lookup = pd.DataFrame({
        "icd9_code": codes,
        "description": [f"ICD-9 {c}" for c in codes],
        "category": [_icd9_category(c) for c in codes],
    })
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnoses_lookup"))
    lookup.to_sql("diagnoses_lookup", engine, if_exists="append", index=False)
    logger.info("Loaded %d ICD-9 codes into diagnoses_lookup.", len(lookup))


def load_diagnosis_encounters(df: pd.DataFrame, engine) -> None:
    """Parse diag_1, diag_2, diag_3 and load into diagnosis_encounters."""
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    present = [c for c in diag_cols if c in df.columns]
    if not present:
        logger.warning("No diagnosis columns found; skipping diagnosis_encounters.")
        return

    frames: list[pd.DataFrame] = []
    for pos, col in enumerate(present, start=1):
        sub = df[["encounter_id", col]].copy()
        sub = sub.rename(columns={col: "icd9_code"})
        sub = sub[sub["icd9_code"].notna()].copy()
        sub["diagnosis_position"] = pos
        sub["icd9_code"] = sub["icd9_code"].astype(str)
        frames.append(sub)

    if not frames:
        return

    diag_df = pd.concat(frames, ignore_index=True)
    diag_df["encounter_id"] = diag_df["encounter_id"].astype(int)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnosis_encounters"))
    diag_df.to_sql("diagnosis_encounters", engine, if_exists="append", index=False, chunksize=5000)
    logger.info("Loaded %d diagnosis encounter rows.", len(diag_df))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load cleaned CSV into the SQLite diabetes database."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_csv = config["data"]["processed_csv"]

    path = Path(processed_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found at '{processed_csv}'.\n"
            "Run scripts/01_ingest.py first."
        )

    logger.info("Reading processed CSV: %s", processed_csv)
    df = pd.read_csv(processed_csv, low_memory=False)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    engine = get_engine(config)
    ensure_schema(engine)

    load_admission_types(df, engine)
    load_discharge_types(df, engine)
    load_patients(df, engine)
    load_admissions(df, engine)
    unpivot_medications(df, engine)
    load_diagnoses_lookup(df, engine)
    load_diagnosis_encounters(df, engine)

    logger.info("All data loaded successfully.")


if __name__ == "__main__":
    main()
