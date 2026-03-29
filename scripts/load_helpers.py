"""
load_helpers.py — Database loading helpers extracted from 02_load.py.

Functions are extracted here so they can be imported by app.core.pipeline
and tests without executing the CLI entry point in 02_load.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

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


def ensure_schema(engine) -> None:
    """Create tables if they don't exist by running database/schema.sql."""
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


def load_admission_types(df: pd.DataFrame, engine) -> None:
    if "admission_type_id" not in df.columns:
        return
    unique_ids = df["admission_type_id"].dropna().unique()
    records = [{"id": int(i), "description": f"Admission Type {int(i)}"} for i in unique_ids]
    records_df = pd.DataFrame(records)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM admission_types"))
    records_df.to_sql("admission_types", engine, if_exists="append", index=False)
    logger.info("Loaded %d admission types.", len(records))


def load_discharge_types(df: pd.DataFrame, engine) -> None:
    if "discharge_disposition_id" not in df.columns:
        return
    unique_ids = df["discharge_disposition_id"].dropna().unique()
    records = [{"id": int(i), "description": f"Discharge Type {int(i)}"} for i in unique_ids]
    records_df = pd.DataFrame(records)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM discharge_types"))
    records_df.to_sql("discharge_types", engine, if_exists="append", index=False)
    logger.info("Loaded %d discharge types.", len(records))


def load_patients(df: pd.DataFrame, engine) -> None:
    if "patient_nbr" not in df.columns:
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
    melt_df = melt_df[melt_df["prescribed"].notna() & (melt_df["prescribed"] != "No")].copy()
    melt_df["change_indicator"] = melt_df["prescribed"].apply(
        lambda x: "Ch" if x in ("Up", "Down") else "No"
    )
    melt_df["encounter_id"] = melt_df["encounter_id"].astype(int)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM medications"))
    melt_df.to_sql("medications", engine, if_exists="append", index=False, chunksize=5000)
    logger.info("Loaded %d medication rows.", len(melt_df))


def _icd9_category(code: str) -> str:
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
    logger.info("Loaded %d ICD-9 codes.", len(lookup))


def load_diagnosis_encounters(df: pd.DataFrame, engine) -> None:
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    present = [c for c in diag_cols if c in df.columns]
    if not present:
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
