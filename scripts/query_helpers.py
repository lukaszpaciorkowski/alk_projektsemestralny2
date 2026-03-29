"""
query_helpers.py — Core SQL query functions shared by scripts and the Streamlit app.

All queries use SQLAlchemy text() to avoid bare SQL strings.
This module is imported by 03_query.py, 04_visualize.py, and app pages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    """Load and return pipeline configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_engine(config: dict) -> Engine:
    """Create and return a SQLAlchemy engine from config."""
    db_path = config["database"]["path"]
    url = f"sqlite:///{db_path}"
    return create_engine(url, echo=False)


def readmission_by_group(
    engine: Engine,
    group_col: str = "age_group",
    binary: bool = False,
) -> pd.DataFrame:
    """
    Return readmission rates grouped by a patient or admission attribute.

    Args:
        engine: SQLAlchemy engine.
        group_col: Column to group by (age_group, race, gender, or admissions column).
        binary: If True, collapse '<30' and '>30' to 1, 'NO' to 0.

    Returns:
        DataFrame with group_value, readmission (or readmitted), count, rate.
    """
    patient_cols = {"age_group", "race", "gender"}
    if group_col in patient_cols:
        join_clause = "JOIN patients p ON a.patient_id = p.patient_id"
        select_col = f"p.{group_col}"
    else:
        join_clause = ""
        select_col = f"a.{group_col}"

    if binary:
        readmission_expr = "CASE WHEN a.readmission IN ('<30', '>30') THEN 1 ELSE 0 END"
        sql = text(f"""
            SELECT
                {select_col} AS group_value,
                {readmission_expr} AS readmission,
                COUNT(*) AS count
            FROM admissions a
            {join_clause}
            WHERE {select_col} IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
    else:
        sql = text(f"""
            SELECT
                {select_col} AS group_value,
                a.readmission,
                COUNT(*) AS count
            FROM admissions a
            {join_clause}
            WHERE {select_col} IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    if not df.empty:
        total = df.groupby("group_value")["count"].transform("sum")
        df["rate"] = df["count"] / total
    logger.info("readmission_by_group('%s', binary=%s): %d rows", group_col, binary, len(df))
    return df


def los_by_readmission(engine: Engine, stratify_col: str = "readmission") -> pd.DataFrame:
    """
    Return length-of-stay statistics by readmission class.

    Returns:
        DataFrame with group_value, mean_los, min_los, max_los, n.
    """
    sql = text(f"""
        SELECT
            {stratify_col} AS group_value,
            AVG(time_in_hospital)  AS mean_los,
            MIN(time_in_hospital)  AS min_los,
            MAX(time_in_hospital)  AS max_los,
            COUNT(*)               AS n
        FROM admissions
        WHERE {stratify_col} IS NOT NULL
          AND time_in_hospital IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    logger.info("los_by_readmission: %d rows", len(df))
    return df


def hba1c_vs_readmission(engine: Engine) -> pd.DataFrame:
    """
    Return readmission counts broken down by HbA1c test result.

    Returns:
        DataFrame with hba1c_result, readmission, count, rate.
    """
    sql = text("""
        SELECT
            hba1c_result,
            readmission,
            COUNT(*) AS count
        FROM admissions
        WHERE hba1c_result IS NOT NULL
          AND readmission IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    if not df.empty:
        total = df.groupby("hba1c_result")["count"].transform("sum")
        df["rate"] = df["count"] / total
    logger.info("hba1c_vs_readmission: %d rows", len(df))
    return df


def top_diagnoses_by_readmission(engine: Engine, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top N ICD-9 primary diagnoses with readmission breakdown.

    Returns:
        DataFrame with icd9_code, readmission, count, rate.
    """
    sql = text("""
        SELECT
            de.icd9_code,
            a.readmission,
            COUNT(*) AS count
        FROM diagnosis_encounters de
        JOIN admissions a ON de.encounter_id = a.encounter_id
        WHERE de.icd9_code IS NOT NULL
          AND a.readmission IS NOT NULL
          AND de.diagnosis_position = 1
        GROUP BY 1, 2
        ORDER BY COUNT(*) DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        return df

    top_codes = (
        df.groupby("icd9_code")["count"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    df = df[df["icd9_code"].isin(top_codes)].copy()
    total = df.groupby("icd9_code")["count"].transform("sum")
    df["rate"] = df["count"] / total
    logger.info("top_diagnoses_by_readmission(top_n=%d): %d rows", top_n, len(df))
    return df


def medication_counts(engine: Engine, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top N most frequently prescribed medications.

    Returns:
        DataFrame with drug_name, count.
    """
    sql = text("""
        SELECT
            drug_name,
            COUNT(*) AS count
        FROM medications
        GROUP BY drug_name
        ORDER BY count DESC
        LIMIT :top_n
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"top_n": top_n})
    logger.info("medication_counts(top_n=%d): %d rows", top_n, len(df))
    return df


def medications_vs_los(engine: Engine) -> pd.DataFrame:
    """
    Return aggregated num_medications vs mean length of stay for scatter/regression.

    Returns:
        DataFrame with num_medications, mean_los, count.
    """
    sql = text("""
        SELECT
            num_medications,
            AVG(time_in_hospital) AS mean_los,
            COUNT(*)              AS count
        FROM admissions
        WHERE num_medications IS NOT NULL
          AND time_in_hospital IS NOT NULL
        GROUP BY num_medications
        ORDER BY num_medications
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    logger.info("medications_vs_los: %d rows", len(df))
    return df


def summary_stats(engine: Engine) -> dict[str, Any]:
    """
    Return a dictionary of summary DataFrames for the dataset overview.

    Returns:
        Dict with keys: row_counts, readmission_dist, age_dist, los_stats, hba1c_dist.
    """
    result: dict[str, Any] = {}

    with engine.connect() as conn:
        tables = ["admissions", "patients", "medications", "diagnosis_encounters"]
        counts: dict[str, int] = {}
        for t in tables:
            row = conn.execute(text(f"SELECT COUNT(*) AS n FROM {t}")).fetchone()
            counts[t] = int(row[0]) if row else 0
        result["row_counts"] = pd.DataFrame(
            list(counts.items()), columns=["table", "rows"]
        )

        result["readmission_dist"] = pd.read_sql(
            text("SELECT readmission, COUNT(*) AS count FROM admissions GROUP BY readmission ORDER BY count DESC"),
            conn,
        )

        result["age_dist"] = pd.read_sql(
            text("SELECT age_group, COUNT(*) AS count FROM patients GROUP BY age_group ORDER BY age_group"),
            conn,
        )

        result["los_stats"] = pd.read_sql(
            text("""
                SELECT
                    MIN(time_in_hospital) AS min_los,
                    MAX(time_in_hospital) AS max_los,
                    AVG(time_in_hospital) AS mean_los,
                    COUNT(*) AS n
                FROM admissions
            """),
            conn,
        )

        result["hba1c_dist"] = pd.read_sql(
            text("""
                SELECT hba1c_result, COUNT(*) AS count
                FROM admissions
                WHERE hba1c_result IS NOT NULL
                GROUP BY hba1c_result
                ORDER BY count DESC
            """),
            conn,
        )

    logger.info("summary_stats: returned %d sections.", len(result))
    return result
