"""
ingest_helpers.py — Core validation/transformation functions for the ingestion pipeline.

These functions are extracted from 01_ingest.py into a standalone module so they
can be imported by tests (tests/test_validation.py) and other scripts without
executing the CLI entry point.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

AGE_GROUP_PATTERN: re.Pattern = re.compile(r"^\[\d+-\d+\)$")


def replace_question_marks(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' sentinel values with NaN."""
    before = df.isnull().sum().sum()
    df = df.replace("?", np.nan)
    after = df.isnull().sum().sum()
    logger.info("Replaced '?' with NaN: %d new nulls", after - before)
    return df


def drop_high_null_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop columns where null percentage exceeds threshold.

    Args:
        df: Input DataFrame.
        threshold: Float in [0, 1]. Columns with null fraction > threshold are dropped.

    Returns:
        New DataFrame with high-null columns removed.
    """
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        logger.warning(
            "Dropping %d columns with null > %.0f%%: %s",
            len(cols_to_drop),
            threshold * 100,
            cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)
    else:
        logger.info("No columns exceeded null threshold of %.0f%%.", threshold * 100)
    return df


def remove_outliers_zscore(df: pd.DataFrame, zscore_threshold: float) -> pd.DataFrame:
    """
    Remove rows where any numeric column has absolute Z-score > zscore_threshold.

    Args:
        df: Input DataFrame.
        zscore_threshold: Z-score cutoff (e.g. 3.0).

    Returns:
        New DataFrame with outlier rows removed.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.info("No numeric columns found for outlier removal.")
        return df

    before = len(df)
    z_scores = np.abs(
        stats.zscore(df[numeric_cols].fillna(df[numeric_cols].median()), nan_policy="omit")
    )
    mask = (z_scores < zscore_threshold).all(axis=1)
    df = df[mask].copy()
    removed = before - len(df)
    logger.info(
        "Outlier removal (z>%.1f): removed %d rows, %d remaining.",
        zscore_threshold,
        removed,
        len(df),
    )
    return df


def validate_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows to keep only valid age group format like '[30-40)'.

    Args:
        df: DataFrame expected to contain an 'age' column.

    Returns:
        Filtered DataFrame. If 'age' column absent, returns df unchanged.
    """
    if "age" not in df.columns:
        logger.warning("Column 'age' not found, skipping age group validation.")
        return df

    before = len(df)
    valid_mask = df["age"].apply(
        lambda x: bool(AGE_GROUP_PATTERN.match(str(x))) if pd.notna(x) else False
    )
    df = df[valid_mask].copy()
    removed = before - len(df)
    if removed:
        logger.warning("Age group validation: removed %d invalid rows.", removed)
    else:
        logger.info("Age group validation: all values valid.")
    return df


def standardize_readmission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise readmission column values to consistent uppercase form.

    Mapping: 'No' / 'no' → 'NO'. '<30' and '>30' preserved unchanged.

    Args:
        df: DataFrame expected to contain a 'readmitted' column.

    Returns:
        New DataFrame with normalised 'readmitted' values.
    """
    if "readmitted" not in df.columns:
        logger.warning("Column 'readmitted' not found, skipping.")
        return df

    mapping = {
        "<30": "<30",
        ">30": ">30",
        "NO": "NO",
        "No": "NO",
        "no": "NO",
    }
    df = df.copy()
    df["readmitted"] = df["readmitted"].map(mapping).fillna(df["readmitted"])
    logger.info("Readmission values normalized.")
    return df


def add_age_group_normalized(
    df: pd.DataFrame,
    age_bins: list,
    age_labels: list,
) -> pd.DataFrame:
    """
    Add an 'age_group' column by binning the midpoint of age bracket strings.

    The 'age' column is expected to contain strings like '[30-40)'.
    Midpoints are computed and then pd.cut is applied with the provided bins/labels.

    Args:
        df: DataFrame with an 'age' column in bracket format.
        age_bins: List of bin edges, e.g. [0, 30, 50, 70, 100].
        age_labels: List of labels, e.g. ['0-30', '30-50', '50-70', '70+'].

    Returns:
        New DataFrame with added 'age_numeric' and 'age_group' columns.
    """
    if "age" not in df.columns:
        return df

    def extract_midpoint(age_str: str) -> float | None:
        match = re.match(r"\[(\d+)-(\d+)\)", str(age_str))
        if match:
            return (int(match.group(1)) + int(match.group(2))) / 2.0
        return None

    df = df.copy()
    df["age_numeric"] = df["age"].apply(extract_midpoint)
    df["age_group"] = pd.cut(
        df["age_numeric"],
        bins=age_bins,
        labels=age_labels,
        right=False,
    ).astype(str)
    logger.info("Added 'age_group' column with bins %s.", age_bins)
    return df
