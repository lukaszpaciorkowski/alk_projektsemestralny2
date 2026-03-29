"""
01_ingest.py — Ingest and validate the Diabetes 130-US Hospitals CSV dataset.

Steps:
  1. Read raw CSV (diabetic_data.csv)
  2. Validate null percentages — drop columns exceeding null_threshold
  3. Remove outliers using Z-score on numeric columns
  4. Validate domain values (age_group format)
  5. Normalise readmission labels
  6. Add age_group column from binning
  7. Save cleaned DataFrame to processed CSV

Usage:
    python scripts/01_ingest.py [--config config.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_helpers import (
    add_age_group_normalized,
    drop_high_null_columns,
    remove_outliers_zscore,
    replace_question_marks,
    standardize_readmission,
    validate_age_groups,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and return pipeline configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_raw_csv(raw_csv: str) -> pd.DataFrame:
    """Read raw CSV file. Raises FileNotFoundError with helpful message if missing."""
    path = Path(raw_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_csv}'.\n"
            "Please download the dataset from Kaggle:\n"
            "  kaggle datasets download -d brandao/diabetes --unzip -p data/raw/\n"
            "Or visit: https://www.kaggle.com/datasets/brandao/diabetes\n"
            "The file should be named 'diabetic_data.csv'."
        )
    logger.info("Reading raw CSV from: %s", raw_csv)
    df = pd.read_csv(raw_csv, low_memory=False)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def log_validation_report(original_rows: int, original_cols: int, df: pd.DataFrame) -> None:
    """Log a summary validation report."""
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("  Original : %d rows × %d columns", original_rows, original_cols)
    logger.info("  Final    : %d rows × %d columns", len(df), len(df.columns))
    logger.info(
        "  Rows dropped : %d (%.1f%%)",
        original_rows - len(df),
        (original_rows - len(df)) / max(original_rows, 1) * 100,
    )
    logger.info("  Cols dropped : %d", original_cols - len(df.columns))
    logger.info("  Remaining nulls : %d", df.isnull().sum().sum())
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest and validate the Diabetes 130-US Hospitals dataset."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = config["pipeline"]
    data_cfg = config["data"]

    df = read_raw_csv(data_cfg["raw_csv"])
    original_rows = len(df)
    original_cols = len(df.columns)

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

    log_validation_report(original_rows, original_cols, df)

    output_path = Path(data_cfg["processed_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset saved to: %s", output_path)


if __name__ == "__main__":
    main()
