"""
type_detector.py — Dataset type detection via column signature matching.

Runs once at upload time before the table is created.
"""

from __future__ import annotations

import pandas as pd

# Each entry: required columns (ALL must be present) to trigger detection.
DATASET_TYPE_SIGNATURES: dict[str, dict] = {
    "diabetes": {
        "required": {
            "patient_nbr",
            "readmitted",
            "metformin",
            "time_in_hospital",
            "num_medications",
            "diag_1",
        },
        "optional": {
            "race",
            "gender",
            "age",
            "admission_type_id",
            "number_diagnoses",
            "insulin",
            "glipizide",
        },
    },
}

# Human-readable labels for each detected type
DATASET_TYPE_LABELS: dict[str, str] = {
    "diabetes": "Diabetes 130-US Hospitals",
    "generic": "Generic",
}

# Emoji badges per type
DATASET_TYPE_ICONS: dict[str, str] = {
    "diabetes": "🧬",
    "generic": "📄",
}


def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Return the dataset type string based on column signature matching.

    Args:
        df: DataFrame whose columns are inspected.

    Returns:
        Type string such as 'diabetes', or 'generic' if no match found.
    """
    if df.empty or len(df.columns) == 0:
        return "generic"
    cols = set(df.columns.str.lower().str.strip())
    for dtype, sig in DATASET_TYPE_SIGNATURES.items():
        required = sig["required"]
        if required.issubset(cols):
            return dtype
    return "generic"


def dataset_type_label(dataset_type: str) -> str:
    """Return human-readable label for a dataset type."""
    return DATASET_TYPE_LABELS.get(dataset_type, dataset_type.capitalize())


def dataset_type_icon(dataset_type: str) -> str:
    """Return emoji icon for a dataset type."""
    return DATASET_TYPE_ICONS.get(dataset_type, "📄")
