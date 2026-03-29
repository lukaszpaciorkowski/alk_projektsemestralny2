"""
03_query.py — CLI wrapper and re-export of query_helpers functions.

All query logic lives in scripts/query_helpers.py to allow shared import
from both this CLI script and the Streamlit app pages.

Usage:
    python scripts/03_query.py --query readmission_by_group --group age_group
    python scripts/03_query.py --query top_diagnoses --top_n 15
    python scripts/03_query.py --query summary_stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path so scripts.query_helpers is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.query_helpers import (
    get_engine,
    hba1c_vs_readmission,
    load_config,
    los_by_readmission,
    medication_counts,
    medications_vs_los,
    readmission_by_group,
    summary_stats,
    top_diagnoses_by_readmission,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "readmission_by_group",
    "los_by_readmission",
    "hba1c_vs_readmission",
    "top_diagnoses_by_readmission",
    "medication_counts",
    "medications_vs_los",
    "summary_stats",
    "get_engine",
    "load_config",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SQL queries against the diabetes database."
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--query",
        choices=[
            "readmission_by_group",
            "los_by_readmission",
            "hba1c_vs_readmission",
            "top_diagnoses",
            "medication_counts",
            "medications_vs_los",
            "summary_stats",
        ],
        default="summary_stats",
        help="Which query to run.",
    )
    parser.add_argument(
        "--group",
        default="age_group",
        help="Group column for readmission_by_group.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Top N results for diagnoses/medications.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary readmission coding (<30/>30 → 1, NO → 0).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config)

    dispatch = {
        "readmission_by_group": lambda: readmission_by_group(
            engine, group_col=args.group, binary=args.binary
        ),
        "los_by_readmission": lambda: los_by_readmission(engine),
        "hba1c_vs_readmission": lambda: hba1c_vs_readmission(engine),
        "top_diagnoses": lambda: top_diagnoses_by_readmission(engine, top_n=args.top_n),
        "medication_counts": lambda: medication_counts(engine, top_n=args.top_n),
        "medications_vs_los": lambda: medications_vs_los(engine),
        "summary_stats": None,
    }

    if args.query == "summary_stats":
        stats = summary_stats(engine)
        for key, val in stats.items():
            print(f"\n--- {key} ---")
            print(val.to_string(index=False))
    else:
        fn = dispatch[args.query]
        df = fn()
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
