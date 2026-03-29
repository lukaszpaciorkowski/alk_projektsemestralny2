"""
run_multi_dataset_test.py — Import and analyze multiple datasets through the pipeline.

Tests:
1. Import heart_disease_cleveland.csv → should be "generic"
2. Import pima_indians_diabetes.csv → should be "generic"
3. Run enrich_dataset() on both
4. Report row counts and saved figures
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from app.core.pipeline import import_csv, enrich_dataset

DATASETS = [
    "data/raw/heart_disease_cleveland.csv",
    "data/raw/pima_indians_diabetes.csv",
]


def main() -> None:
    results = []

    for csv_path in DATASETS:
        logger.info("=" * 60)
        logger.info("Importing: %s", csv_path)
        result = import_csv(csv_path, config_path="config.json")
        logger.info("  dataset_name : %s", result["dataset_name"])
        logger.info("  dataset_type : %s", result["dataset_type"])
        logger.info("  row_count    : %d", result["row_count"])
        logger.info("  col_count    : %d", result["col_count"])
        assert result["dataset_type"] == "generic", (
            f"Expected 'generic' for {csv_path}, got '{result['dataset_type']}'"
        )
        logger.info("  [PASS] Auto-detected as 'generic'")

        logger.info("  Enriching dataset...")
        enrich = enrich_dataset(result["dataset_name"], config_path="config.json")
        logger.info("  Figures saved: %s", enrich.get("figures_saved", []))
        result["figures"] = enrich.get("figures_saved", [])
        results.append(result)

    print("\n" + "=" * 60)
    print("MULTI-DATASET RESULT SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['dataset_name']:<40} type={r['dataset_type']:<10} "
              f"rows={r['row_count']:>6,}  figs={len(r['figures'])}")
    print("=" * 60)
    print("[ALL PASSED]")


if __name__ == "__main__":
    main()
