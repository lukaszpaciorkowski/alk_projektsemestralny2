"""
run_pipeline_test.py — End-to-end pipeline test script.

Tests import_csv() and enrich_dataset() from app.core.pipeline.
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


def main() -> None:
    # ── Step 1: Import diabetes dataset ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Importing diabetes dataset")
    result = import_csv("data/raw/diabetic_data.csv", config_path="config.json")
    logger.info("  dataset_name : %s", result["dataset_name"])
    logger.info("  dataset_type : %s", result["dataset_type"])
    logger.info("  row_count    : %d", result["row_count"])
    logger.info("  col_count    : %d", result["col_count"])
    logger.info("  table_name   : %s", result["table_name"])

    assert result["dataset_type"] == "diabetes", (
        f"Expected 'diabetes', got '{result['dataset_type']}'"
    )
    assert result["row_count"] > 0, "Expected rows > 0"
    logger.info("  [PASS] Auto-detected as 'diabetes'")
    logger.info("  [PASS] Row count: %d", result["row_count"])

    # ── Step 2: Enrich diabetes dataset ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Enriching diabetes dataset")
    enrich_result = enrich_dataset(result["dataset_name"], config_path="config.json")
    logger.info("  Figures saved: %s", enrich_result.get("figures_saved", []))
    logger.info("  Readmission dist (first 3): %s", enrich_result.get("readmission_dist", [])[:3])

    logger.info("=" * 60)
    logger.info("Pipeline test PASSED")
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"  dataset_name : {result['dataset_name']}")
    print(f"  dataset_type : {result['dataset_type']}")
    print(f"  row_count    : {result['row_count']:,}")
    print(f"  col_count    : {result['col_count']}")
    print(f"  table_name   : {result['table_name']}")
    print(f"  figures      : {len(enrich_result.get('figures_saved', []))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
