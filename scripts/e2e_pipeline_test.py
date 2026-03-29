"""
e2e_pipeline_test.py — End-to-end test of the generic data pipeline.

Run from the project root:
    python3 scripts/e2e_pipeline_test.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Ensure app/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from sqlalchemy import text

from app.core.pipeline import (
    DB_PATH,
    get_engine,
    import_csv,
    enrich_dataset,
    list_datasets,
    get_dataset_meta,
    drop_dataset,
)
from app.core.type_detector import detect_dataset_type, dataset_type_label
from app.core.registry import get_functions_for, REGISTRY
from app.core.query import fetch_table, fetch_column_stats


def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_tests():
    engine = get_engine(DB_PATH)

    # -------------------------------------------------------
    sep("STEP 1: Bootstrap DB")
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    print(f"Tables in data.db: {[t[0] for t in tables]}")
    assert any("_datasets" in t[0] for t in tables), "_datasets table missing!"
    print("OK: _datasets table exists")

    # -------------------------------------------------------
    sep("STEP 2: Import diabetic_data.csv")
    diabetes_csv = Path("data/raw/diabetic_data.csv")
    assert diabetes_csv.exists(), f"File not found: {diabetes_csv}"
    print(f"File: {diabetes_csv} ({diabetes_csv.stat().st_size / 1024 / 1024:.1f} MB)")

    # Drop existing if re-running
    existing = list_datasets(engine)
    for ds in existing:
        if "diabetic" in ds["display_name"].lower():
            print(f"Dropping existing: {ds['table_name']}")
            drop_dataset(ds["table_name"], engine)

    with open(diabetes_csv, "rb") as f:
        import io
        raw = f.read()
        buf = io.BytesIO(raw)
        buf.name = "diabetic_data.csv"

    print("Importing... (this may take 30-60 seconds for 100k rows)")
    result = import_csv(buf, con=engine)

    print(f"\nImport result:")
    print(f"  table_name:   {result.table_name}")
    print(f"  display_name: {result.display_name}")
    print(f"  dataset_type: {result.dataset_type}  ({dataset_type_label(result.dataset_type)})")
    print(f"  row_count:    {result.row_count:,}")
    print(f"  col_count:    {result.col_count}")
    print(f"  checksum:     {result.checksum}")

    assert result.dataset_type == "diabetes", f"Expected 'diabetes', got '{result.dataset_type}'"
    assert result.row_count > 50000, f"Expected >50k rows, got {result.row_count}"
    print("\nOK: Detected as 'diabetes', row count correct")

    diabetes_table = result.table_name

    # -------------------------------------------------------
    sep("STEP 3: Run Enrichment")
    print("Running enrichment (unpivot meds + decode ICD-9)...")
    enrich_result = enrich_dataset(diabetes_table, "diabetes", engine)
    print(f"\nEnrichment result:")
    print(f"  meds_rows:  {enrich_result.meds_rows:,}")
    print(f"  diag_rows:  {enrich_result.diag_rows:,}")

    assert enrich_result.meds_rows > 0, "No medication rows produced"
    assert enrich_result.diag_rows > 0, "No diagnosis rows produced"

    # Check enrichment tables exist
    meds_table = f"{diabetes_table}_meds"
    diag_table = f"{diabetes_table}_diag"
    with engine.connect() as conn:
        meds_count = conn.execute(text(f"SELECT COUNT(*) FROM [{meds_table}]")).fetchone()[0]
        diag_count = conn.execute(text(f"SELECT COUNT(*) FROM [{diag_table}]")).fetchone()[0]
    print(f"\nVerified in DB:")
    print(f"  {meds_table}: {meds_count:,} rows")
    print(f"  {diag_table}: {diag_count:,} rows")

    # Verify enrichment_status updated
    meta = get_dataset_meta(diabetes_table, engine)
    assert meta["enrichment_status"] == "done", f"Expected 'done', got {meta['enrichment_status']}"
    print("OK: enrichment_status = 'done'")

    # -------------------------------------------------------
    sep("STEP 4: Registry — functions for diabetes")
    fns = get_functions_for("diabetes")
    generic_fns = [f for f in fns if f.scope == "generic"]
    diabetes_fns = [f for f in fns if f.scope == "diabetes"]
    print(f"  Generic functions:  {len(generic_fns)}")
    print(f"  Diabetes functions: {len(diabetes_fns)}")
    for fn in fns:
        marker = "[enrichment req]" if fn.requires_enrichment else ""
        print(f"    {fn.scope:10s}  {fn.id:45s}  {marker}")

    assert len(generic_fns) >= 8
    assert len(diabetes_fns) >= 6
    print("OK: Registry populated correctly")

    # -------------------------------------------------------
    sep("STEP 5: Run sample analytics")
    import pandas as pd
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM [{diabetes_table}] LIMIT 5000"), conn)

    print(f"Loaded {len(df)} rows for analytics test (sample)")
    diabetes_meta = meta.get("columns", [])

    results = {}

    # Generic: describe
    fn = REGISTRY["generic.describe"]
    result_df, fig = fn.fn(df, diabetes_meta)
    print(f"\n  generic.describe → {len(result_df)} columns described, fig={fig}")
    results["describe"] = result_df

    # Generic: correlation
    fn = REGISTRY["generic.correlation"]
    result_df, fig = fn.fn(df, diabetes_meta, method="pearson")
    print(f"  generic.correlation → {result_df.shape}, fig={'yes' if fig else 'no'}")
    results["correlation"] = (result_df, fig)

    # Generic: null_analysis
    fn = REGISTRY["generic.null_analysis"]
    result_df, fig = fn.fn(df, diabetes_meta)
    print(f"  generic.null_analysis → {len(result_df)} columns, fig={'yes' if fig else 'no'}")
    results["null_analysis"] = (result_df, fig)

    # Diabetes: readmission_by_group
    fn = REGISTRY["diabetes.readmission_by_group"]
    result_df, fig = fn.fn(df, diabetes_meta, group_by="age", readmission_binary=True)
    print(f"  diabetes.readmission_by_group → {len(result_df)} rows, fig={'yes' if fig else 'no'}")
    results["readmission_age"] = (result_df, fig)

    # Diabetes: hba1c_vs_readmission
    fn = REGISTRY["diabetes.hba1c_vs_readmission"]
    result_df, fig = fn.fn(df, diabetes_meta, readmission_binary=True)
    print(f"  diabetes.hba1c_vs_readmission → {len(result_df)} rows, fig={'yes' if fig else 'no'}")
    results["hba1c"] = (result_df, fig)

    # Diabetes: los_by_readmission
    fn = REGISTRY["diabetes.los_by_readmission"]
    result_df, fig = fn.fn(df, diabetes_meta)
    print(f"  diabetes.los_by_readmission → {len(result_df)} rows")
    results["los"] = (result_df, fig)

    # Enrichment-required: top_diagnoses (using full df)
    with engine.connect() as conn:
        full_df = pd.read_sql(text(f"SELECT * FROM [{diabetes_table}]"), conn)
    fn = REGISTRY["diabetes.top_diagnoses"]
    result_df, fig = fn.fn(
        full_df, diabetes_meta,
        top_n=10,
        con=engine,
        table_name=diabetes_table,
        enrichment_status="done",
    )
    print(f"  diabetes.top_diagnoses → {len(result_df)} rows, fig={'yes' if fig else 'no'}")
    results["top_diagnoses"] = (result_df, fig)

    # Enrichment-required: medication_frequency
    fn = REGISTRY["diabetes.medication_frequency"]
    result_df, fig = fn.fn(
        full_df, diabetes_meta,
        top_n=15,
        con=engine,
        table_name=diabetes_table,
        enrichment_status="done",
    )
    print(f"  diabetes.medication_frequency → {len(result_df)} rows")
    results["medication_freq"] = (result_df, fig)

    print("\nOK: All analytics ran without errors")

    # -------------------------------------------------------
    sep("STEP 6: Save figures to outputs/figures/")
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, value in results.items():
        if isinstance(value, tuple):
            _, fig = value
        else:
            fig = None
        if fig is not None:
            try:
                out_path = figures_dir / f"pipeline_{name}.html"
                fig.write_html(str(out_path))
                saved.append(out_path)
                print(f"  Saved: {out_path}")
            except Exception as e:
                print(f"  WARN: Could not save {name}: {e}")

    print(f"\nOK: {len(saved)} figures saved as HTML")

    # -------------------------------------------------------
    sep("STEP 7: List all datasets")
    all_ds = list_datasets(engine)
    print(f"\nTotal datasets in DB: {len(all_ds)}")
    for ds in all_ds:
        print(f"  {ds['display_name']:30s}  type={ds['dataset_type']:10s}  "
              f"rows={ds['row_count']:>8,}  enriched={ds['enrichment_status']}")

    # -------------------------------------------------------
    sep("SUMMARY")
    print("All pipeline E2E tests PASSED")
    print(f"  DB path:         {DB_PATH}")
    print(f"  Diabetes table:  {diabetes_table}")
    print(f"  Meds rows:       {enrich_result.meds_rows:,}")
    print(f"  Diag rows:       {enrich_result.diag_rows:,}")
    print(f"  Figures saved:   {len(saved)}")
    print()

    return {
        "diabetes_table": diabetes_table,
        "enrich_meds": enrich_result.meds_rows,
        "enrich_diag": enrich_result.diag_rows,
        "results": results,
    }


if __name__ == "__main__":
    run_tests()
