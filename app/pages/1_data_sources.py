"""
1_data_sources.py — Data Sources page for the Streamlit app.

Features:
- CSV file uploader
- Import pipeline trigger (ingest + load)
- Active data sources table (DB row counts)
- Import log expander
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.state import init_state

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_engine(config: dict):
    db_path = config["database"]["path"]
    if not Path(db_path).exists():
        return None
    return create_engine(f"sqlite:///{db_path}", echo=False)


def _get_table_counts(engine) -> pd.DataFrame:
    """Query row counts from all tables."""
    tables = ["admissions", "patients", "medications", "diagnosis_encounters",
              "admission_types", "discharge_types", "diagnoses_lookup"]
    rows = []
    with engine.connect() as conn:
        for t in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).fetchone()
                rows.append({"Table": t, "Rows": int(result[0])})
            except Exception:
                rows.append({"Table": t, "Rows": "N/A"})
    return pd.DataFrame(rows)


def _run_pipeline_script(script: str, config_path: str) -> tuple[bool, str]:
    """Run a pipeline script and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, script, "--config", config_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Pipeline script timed out after 300s."
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

render_sidebar()
init_state()

st.title("📂 Data Sources")
st.markdown(
    "Upload the raw CSV dataset or trigger the full import pipeline. "
    "The pipeline validates, cleans, and loads data into the SQLite database."
)

try:
    config = _load_config()
except FileNotFoundError:
    st.error("config.json not found. Ensure you are running from the project root.")
    st.stop()

raw_csv_path = config["data"]["raw_csv"]

# ---- File Uploader ----
st.subheader("Upload Dataset")
uploaded = st.file_uploader(
    "Upload diabetic_data.csv",
    type=["csv"],
    help="Download from https://www.kaggle.com/datasets/brandao/diabetes",
)

if uploaded is not None:
    dest_path = Path(raw_csv_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"File saved to `{raw_csv_path}` ({uploaded.size / 1024:.1f} KB)")
    st.session_state["import_log"].append(f"Uploaded: {uploaded.name} → {raw_csv_path}")

# ---- CSV Status ----
raw_exists = Path(raw_csv_path).exists()
if raw_exists:
    st.info(f"Raw CSV found at `{raw_csv_path}`")
else:
    st.warning(
        f"Raw CSV not found at `{raw_csv_path}`. "
        "Please upload the file above or download it from Kaggle."
    )

st.divider()

# ---- Import Pipeline ----
st.subheader("Run Import Pipeline")

col1, col2, col3 = st.columns(3)
with col1:
    run_ingest = st.button("1. Ingest & Validate", disabled=not raw_exists, use_container_width=True)
with col2:
    run_load = st.button(
        "2. Load to Database",
        disabled=not Path(config["data"]["processed_csv"]).exists(),
        use_container_width=True,
    )
with col3:
    run_all = st.button(
        "Run Full Pipeline",
        disabled=not raw_exists,
        type="primary",
        use_container_width=True,
    )

log_container = st.empty()

if run_ingest or run_all:
    with st.spinner("Running ingestion and validation..."):
        ok, output = _run_pipeline_script("scripts/01_ingest.py", CONFIG_PATH)
    st.session_state["import_log"].append(f"=== 01_ingest.py ===\n{output}")
    if ok:
        st.success("Ingestion complete.")
    else:
        st.error("Ingestion failed. See log below.")

if run_load or run_all:
    with st.spinner("Loading data into database..."):
        ok, output = _run_pipeline_script("scripts/02_load.py", CONFIG_PATH)
    st.session_state["import_log"].append(f"=== 02_load.py ===\n{output}")
    if ok:
        st.success("Database load complete.")
        st.session_state["db_loaded"] = True
    else:
        st.error("Load failed. See log below.")

# ---- Active Data Sources ----
st.divider()
st.subheader("Active Data Sources")

engine = _get_engine(config)
if engine is not None:
    counts_df = _get_table_counts(engine)
    st.dataframe(counts_df, use_container_width=True, hide_index=True)
else:
    st.info("Database not yet created. Run the import pipeline above.")

# ---- Import Log ----
st.divider()
with st.expander("Import Log", expanded=False):
    log_entries = st.session_state.get("import_log", [])
    if log_entries:
        st.code("\n\n".join(log_entries), language="text")
    else:
        st.caption("No import activity yet.")
