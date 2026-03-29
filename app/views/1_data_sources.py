"""
1_data_sources.py — Data Sources page.

Single upload path for any CSV:
  1. Upload → preview → type detection badge → Import
  2. Enrichment panel (for recognised dataset types)
  3. Loaded datasets table
  4. Import log
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.core.pipeline import (
    DB_PATH,
    EnrichmentResult,
    ImportResult,
    drop_dataset,
    enrich_dataset,
    get_engine,
    import_csv,
    list_datasets,
    update_description,
)
from app.core.type_detector import dataset_type_icon, dataset_type_label, detect_dataset_type
from app.state import init_state, set_active_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine():
    return get_engine(DB_PATH)


def _ensure_db():
    """Create the bootstrap schema if data.db does not exist yet."""
    db_file = Path(DB_PATH)
    if not db_file.exists():
        from database.create_db import create_registry
        eng = _get_engine()
        create_registry(eng)
    return _get_engine()


def _type_badge(dataset_type: str) -> str:
    icon = dataset_type_icon(dataset_type)
    label = dataset_type_label(dataset_type)
    return f"{icon} {label}"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_sidebar()
engine = _ensure_db()
init_state(engine)

st.title("📂 Data Sources")
st.markdown(
    "Upload any CSV dataset. The pipeline detects the dataset type automatically "
    "and offers specialised enrichment for recognised datasets."
)

# ---------------------------------------------------------------------------
# Import Section
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.subheader("Import Dataset")

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload any CSV. The system auto-detects dataset type from column names.",
    )

    col_delim, col_enc, _ = st.columns([1, 1, 2])
    with col_delim:
        delimiter = st.selectbox("Delimiter", [",", ";", "\\t", "|"], index=0)
        if delimiter == "\\t":
            delimiter = "\t"
    with col_enc:
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1"], index=0)

    if uploaded is not None:
        # Read preview
        try:
            from io import BytesIO
            raw = uploaded.read()
            uploaded.seek(0)
            preview_df = pd.read_csv(
                BytesIO(raw),
                delimiter=delimiter,
                encoding=encoding,
                nrows=5,
                low_memory=False,
            )
            preview_df.columns = preview_df.columns.str.strip()

            st.markdown("**Preview (5 rows)**")
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            # Type detection
            detected_type = detect_dataset_type(preview_df)
            badge = _type_badge(detected_type)
            if detected_type != "generic":
                st.info(
                    f"**Detected: {badge}**  \n"
                    "Specialised analytics will be available after import.",
                    icon="🔍",
                )
            else:
                st.caption(f"Detected type: {badge}")

        except Exception as exc:
            st.error(f"Cannot preview file: {exc}")
            preview_df = None

        # Import button
        if st.button("Import Dataset", type="primary", use_container_width=True):
            with st.spinner("Importing..."):
                try:
                    uploaded.seek(0)
                    result: ImportResult = import_csv(
                        uploaded,
                        con=engine,
                        delimiter=delimiter,
                        encoding=encoding,
                    )
                    msg = (
                        f"Imported **{result.display_name}** → `{result.table_name}` "
                        f"({result.row_count:,} rows, {result.col_count} cols, "
                        f"type: {_type_badge(result.dataset_type)})"
                    )
                    st.success(msg)
                    st.session_state["import_log"].append(msg)

                    # Set as active dataset
                    set_active_dataset(
                        table_name=result.table_name,
                        display_name=result.display_name,
                        dataset_type=result.dataset_type,
                        enrichment_status="none",
                        meta=[
                            {
                                "name": m.name,
                                "dtype": m.dtype,
                                "sql_type": m.sql_type,
                                "nullable": m.nullable,
                                "unique_count": m.unique_count,
                            }
                            for m in result.column_meta
                        ],
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Import failed: {exc}")
                    logger.exception("Import failed")

# ---------------------------------------------------------------------------
# Enrichment Panel
# ---------------------------------------------------------------------------

datasets = list_datasets(engine)
pending_enrichment = [
    d for d in datasets
    if d["dataset_type"] != "generic" and d["enrichment_status"] != "done"
]

if pending_enrichment:
    st.divider()
    for ds in pending_enrichment:
        with st.container(border=True):
            icon = dataset_type_icon(ds["dataset_type"])
            st.markdown(
                f"### {icon} Enrichment Available — {ds['display_name']}"
            )
            if ds["dataset_type"] == "diabetes":
                st.markdown(
                    "Enrichment unpivots 24 medication columns into a long-format table "
                    "and decodes ICD-9 diagnosis codes into chapters. "
                    "Required for **Medication Frequency** and **Top Diagnoses** analytics."
                )
            est = "not yet run" if ds["enrichment_status"] == "none" else "in progress"
            st.caption(f"Status: {est}")

            if st.button(
                "Run Enrichment Pipeline",
                key=f"enrich_{ds['table_name']}",
                type="primary",
            ):
                with st.spinner("Running enrichment (may take 30–60s for large datasets)..."):
                    try:
                        result_e: EnrichmentResult = enrich_dataset(
                            ds["table_name"], ds["dataset_type"], engine
                        )
                        msg = (
                            f"Enrichment complete: {result_e.meds_rows:,} medication rows, "
                            f"{result_e.diag_rows:,} diagnosis rows."
                        )
                        st.success(msg)
                        st.session_state["import_log"].append(msg)
                        # Update enrichment status in active session
                        if st.session_state.get("active_dataset") == ds["table_name"]:
                            st.session_state["active_enrichment_status"] = "done"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Enrichment failed: {exc}")
                        logger.exception("Enrichment failed")

# ---------------------------------------------------------------------------
# Loaded Datasets Table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Loaded Datasets")

if not datasets:
    st.info("No datasets imported yet. Upload a CSV above.")
else:
    for ds in datasets:
        tn = ds["table_name"]
        icon = dataset_type_icon(ds["dataset_type"])
        enrich_badge = (
            "✅ done"
            if ds["enrichment_status"] == "done"
            else ("⏳ pending" if ds["enrichment_status"] == "pending" else "—")
        )

        col_name, col_type, col_rows, col_enrich, col_action = st.columns([3, 2, 1, 1, 1])
        with col_name:
            st.markdown(f"**{ds['display_name']}**  \n`{tn}`")
        with col_type:
            st.markdown(f"{icon} {dataset_type_label(ds['dataset_type'])}")
        with col_rows:
            st.markdown(f"{ds['row_count']:,}" if ds["row_count"] else "—")
        with col_enrich:
            st.markdown(enrich_badge)
        with col_action:
            if st.button("✕", key=f"del_{tn}", help="Delete dataset"):
                drop_dataset(tn, engine)
                st.session_state["import_log"].append(f"Deleted dataset: {ds['display_name']}")
                if st.session_state.get("active_dataset") == tn:
                    st.session_state["active_dataset"] = None
                st.rerun()

        # Description row
        current_desc = ds.get("description") or ""
        edit_key = f"edit_desc_{tn}"
        if not st.session_state.get(edit_key, False):
            # Display mode
            desc_col, btn_col = st.columns([9, 1])
            with desc_col:
                if current_desc:
                    st.caption(current_desc)
                else:
                    st.caption("_No description. Click ✏️ to add one._")
            with btn_col:
                if st.button("✏️", key=f"open_edit_{tn}", help="Edit description"):
                    st.session_state[edit_key] = True
                    st.rerun()
        else:
            # Edit mode
            new_desc = st.text_area(
                "Description",
                value=current_desc,
                key=f"desc_text_{tn}",
                height=80,
                label_visibility="collapsed",
            )
            save_col, cancel_col, _ = st.columns([1, 1, 6])
            with save_col:
                if st.button("Save", key=f"save_desc_{tn}", type="primary"):
                    update_description(tn, new_desc, engine)
                    st.session_state[edit_key] = False
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"cancel_desc_{tn}"):
                    st.session_state[edit_key] = False
                    st.rerun()

        # Set active button
        if st.session_state.get("active_dataset") != tn:
            if st.button("Set as active", key=f"activate_{tn}"):
                meta_raw = ds.get("columns") or []
                if isinstance(meta_raw, str):
                    meta_raw = json.loads(meta_raw)
                set_active_dataset(
                    table_name=tn,
                    display_name=ds["display_name"],
                    dataset_type=ds["dataset_type"],
                    enrichment_status=ds["enrichment_status"],
                    meta=meta_raw,
                )
                st.rerun()
        else:
            st.caption("← active")
        st.divider()

# ---------------------------------------------------------------------------
# Import Log
# ---------------------------------------------------------------------------

with st.expander("Import Log", expanded=False):
    log_entries = st.session_state.get("import_log", [])
    if log_entries:
        st.code("\n\n".join(log_entries[-50:]), language="text")
    else:
        st.caption("No import activity yet.")
