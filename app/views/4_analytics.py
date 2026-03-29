"""
4_analytics.py — Registry-driven Analytics page.

Function selection uses st.radio (persistent across reruns), not st.button.
Two radio groups (Generic / Specialised) with mutual-exclusion via pre-render
session-state manipulation so only one group shows a selection at a time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.core.pipeline import (
    DB_PATH,
    EnrichmentRequiredError,
    get_engine,
    list_datasets,
)
from app.core.registry import REGISTRY, AnalyticsFunction, get_functions_for
from app.core.type_detector import dataset_type_icon, dataset_type_label
from app.state import add_to_report, init_state, set_active_dataset


def _get_engine():
    return get_engine(DB_PATH)


def _col_names_by_dtype(meta: list[dict], dtype_filter: str | None) -> list[str]:
    if not dtype_filter:
        return [c["name"] for c in meta]
    if dtype_filter == "numeric":
        return [c["name"] for c in meta if "int" in c.get("dtype", "") or "float" in c.get("dtype", "")]
    if dtype_filter == "categorical":
        return [c["name"] for c in meta if "int" not in c.get("dtype", "") and "float" not in c.get("dtype", "")]
    return [c["name"] for c in meta]


def _render_param_widget(param, meta: list[dict]) -> Any:
    label = param.label or param.name.replace("_", " ").capitalize()
    if param.widget == "select":
        return st.selectbox(label, param.options,
                            index=param.options.index(param.default) if param.default in param.options else 0)
    if param.widget == "bool":
        return st.checkbox(label, value=bool(param.default))
    if param.widget == "int":
        return st.number_input(label, value=int(param.default), min_value=1, max_value=500, step=1)
    if param.widget in ("select_column", "multiselect_column"):
        cols = _col_names_by_dtype(meta, param.dtype_filter)
        if not cols:
            st.caption(f"No {param.dtype_filter or ''} columns available.")
            return param.default
        default_val = param.default if param.default in cols else (cols[0] if cols else "")
        if param.widget == "multiselect_column":
            return st.multiselect(label, cols, default=[default_val] if default_val else [])
        return st.selectbox(label, cols, index=cols.index(default_val) if default_val in cols else 0)
    return param.default


def _load_table(engine, table_name: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(sql_text(f"SELECT * FROM [{table_name}]"), conn)


def _execute_fn(fn: AnalyticsFunction, engine, table_name: str, meta: list[dict],
                enrichment_status: str, params: dict) -> dict:
    """Load table, run fn, return result dict for session state storage."""
    df = _load_table(engine, table_name)
    call_params = dict(params)
    if fn.requires_enrichment:
        call_params["con"] = engine
        call_params["table_name"] = table_name
        call_params["enrichment_status"] = enrichment_status
    result_df, fig = fn.fn(df, meta, **call_params)
    return {"fn_id": fn.id, "fn_label": fn.label, "result_df": result_df, "fig": fig}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_sidebar()

try:
    engine = _get_engine()
except Exception:
    engine = None

init_state(engine)

st.title("📊 Analytics")

if engine is None:
    st.warning("Database not found. Import a dataset on the **Data Sources** page first.")
    st.stop()

datasets = list_datasets(engine)
if not datasets:
    st.warning("No datasets imported yet. Go to **Data Sources** to upload a CSV.")
    st.stop()

# ── Dataset Selector ──────────────────────────────────────────────────────────
ds_options = {
    f"{dataset_type_icon(d['dataset_type'])} {d['display_name']} "
    f"({d['row_count']:,} rows)": d
    for d in datasets
}

active_name = st.session_state.get("active_dataset_name", "")
default_idx = 0
for idx, lbl in enumerate(ds_options):
    if active_name and active_name in lbl:
        default_idx = idx
        break

selected_label = st.selectbox("Dataset", list(ds_options.keys()), index=default_idx)
selected_ds = ds_options[selected_label]
table_name = selected_ds["table_name"]
dataset_type = selected_ds["dataset_type"]
enrichment_status = selected_ds.get("enrichment_status", "none")

meta_raw = selected_ds.get("columns") or []
if isinstance(meta_raw, str):
    meta_raw = json.loads(meta_raw)

if st.session_state.get("active_dataset") != table_name:
    set_active_dataset(
        table_name=table_name,
        display_name=selected_ds["display_name"],
        dataset_type=dataset_type,
        enrichment_status=enrichment_status,
        meta=meta_raw,
    )
    # Reset all analytics state when dataset changes
    for key in ("_gen_radio", "_spec_radio", "_prev_gen", "_prev_spec",
                "selected_fn", "analytics_result", "analytics_error"):
        st.session_state.pop(key, None)
    st.rerun()

# ── Build function lists ───────────────────────────────────────────────────────
functions = get_functions_for(dataset_type)
generic_fns = [f for f in functions if f.scope == "generic"]
specialized_fns = [f for f in functions if f.scope != "generic"]

needs_enrichment = {f.id for f in specialized_fns if f.requires_enrichment}
disabled_ids = needs_enrichment if enrichment_status != "done" else set()
active_spec_fns = [f for f in specialized_fns if f.id not in disabled_ids]

gen_labels = [f.label for f in generic_fns]
spec_labels = [f.label for f in active_spec_fns]
gen_label_to_fn = {f.label: f for f in generic_fns}
spec_label_to_fn = {f.label: f for f in active_spec_fns}

# ── Mutual-exclusion: detect which radio changed, update selected_fn ──────────
# Streamlit has already written the new radio value into session_state before
# this code runs. We compare against what we recorded on the previous pass.

curr_gen = st.session_state.get("_gen_radio")    # current value of gen radio
curr_spec = st.session_state.get("_spec_radio")  # current value of spec radio
prev_gen = st.session_state.get("_prev_gen")
prev_spec = st.session_state.get("_prev_spec")

if curr_gen != prev_gen and curr_gen is not None:
    # User just picked something in the Generic group
    fn = gen_label_to_fn.get(curr_gen)
    if fn:
        st.session_state["selected_fn"] = fn.id
        st.session_state["analytics_result"] = None
        st.session_state["analytics_error"] = None
        # Deselect spec group BEFORE it renders
        st.session_state["_spec_radio"] = None
        st.session_state["_prev_spec"] = None
    st.session_state["_prev_gen"] = curr_gen

    # Parameterless → run immediately (result is in session_state for this render)
    if fn and not fn.params:
        try:
            st.session_state["analytics_result"] = _execute_fn(
                fn, engine, table_name, meta_raw, enrichment_status, {}
            )
        except EnrichmentRequiredError as exc:
            st.session_state["analytics_error"] = ("enrichment", str(exc))
        except Exception as exc:
            st.session_state["analytics_error"] = ("error", str(exc))

elif curr_spec != prev_spec and curr_spec is not None:
    # User just picked something in the Specialised group
    fn = spec_label_to_fn.get(curr_spec)
    if fn:
        st.session_state["selected_fn"] = fn.id
        st.session_state["analytics_result"] = None
        st.session_state["analytics_error"] = None
        # Deselect gen group BEFORE it renders
        st.session_state["_gen_radio"] = None
        st.session_state["_prev_gen"] = None
    st.session_state["_prev_spec"] = curr_spec

    if fn and not fn.params:
        try:
            st.session_state["analytics_result"] = _execute_fn(
                fn, engine, table_name, meta_raw, enrichment_status, {}
            )
        except EnrichmentRequiredError as exc:
            st.session_state["analytics_error"] = ("enrichment", str(exc))
        except Exception as exc:
            st.session_state["analytics_error"] = ("error", str(exc))

# ── Render radio groups ────────────────────────────────────────────────────────
st.divider()
st.subheader("Available Analyses")

if disabled_ids:
    missing = ", ".join(REGISTRY[fid].label for fid in disabled_ids if fid in REGISTRY)
    st.warning(f"**{missing}** require enrichment. Run enrichment on the **Data Sources** page.")

st.markdown("**GENERIC — works on any dataset**")
st.radio(
    "generic_analyses",
    gen_labels,
    key="_gen_radio",
    index=None,
    label_visibility="collapsed",
)

if specialized_fns:
    icon = dataset_type_icon(dataset_type)
    lbl = dataset_type_label(dataset_type)
    st.markdown(f"**SPECIALISED — {icon} {lbl}**")
    if active_spec_fns:
        st.radio(
            "specialised_analyses",
            spec_labels,
            key="_spec_radio",
            index=None,
            label_visibility="collapsed",
        )
    unavailable = [f for f in specialized_fns if f.id in disabled_ids]
    if unavailable:
        st.caption("Requires enrichment: " + ", ".join(f.label for f in unavailable))

# ── Selected function: params + run ───────────────────────────────────────────
selected_fn_id = st.session_state.get("selected_fn")

if not selected_fn_id:
    st.info("Select an analysis above to get started.")
    st.stop()

selected_fn = REGISTRY.get(selected_fn_id)
if not selected_fn:
    st.session_state.pop("selected_fn", None)
    st.info("Select an analysis above to get started.")
    st.stop()

st.divider()
st.markdown(f"**Selected: {selected_fn.label}**")
st.caption(selected_fn.description)

# Parameterized functions: show widgets + Run button
collected_params: dict[str, Any] = {}
if selected_fn.params:
    with st.container(border=True):
        st.subheader("Parameters")
        param_cols = st.columns(min(len(selected_fn.params), 3))
        for i, param in enumerate(selected_fn.params):
            with param_cols[i % len(param_cols)]:
                collected_params[param.name] = _render_param_widget(param, meta_raw)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running…"):
            try:
                st.session_state["analytics_result"] = _execute_fn(
                    selected_fn, engine, table_name, meta_raw, enrichment_status, collected_params
                )
                st.session_state["analytics_error"] = None
            except EnrichmentRequiredError as exc:
                st.session_state["analytics_error"] = ("enrichment", str(exc))
                st.session_state["analytics_result"] = None
            except Exception as exc:
                st.session_state["analytics_error"] = ("error", str(exc))
                st.session_state["analytics_result"] = None

# ── Render persisted result ────────────────────────────────────────────────────
analytics_error = st.session_state.get("analytics_error")
analytics_result = st.session_state.get("analytics_result")

if analytics_error:
    kind, msg = analytics_error
    if kind == "enrichment":
        st.warning(msg)
        st.page_link("views/1_data_sources.py", label="Go to Data Sources →", icon="📂")
    else:
        st.error(f"Analysis failed: {msg}")
elif analytics_result and analytics_result.get("fn_id") == selected_fn_id:
    result_df = analytics_result["result_df"]
    fig = analytics_result["fig"]
    fn_label = analytics_result["fn_label"]

    st.divider()
    view_mode = st.radio("View as", ["Chart", "Table"], horizontal=True, key="analytics_view_mode")

    if fig is not None and view_mode == "Chart":
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    btn1, btn2 = st.columns(2)
    with btn1:
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{selected_fn_id}_results.csv",
            mime="text/csv",
        )
    with btn2:
        if fig is not None:
            if st.button("Add to Report", key="add_to_report"):
                add_to_report(fig, fn_label)
                st.success(f"Added '{fn_label}' to report.")
