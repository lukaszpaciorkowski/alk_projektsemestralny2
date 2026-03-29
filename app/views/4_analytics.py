"""
4_analytics.py — Registry-driven Analytics page.

Loads all applicable analytics functions for the active dataset type.
Parameters are rendered dynamically from each function's ParamSpec list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.core.pipeline import (
    DB_PATH,
    EnrichmentRequiredError,
    get_engine,
    get_dataset_meta,
    list_datasets,
)
from app.core.registry import AnalyticsFunction, get_functions_for
from app.core.type_detector import dataset_type_icon, dataset_type_label
from app.state import add_to_report, init_state, set_active_dataset

PAGE_SIZE = 50


def _get_engine():
    return get_engine(DB_PATH)


def _col_names_by_dtype(meta: list[dict], dtype_filter: str | None) -> list[str]:
    """Filter column names by dtype category."""
    if not dtype_filter:
        return [c["name"] for c in meta]
    if dtype_filter == "numeric":
        return [c["name"] for c in meta if "int" in c.get("dtype", "") or "float" in c.get("dtype", "")]
    if dtype_filter == "categorical":
        return [c["name"] for c in meta if "int" not in c.get("dtype", "") and "float" not in c.get("dtype", "")]
    return [c["name"] for c in meta]


def _render_param_widget(param, meta: list[dict]) -> Any:
    """Render a Streamlit widget for a single ParamSpec. Returns the value."""
    label = param.label or param.name.replace("_", " ").capitalize()

    if param.widget == "select":
        return st.selectbox(label, param.options, index=param.options.index(param.default) if param.default in param.options else 0)

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

# ---- Dataset Selector ----
ds_options = {
    f"{dataset_type_icon(d['dataset_type'])} {d['display_name']} "
    f"({d['row_count']:,} rows)": d
    for d in datasets
}

active_name = st.session_state.get("active_dataset_name", "")
default_idx = 0
for idx, label in enumerate(ds_options):
    if active_name and active_name in label:
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
    st.rerun()

# Get applicable functions
functions = get_functions_for(dataset_type)
generic_fns = [f for f in functions if f.scope == "generic"]
specialized_fns = [f for f in functions if f.scope != "generic"]

# ---- Function Grid ----
st.divider()
st.subheader("Available Analyses")

current_selection = st.session_state.get("selected_analysis")
all_fns = {f.id: f for f in functions}

cols_per_row = 4


def _render_fn_buttons(fns: list[AnalyticsFunction], disabled_ids: set[str]) -> None:
    """Render a row-based button grid; updates selected_analysis in session state."""
    for row_start in range(0, len(fns), cols_per_row):
        row_fns = fns[row_start:row_start + cols_per_row]
        btn_cols = st.columns(cols_per_row)
        for bcol, fn in zip(btn_cols, row_fns):
            with bcol:
                is_active = st.session_state.get("selected_analysis") == fn.id
                disabled = fn.id in disabled_ids
                label_text = fn.label + (" ⚠" if disabled else "")
                # No st.rerun() — Streamlit already reruns on button click.
                # Calling st.rerun() here causes a double-rerun that drops
                # the session-state write made in this same pass.
                if st.button(
                    label_text,
                    key=f"fn_{fn.id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=fn.description + (" (requires enrichment)" if disabled else ""),
                    disabled=disabled,
                ):
                    if st.session_state.get("selected_analysis") != fn.id:
                        # New function selected — clear any cached result
                        st.session_state["selected_analysis"] = fn.id
                        st.session_state["analytics_result"] = None
                        st.session_state["analytics_error"] = None


# Generic group
st.markdown("**GENERIC — works on any dataset**")
_render_fn_buttons(generic_fns, disabled_ids=set())

# Specialized group
if specialized_fns:
    icon = dataset_type_icon(dataset_type)
    label = dataset_type_label(dataset_type)
    st.markdown(f"**SPECIALISED — {icon} {label}**")

    needs_enrichment = {f.id for f in specialized_fns if f.requires_enrichment}
    if enrichment_status != "done" and needs_enrichment:
        st.warning(
            f"**{', '.join(all_fns[fid].label for fid in needs_enrichment if fid in all_fns)}**"
            " require enrichment. Run enrichment on the **Data Sources** page."
        )

    disabled_ids = needs_enrichment if enrichment_status != "done" else set()
    _render_fn_buttons(specialized_fns, disabled_ids=disabled_ids)

# ---- Parameters + Run ----
selected_fn_id = st.session_state.get("selected_analysis")
if not selected_fn_id:
    st.info("Select an analysis above to get started.")
    st.stop()

selected_fn = all_fns.get(selected_fn_id)
if not selected_fn:
    # Selected function no longer exists (e.g. switched dataset type)
    st.session_state["selected_analysis"] = None
    st.session_state["analytics_result"] = None
    st.info("Select an analysis above to get started.")
    st.stop()

st.divider()
st.markdown(f"**Selected: {selected_fn.label}**")
st.caption(selected_fn.description)

# Collect params from UI
collected_params: dict[str, Any] = {}
if selected_fn.params:
    with st.container(border=True):
        st.subheader("Parameters")
        param_cols = st.columns(min(len(selected_fn.params), 3))
        for i, param in enumerate(selected_fn.params):
            with param_cols[i % len(param_cols)]:
                collected_params[param.name] = _render_param_widget(param, meta_raw)

# Parameterless functions run immediately on selection.
# Parameterized functions require an explicit "Run Analysis" click.
no_params = not selected_fn.params
cached = st.session_state.get("analytics_result")
cached_is_current = cached and cached.get("fn_id") == selected_fn_id

run_clicked = False
if no_params:
    if not cached_is_current:
        run_clicked = True   # auto-run
else:
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running…"):
        try:
            from sqlalchemy import text as sql_text
            with engine.connect() as conn:
                df = pd.read_sql(sql_text(f"SELECT * FROM [{table_name}]"), conn)

            if selected_fn.requires_enrichment:
                collected_params["con"] = engine
                collected_params["table_name"] = table_name
                collected_params["enrichment_status"] = enrichment_status

            result_df, fig = selected_fn.fn(df, meta_raw, **collected_params)

            st.session_state["analytics_result"] = {
                "fn_id": selected_fn_id,
                "fn_label": selected_fn.label,
                "result_df": result_df,
                "fig": fig,
            }
            st.session_state["analytics_error"] = None

        except EnrichmentRequiredError as exc:
            st.session_state["analytics_error"] = ("enrichment", str(exc))
            st.session_state["analytics_result"] = None
        except Exception as exc:
            st.session_state["analytics_error"] = ("error", str(exc))
            st.session_state["analytics_result"] = None

# ---- Render persisted result ----
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
