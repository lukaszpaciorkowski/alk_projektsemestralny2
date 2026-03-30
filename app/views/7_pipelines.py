"""
7_pipelines.py — Custom analytical pipelines page.

Three tabs:
  My Pipelines — list saved pipelines with run/edit/clone/delete/export
  Builder      — create or edit a pipeline step-by-step
  Run          — select pipeline + dataset, execute with per-step progress
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.core.pipeline import DB_PATH, get_engine, list_datasets, save_dataframe_as_dataset
from app.core.pipelines import (
    clear_pipeline_runs,
    clone_pipeline,
    create_pipeline,
    delete_pipeline,
    ensure_pipelines_tables,
    execute_pipeline_step,
    export_pipeline_json,
    get_pipeline,
    import_pipeline_json,
    list_pipelines,
    list_runs_for_pipeline,
    save_pipeline_run,
    start_pipeline_run,
    update_pipeline,
)
from app.core.registry import REGISTRY
from app.core.type_detector import dataset_type_icon
from app.state import add_to_report, init_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRAFT_KEY = "pipeline_draft"
_RUN_STATE_KEY = "pipeline_run_state"

_SCOPE_ICON = {"generic": "🌐", "diabetes": "🧬"}
_STATUS_ICON = {"completed": "✅", "failed": "❌", "running": "⏳"}

_OP_OPTIONS = ["eq", "neq", "gte", "lte", "gt", "lt", "like", "in", "isnull", "notnull"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine():
    try:
        return get_engine(DB_PATH)
    except Exception:
        return None


def _init_draft(
    id: int | None = None,
    name: str = "",
    description: str = "",
    dataset_type: str = "generic",
    steps: list[dict] | None = None,
) -> None:
    st.session_state[_DRAFT_KEY] = {
        "id":           id,
        "name":         name,
        "description":  description,
        "dataset_type": dataset_type,
        "steps":        [s.copy() for s in (steps or [])],
    }
    # Overwrite widget-bound keys so st.text_input / st.selectbox pick up the
    # new values on the next render (Streamlit ignores value= when the key
    # already exists in session_state).
    st.session_state["builder_name"]  = name
    st.session_state["builder_desc"]  = description
    type_options = ["generic", "diabetes"]
    st.session_state["builder_dtype"] = dataset_type if dataset_type in type_options else "generic"
    # Clear per-step widget keys from any previous draft
    for k in [k for k in st.session_state
              if k.startswith(("bp_", "builder_fn_", "builder_lbl_", "builder_rpt_",
                               "fc_", "fo_", "fv_", "fadd_", "fr_", "up_", "dn_", "rm_"))]:
        st.session_state.pop(k, None)
    # Seed param widgets from the loaded steps so values show correctly
    for step in (steps or []):
        step_id = step.get("step_id", "")
        fn = REGISTRY.get(step.get("function_id", ""))
        if fn is None:
            continue
        saved_params = step.get("params") or {}
        for param in fn.params:
            key = f"bp_{step_id}_{param.name}"
            st.session_state[key] = saved_params.get(param.name, param.default)


def _get_draft() -> dict:
    if _DRAFT_KEY not in st.session_state:
        _init_draft()
    return st.session_state[_DRAFT_KEY]


def _render_param_widget(param, step_id: str) -> Any:
    """Render a param widget using step_id-scoped key (no live meta needed)."""
    label = param.label or param.name.replace("_", " ").capitalize()
    key = f"bp_{step_id}_{param.name}"

    if param.widget == "select":
        idx = param.options.index(param.default) if param.default in param.options else 0
        return st.selectbox(label, param.options, index=idx, key=key)
    if param.widget == "bool":
        return st.checkbox(label, value=bool(param.default), key=key)
    if param.widget == "int":
        return st.number_input(
            label, value=int(param.default), min_value=1, max_value=500, step=1, key=key
        )
    if param.widget in ("select_column", "multiselect_column"):
        # No live table at build time — accept free text
        default_str = str(param.default) if param.default else ""
        return st.text_input(
            label, value=default_str, key=key,
            placeholder="column name",
            help="Type the column name; it will be resolved at run time.",
        )
    return param.default


def _read_params_from_widgets(step: dict) -> dict:
    """Collect current param widget values for a step from session_state."""
    fn = REGISTRY.get(step.get("function_id", ""))
    if fn is None:
        return {}
    return {
        p.name: st.session_state.get(f"bp_{step['step_id']}_{p.name}", p.default)
        for p in fn.params
    }


def _next_step_id(existing_steps: list[dict]) -> str:
    taken = {s.get("step_id", "") for s in existing_steps}
    i = len(existing_steps) + 1
    while f"step_{i}" in taken:
        i += 1
    return f"step_{i}"


# ---------------------------------------------------------------------------
# Tab 1 — My Pipelines
# ---------------------------------------------------------------------------

def _render_my_pipelines(engine) -> None:
    top_a, top_b = st.columns([1, 1])
    with top_a:
        if st.button("＋ New Pipeline", use_container_width=True):
            _init_draft()
            st.session_state["_pipeline_tab_hint"] = "builder"
            st.rerun()
    with top_b:
        show_import = st.toggle("Import JSON", key="show_import_toggle")

    if show_import:
        json_input = st.text_area(
            "Paste pipeline JSON",
            height=100,
            key="import_json_text",
            placeholder='{"name": "My Pipeline", "dataset_type": "generic", "steps": [...]}',
        )
        if st.button("Import", key="import_json_btn") and json_input.strip():
            try:
                new_id = import_pipeline_json(engine, json_input.strip())
                st.success(f"Pipeline imported (id={new_id}).")
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    pipelines = list_pipelines(engine)
    if not pipelines:
        st.info("No pipelines yet. Click **＋ New Pipeline** or switch to the **Builder** tab.")
        return

    for pl in pipelines:
        scope_icon = _SCOPE_ICON.get(pl["dataset_type"], "")
        created = str(pl["created_at"])[:10]

        with st.container(border=True):
            hdr, meta = st.columns([3, 1])
            with hdr:
                st.markdown(f"**{pl['name']}**")
                if pl["description"]:
                    st.caption(pl["description"])
            with meta:
                st.caption(
                    f"{scope_icon} {pl['dataset_type']} · "
                    f"{pl['step_count']} step(s) · {created}"
                )

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                if st.button("▶ Run", key=f"run_{pl['id']}", use_container_width=True):
                    st.session_state["run_pipeline_id"] = pl["id"]
                    st.session_state["_pipeline_tab_hint"] = "run"
                    st.rerun()
            with c2:
                if st.button("✎ Edit", key=f"edit_{pl['id']}", use_container_width=True):
                    _init_draft(
                        id=pl["id"],
                        name=pl["name"],
                        description=pl["description"],
                        dataset_type=pl["dataset_type"],
                        steps=pl["steps"],
                    )
                    st.session_state["_pipeline_tab_hint"] = "builder"
                    st.rerun()
            with c3:
                if st.button("⧉ Clone", key=f"clone_{pl['id']}", use_container_width=True):
                    new_id = clone_pipeline(engine, pl["id"], f"{pl['name']} (copy)")
                    st.success(f"Cloned → id={new_id}")
                    st.rerun()
            with c4:
                try:
                    export_str = export_pipeline_json(engine, pl["id"])
                    st.download_button(
                        "↓ JSON",
                        data=export_str.encode("utf-8"),
                        file_name=f"pipeline_{pl['id']}.json",
                        mime="application/json",
                        key=f"export_{pl['id']}",
                        use_container_width=True,
                    )
                except Exception:
                    pass
            with c5:
                if st.button("✕ Delete", key=f"del_{pl['id']}", use_container_width=True):
                    delete_pipeline(engine, pl["id"])
                    if st.session_state.get("run_pipeline_id") == pl["id"]:
                        st.session_state.pop("run_pipeline_id", None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Tab 2 — Builder
# ---------------------------------------------------------------------------

def _render_builder(engine) -> None:
    draft = _get_draft()

    # ── Metadata ──────────────────────────────────────────────────────────
    col_name, col_type = st.columns([3, 1])
    with col_name:
        draft["name"] = st.text_input(
            "Pipeline name", value=draft.get("name", ""),
            key="builder_name", placeholder="My Analysis Pipeline",
        )
    with col_type:
        type_options = ["generic", "diabetes"]
        cur_type = draft.get("dataset_type", "generic")
        draft["dataset_type"] = st.selectbox(
            "Dataset type", type_options,
            index=type_options.index(cur_type) if cur_type in type_options else 0,
            key="builder_dtype",
            format_func=lambda x: f"{_SCOPE_ICON.get(x, '')} {x}",
        )
    draft["description"] = st.text_input(
        "Description", value=draft.get("description", ""),
        key="builder_desc", placeholder="What does this pipeline do?",
    )

    st.divider()
    st.markdown(f"**Steps** ({len(draft['steps'])} total)")

    steps = draft["steps"]
    to_remove: int | None = None
    swap: tuple[int, int] | None = None

    for i, step in enumerate(steps):
        step_id = step["step_id"]
        fn_id   = step.get("function_id", "")
        fn      = REGISTRY.get(fn_id)

        with st.container(border=True):
            # ── Header row ────────────────────────────────────────────────
            fn_col, lbl_col, up_col, dn_col, rm_col = st.columns([3, 2, 0.5, 0.5, 0.5])

            with fn_col:
                all_ids = list(REGISTRY.keys())
                cur_idx = all_ids.index(fn_id) if fn_id in all_ids else 0
                chosen_fn_id = st.selectbox(
                    f"Step {i + 1}",
                    all_ids,
                    index=cur_idx,
                    key=f"builder_fn_{step_id}",
                    format_func=lambda k: REGISTRY[k].label,
                )
                if chosen_fn_id != fn_id:
                    step["function_id"] = chosen_fn_id
                    step["label"]       = REGISTRY[chosen_fn_id].label
                    step["params"]      = {}
                    fn = REGISTRY[chosen_fn_id]

            with lbl_col:
                step["label"] = st.text_input(
                    "Label",
                    value=step.get("label", fn.label if fn else ""),
                    key=f"builder_lbl_{step_id}",
                    label_visibility="collapsed",
                    placeholder="Step label",
                )
            with up_col:
                if i > 0 and st.button("↑", key=f"up_{step_id}"):
                    swap = (i - 1, i)
            with dn_col:
                if i < len(steps) - 1 and st.button("↓", key=f"dn_{step_id}"):
                    swap = (i, i + 1)
            with rm_col:
                if st.button("✕", key=f"rm_{step_id}"):
                    to_remove = i

            # ── Params ────────────────────────────────────────────────────
            fn = REGISTRY.get(step["function_id"])
            if fn and fn.params:
                pcols = st.columns(min(len(fn.params), 3))
                for j, param in enumerate(fn.params):
                    with pcols[j % len(pcols)]:
                        _render_param_widget(param, step_id)
            else:
                st.caption("No configurable parameters.")

            # ── Per-step filters ──────────────────────────────────────────
            filters = step.get("filters") or []
            filter_label = f"Filters ({len(filters)} active)" if filters else "Filters"
            with st.expander(filter_label, expanded=False):
                surviving: list[dict] = []
                removed_any = False
                for fi, flt in enumerate(filters):
                    c1, c2, c3, c4 = st.columns([2, 2, 3, 0.5])
                    with c1:
                        col_val = st.text_input(
                            "Col", value=flt.get("column", ""),
                            key=f"fc_{step_id}_{fi}", label_visibility="collapsed",
                            placeholder="column",
                        )
                    with c2:
                        op_val = st.selectbox(
                            "Op",
                            _OP_OPTIONS,
                            index=_OP_OPTIONS.index(flt.get("op", "eq")) if flt.get("op") in _OP_OPTIONS else 0,
                            key=f"fo_{step_id}_{fi}", label_visibility="collapsed",
                        )
                    with c3:
                        if op_val in ("isnull", "notnull"):
                            st.caption("(no value)")
                            v_val = None
                        else:
                            v_val = st.text_input(
                                "Val", value=str(flt.get("value", "") or ""),
                                key=f"fv_{step_id}_{fi}", label_visibility="collapsed",
                                placeholder="value",
                            )
                    with c4:
                        if st.button("✕", key=f"fr_{step_id}_{fi}"):
                            removed_any = True
                            continue
                    surviving.append({"column": col_val, "op": op_val, "value": v_val})

                step["filters"] = surviving
                if removed_any:
                    st.rerun()

                if st.button("+ Add filter", key=f"fadd_{step_id}"):
                    step["filters"].append({"column": "", "op": "eq", "value": ""})
                    st.rerun()

            # ── Add to report ─────────────────────────────────────────────
            step["add_to_report"] = st.checkbox(
                "Add to report",
                value=step.get("add_to_report", True),
                key=f"builder_rpt_{step_id}",
            )

    # ── Structural mutations (deferred to avoid index shifting) ───────────
    if to_remove is not None:
        draft["steps"] = [s for j, s in enumerate(steps) if j != to_remove]
        st.rerun()
    if swap is not None:
        a, b = swap
        draft["steps"][a], draft["steps"][b] = draft["steps"][b], draft["steps"][a]
        st.rerun()

    # ── Add step ──────────────────────────────────────────────────────────
    if st.button("＋ Add Step", use_container_width=True):
        first_id = next(iter(REGISTRY))
        draft["steps"].append({
            "step_id":      _next_step_id(draft["steps"]),
            "function_id":  first_id,
            "label":        REGISTRY[first_id].label,
            "params":       {},
            "filters":      [],
            "add_to_report": True,
        })
        st.rerun()

    st.divider()

    # ── Action buttons ────────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)

    def _collect_params_into_draft() -> bool:
        name = draft.get("name", "").strip()
        if not name:
            st.error("Pipeline name is required.")
            return False
        if not draft["steps"]:
            st.error("Add at least one step.")
            return False
        for step in draft["steps"]:
            step["params"] = _read_params_from_widgets(step)
        return True

    with b1:
        if st.button("💾 Save", type="primary", use_container_width=True):
            if _collect_params_into_draft():
                if draft.get("id"):
                    update_pipeline(
                        engine, draft["id"],
                        name=draft["name"],
                        description=draft.get("description", ""),
                        dataset_type=draft.get("dataset_type", "generic"),
                        steps=draft["steps"],
                    )
                    st.success(f"'{draft['name']}' updated.")
                else:
                    new_id = create_pipeline(
                        engine,
                        name=draft["name"],
                        description=draft.get("description", ""),
                        dataset_type=draft.get("dataset_type", "generic"),
                        steps=draft["steps"],
                    )
                    draft["id"] = new_id
                    st.success(f"'{draft['name']}' saved (id={new_id}).")

    with b2:
        if st.button("▶ Run Now", use_container_width=True):
            if _collect_params_into_draft():
                if draft.get("id"):
                    update_pipeline(engine, draft["id"], steps=draft["steps"])
                    pid = draft["id"]
                else:
                    pid = create_pipeline(
                        engine,
                        name=draft["name"],
                        description=draft.get("description", ""),
                        dataset_type=draft.get("dataset_type", "generic"),
                        steps=draft["steps"],
                    )
                    draft["id"] = pid
                st.session_state["run_pipeline_id"] = pid
                st.session_state["_pipeline_tab_hint"] = "run"
                st.info("Pipeline saved. Switch to the **Run** tab to execute it.")

    with b3:
        if draft.get("id"):
            try:
                export_str = export_pipeline_json(engine, draft["id"])
                st.download_button(
                    "↓ Export",
                    data=export_str.encode("utf-8"),
                    file_name=f"pipeline_{draft['id']}.json",
                    mime="application/json",
                    key="builder_export_btn",
                    use_container_width=True,
                )
            except Exception:
                pass

    with b4:
        if st.button("⟳ Clear", use_container_width=True):
            _init_draft()
            for k in [k for k in st.session_state if k.startswith(("bp_", "builder_", "fc_", "fo_", "fv_"))]:
                st.session_state.pop(k, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — Run
# ---------------------------------------------------------------------------

def _render_run(engine, datasets: list[dict]) -> None:
    pipelines = list_pipelines(engine)
    if not pipelines:
        st.info("No pipelines yet. Create one in the **Builder** tab.")
        return
    if not datasets:
        st.warning("No datasets imported. Go to **Data Sources** first.")
        return

    # ── Selectors ────────────────────────────────────────────────────────
    preselect_id = st.session_state.get("run_pipeline_id")
    pl_labels = [
        f"{_SCOPE_ICON.get(p['dataset_type'], '')} {p['name']} ({p['step_count']} steps)"
        for p in pipelines
    ]
    default_pl_idx = 0
    if preselect_id:
        for idx, pl in enumerate(pipelines):
            if pl["id"] == preselect_id:
                default_pl_idx = idx
                break

    selected_pl_idx = st.selectbox(
        "Pipeline",
        range(len(pl_labels)),
        index=default_pl_idx,
        format_func=lambda i: pl_labels[i],
    )
    selected_pl = pipelines[selected_pl_idx]

    ds_labels = [
        f"{dataset_type_icon(d['dataset_type'])} {d['display_name']} ({d['row_count']:,} rows)"
        for d in datasets
    ]
    selected_ds_idx = st.selectbox(
        "Dataset",
        range(len(ds_labels)),
        format_func=lambda i: ds_labels[i],
    )
    selected_ds = datasets[selected_ds_idx]

    # ── Compatibility ─────────────────────────────────────────────────────
    compatible = (
        selected_pl["dataset_type"] == "generic"
        or selected_pl["dataset_type"] == selected_ds["dataset_type"]
    )
    if not compatible:
        st.warning(
            f"This pipeline requires a **{selected_pl['dataset_type']}** dataset, "
            f"but **{selected_ds['display_name']}** is **{selected_ds['dataset_type']}**. "
            "Select a compatible dataset or switch to a generic pipeline."
        )

    # ── Past runs ─────────────────────────────────────────────────────────
    past_runs = list_runs_for_pipeline(engine, selected_pl["id"])
    if past_runs:
        with st.expander(f"Run history ({len(past_runs)})", expanded=False):
            for r in past_runs[:10]:
                icon = _STATUS_ICON.get(r["status"], "")
                st.caption(
                    f"{icon} {r['started_at']} — {r['status']} — {r['dataset_table']}"
                )

    st.divider()

    if st.button(
        "▶ Run Pipeline",
        type="primary",
        disabled=not compatible,
        use_container_width=True,
    ):
        pl = get_pipeline(engine, selected_pl["id"])
        if pl is None:
            st.error("Pipeline not found.")
            return

        ds          = selected_ds
        table_name  = ds["table_name"]
        enrich_stat = ds.get("enrichment_status", "none")
        meta_raw    = ds.get("columns") or []
        if isinstance(meta_raw, str):
            meta_raw = json.loads(meta_raw)

        # Clear previous run results so history stays clean
        clear_pipeline_runs(engine, pl["id"])
        run_id = start_pipeline_run(engine, pl["id"], table_name)

        step_results: list[dict] = []
        all_failed = True

        with st.status("Running pipeline…", expanded=True) as status_box:
            for i, step in enumerate(pl["steps"]):
                st.write(f"Step {i + 1}/{len(pl['steps'])}: **{step['label']}**…")
                sr = execute_pipeline_step(engine, step, table_name, meta_raw, enrich_stat)
                step_results.append(sr)
                if sr["status"] == "completed":
                    all_failed = False
                    st.write(
                        f"  ✅ {sr['label']} — {sr['result_summary']} ({sr['duration_s']}s)"
                    )
                else:
                    st.write(f"  ❌ {sr['label']} — {sr['error']}")

            final_status = "failed" if all_failed else "completed"
            save_pipeline_run(engine, run_id, step_results, final_status)

            # Auto-add charts to report
            n_added = 0
            for step, sr in zip(pl["steps"], step_results):
                if step.get("add_to_report") and sr["status"] == "completed" and sr.get("fig_json"):
                    try:
                        import plotly.io as _pio
                        add_to_report(
                            fig=_pio.from_json(sr["fig_json"]),
                            title=f"{pl['name']} — {sr['label']}",
                            dataset_name=ds["display_name"],
                            row_count=ds.get("row_count"),
                            total_rows=ds.get("row_count"),
                        )
                        n_added += 1
                    except Exception:
                        pass

            state_label = "complete" if final_status == "completed" else "error"
            status_box.update(
                label=f"Pipeline {final_status}. {n_added} chart(s) added to report.",
                state=state_label,
            )

        st.session_state[_RUN_STATE_KEY] = {
            "pipeline_name":     pl["name"],
            "dataset_name":      ds["display_name"],
            "dataset_table":     table_name,
            "step_results":      step_results,
            "pipeline_steps":    pl["steps"],   # kept so we can re-fetch filtered data
            "run_id":            run_id,
            "n_added_to_report": n_added,
        }

    # ── Results view ──────────────────────────────────────────────────────
    run_state = st.session_state.get(_RUN_STATE_KEY)
    if run_state and run_state.get("step_results"):
        st.divider()
        st.subheader("Last Run Results")
        st.caption(
            f"Pipeline: **{run_state['pipeline_name']}** · "
            f"Dataset: **{run_state['dataset_name']}** · "
            f"{run_state['n_added_to_report']} chart(s) added to report"
        )

        import pandas as pd
        pipeline_steps = run_state.get("pipeline_steps", [])
        for i, sr in enumerate(run_state["step_results"]):
            icon = _STATUS_ICON.get(sr["status"], "")
            dur  = sr.get("duration_s", 0)
            with st.expander(
                f"{icon} Step {i + 1}: {sr['label']} ({dur}s)",
                expanded=(sr["status"] == "failed"),
            ):
                if sr["status"] == "failed":
                    st.error(sr.get("error", "Unknown error"))
                    continue

                view_mode = st.radio(
                    "View as",
                    ["Chart", "Table"],
                    horizontal=True,
                    key=f"run_view_{run_state['run_id']}_{i}",
                )

                if view_mode == "Chart" and sr.get("fig_json"):
                    try:
                        import plotly.io as _pio
                        st.plotly_chart(_pio.from_json(sr["fig_json"]), use_container_width=True)
                    except Exception as exc:
                        st.error(f"Could not render chart: {exc}")

                if view_mode == "Table" and sr.get("result_df_json"):
                    try:
                        result_df = pd.DataFrame(json.loads(sr["result_df_json"]))
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        csv = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download CSV",
                            data=csv,
                            file_name=f"step_{i + 1}_{sr['label'].replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"dl_{run_state['run_id']}_{i}",
                        )
                    except Exception as exc:
                        st.error(f"Could not load table: {exc}")

                st.caption(sr.get("result_summary", ""))

                # ── Save as Dataset ───────────────────────────────────────
                step = pipeline_steps[i] if i < len(pipeline_steps) else {}
                step_filters = step.get("filters") or []
                source_table = run_state.get("dataset_table", "")

                save_key = f"save_ds_{run_state['run_id']}_{i}"
                name_key  = f"save_ds_name_{run_state['run_id']}_{i}"

                # Suggest a sanitised default name
                import re as _re
                label_slug = _re.sub(r"[^a-z0-9]+", "_", sr["label"].lower()).strip("_")
                source_slug = _re.sub(r"[^a-z0-9]+", "_", run_state.get("dataset_name", "data").lower()).strip("_")
                filter_hint = f"_{len(step_filters)}filters" if step_filters else ""
                default_name = f"{source_slug}_{label_slug}{filter_hint}"

                with st.container():
                    ds_name_input = st.text_input(
                        "Dataset name",
                        value=default_name,
                        key=name_key,
                        label_visibility="collapsed",
                        placeholder="Name for the new dataset",
                    )
                    if st.button("💾 Save as Dataset", key=save_key):
                        try:
                            from app.core.query import Filter, fetch_table
                            filter_objs = [
                                Filter(f["column"], f["op"], f.get("value"))
                                for f in step_filters
                                if f.get("column") and f.get("op")
                            ]
                            full_df = fetch_table(
                                source_table, engine,
                                filters=filter_objs, limit=500_000,
                            )
                            desc = (
                                f"Derived from pipeline '{run_state['pipeline_name']}' "
                                f"step '{sr['label']}' on dataset '{run_state['dataset_name']}'"
                                + (f" with {len(step_filters)} filter(s)." if step_filters else ".")
                            )
                            new_table = save_dataframe_as_dataset(
                                full_df, ds_name_input.strip() or default_name,
                                engine, description=desc,
                            )
                            st.success(
                                f"Saved {len(full_df):,} rows as **{ds_name_input}** "
                                f"(table: `{new_table}`). Available on Data Sources page."
                            )
                        except Exception as exc:
                            st.error(f"Save failed: {exc}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_sidebar()

engine = _get_engine()
init_state(engine)

st.title("🔧 Pipelines")
st.markdown(
    "Build reusable analytical workflows. Each pipeline is a sequence of registry "
    "functions with pre-configured parameters and optional row filters."
)

if engine is None:
    st.warning("Database not available. Import a dataset on the **Data Sources** page first.")
    st.stop()

ensure_pipelines_tables(engine)

datasets = list_datasets(engine)

hint = st.session_state.pop("_pipeline_tab_hint", None)

tab1, tab2, tab3 = st.tabs(["My Pipelines", "Builder", "Run"])

if hint == "builder":
    st.info("Draft loaded — click the **Builder** tab.")
elif hint == "run":
    st.info("Pipeline selected — click the **Run** tab.")

with tab1:
    _render_my_pipelines(engine)

with tab2:
    _render_builder(engine)

with tab3:
    _render_run(engine, datasets)
