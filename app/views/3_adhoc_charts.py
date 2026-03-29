"""
3_adhoc_charts.py — Ad Hoc Charts page.

Allows visual chart building on any imported dataset.
Chart type selection drives dynamic axis controls.
Results can be added to the report or downloaded as PNG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.chart_builder import build_chart
from app.components.filter_panel import active_filter_count, render_filter_panel
from app.components.sidebar import render_sidebar
from app.core.pipeline import DB_PATH, get_engine, list_datasets
from app.core.type_detector import dataset_type_icon
from app.state import add_to_report, init_state, set_active_dataset

CHART_TYPES = [
    "Bar", "Line", "Scatter", "Box", "Histogram", "Heatmap", "Choropleth Map",
    "Pie", "Donut", "Multi-Line", "Area (Stacked)", "3D Scatter", "Sunburst", "Treemap",
]

AGG_FUNCS = ["mean", "sum", "count", "min", "max", "median"]

LOCATION_MODES = ["Auto-detect", "Country names", "ISO-3 codes"]
_LOCATION_MODE_MAP = {
    "Auto-detect": "auto",
    "Country names": "country names",
    "ISO-3 codes": "ISO-3",
}


def _get_engine():
    return get_engine(DB_PATH)


def _col_category(dtype: str) -> str:
    if "int" in dtype or "float" in dtype:
        return "numeric"
    return "categorical"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_sidebar()

try:
    engine = _get_engine()
except Exception:
    engine = None

init_state(engine)

st.title("📈 Ad Hoc Charts")
st.markdown("Build interactive charts from any imported dataset.")

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

meta_raw = selected_ds.get("columns") or []
if isinstance(meta_raw, str):
    meta_raw = json.loads(meta_raw)

# Sync active dataset
if st.session_state.get("active_dataset") != table_name:
    set_active_dataset(
        table_name=table_name,
        display_name=selected_ds["display_name"],
        dataset_type=selected_ds["dataset_type"],
        enrichment_status=selected_ds["enrichment_status"],
        meta=meta_raw,
    )

all_cols = [c["name"] for c in meta_raw]
numeric_cols = [c["name"] for c in meta_raw if _col_category(c.get("dtype", "")) == "numeric"]
cat_cols = [c["name"] for c in meta_raw if _col_category(c.get("dtype", "")) == "categorical"]

# ---- Filter Panel ----
st.divider()
n_adhoc_filters = active_filter_count("adhoc")
_adhoc_label = (
    f"🔍 Data Filters ({n_adhoc_filters} active)" if n_adhoc_filters else "🔍 Data Filters"
)
with st.expander(_adhoc_label, expanded=n_adhoc_filters > 0):
    adhoc_filters = render_filter_panel(table_name, meta_raw, engine, key_prefix="adhoc")

# ---- Chart Builder ----
st.divider()
with st.container(border=True):
    st.subheader("Chart Builder")

    b_col1, b_col2 = st.columns([1, 3])
    with b_col1:
        chart_type = st.selectbox("Chart type", CHART_TYPES)

    # Dynamic axis controls — all params initialised to safe defaults
    x_col = y_col = color_col = facet_col = z_col = size_col = None
    path_cols: list[str] = []
    agg_func = "mean"
    bins = 30
    location_mode = "auto"
    top_n = 10

    if chart_type == "Choropleth Map":
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        with c1:
            x_col = st.selectbox("Location column", all_cols, key="adhoc_x",
                                 help="Column with country names or ISO codes")
        with c2:
            y_col = st.selectbox("Value column (numeric)", numeric_cols or all_cols,
                                 key="adhoc_y")
        with c3:
            agg_func = st.selectbox("Aggregation", AGG_FUNCS, key="adhoc_agg")
        with c4:
            loc_mode_label = st.selectbox("Location mode", LOCATION_MODES,
                                          key="adhoc_locmode")
            location_mode = _LOCATION_MODE_MAP[loc_mode_label]
        st.caption("🗺️ Colors countries by the aggregated value.")

    elif chart_type == "Histogram":
        x_col = st.selectbox("Column (X)", numeric_cols or all_cols, key="adhoc_x")
        color_col = st.selectbox("Color (optional)", ["None"] + cat_cols, key="adhoc_color")
        bins = st.slider("Bins", min_value=5, max_value=100, value=30)
        color_col = None if color_col == "None" else color_col

    elif chart_type == "Scatter":
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X axis", numeric_cols or all_cols, key="adhoc_x")
        with c2:
            y_col = st.selectbox("Y axis", numeric_cols or all_cols, key="adhoc_y")
        with c3:
            color_col = st.selectbox("Color (optional)", ["None"] + all_cols, key="adhoc_color")
            color_col = None if color_col == "None" else color_col

    elif chart_type == "Box":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis (categorical)", cat_cols or all_cols, key="adhoc_x")
        with c2:
            y_col = st.selectbox("Y axis (numeric)", numeric_cols or all_cols, key="adhoc_y")

    elif chart_type == "Heatmap":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis (categorical)", cat_cols or all_cols, key="adhoc_x")
        with c2:
            y_col = st.selectbox("Y axis (categorical)", cat_cols or all_cols, key="adhoc_y")

    elif chart_type in ("Pie", "Donut"):
        c1, c2, c3 = st.columns([3, 3, 1])
        with c1:
            x_col = st.selectbox("Category column", cat_cols or all_cols, key="adhoc_x")
        with c2:
            _y_opt = st.selectbox("Value column (optional — default: count)",
                                  ["None"] + numeric_cols, key="adhoc_y")
            y_col = None if _y_opt == "None" else _y_opt
        with c3:
            top_n = st.number_input("Top N", min_value=2, max_value=100, value=10,
                                    key="adhoc_topn", step=1)
        st.caption("Slices beyond Top N are grouped into 'Other'.")

    elif chart_type in ("Multi-Line", "Area (Stacked)"):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            x_col = st.selectbox("X axis", all_cols, key="adhoc_x")
        with c2:
            y_col = st.selectbox("Y axis (numeric)", numeric_cols or all_cols, key="adhoc_y")
        with c3:
            _grp = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="adhoc_color")
            color_col = None if _grp == "None" else _grp
        with c4:
            agg_func = st.selectbox("Agg", AGG_FUNCS, key="adhoc_agg")

    elif chart_type == "3D Scatter":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            x_col = st.selectbox("X axis", numeric_cols or all_cols, key="adhoc_x")
        with c2:
            y_col = st.selectbox("Y axis", numeric_cols or all_cols, key="adhoc_y")
        with c3:
            z_col = st.selectbox("Z axis", numeric_cols or all_cols, key="adhoc_z")
        with c4:
            _c = st.selectbox("Color (optional)", ["None"] + all_cols, key="adhoc_color")
            color_col = None if _c == "None" else _c
        with c5:
            _s = st.selectbox("Size (optional)", ["None"] + numeric_cols, key="adhoc_size")
            size_col = None if _s == "None" else _s
        st.caption("Sampled to 10,000 points for performance.")

    elif chart_type in ("Sunburst", "Treemap"):
        c1, c2 = st.columns([3, 2])
        with c1:
            path_cols = st.multiselect(
                "Path columns (hierarchical order)", cat_cols or all_cols,
                key="adhoc_path",
                help="Select in order: outermost → innermost (e.g., continent, country, disease)",
            )
        with c2:
            _y_opt = st.selectbox("Value column (optional — default: count)",
                                  ["None"] + numeric_cols, key="adhoc_y")
            y_col = None if _y_opt == "None" else _y_opt
        st.caption("Select path columns in hierarchical order.")

    else:  # Bar / Line
        c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 2])
        with c1:
            x_col = st.selectbox("X axis", all_cols, key="adhoc_x")
        with c2:
            y_col = st.selectbox("Y axis (numeric)", numeric_cols or all_cols, key="adhoc_y")
        with c3:
            agg_func = st.selectbox("Agg", AGG_FUNCS, key="adhoc_agg")
        with c4:
            color_col = st.selectbox("Color (optional)", ["None"] + cat_cols, key="adhoc_color")
            color_col = None if color_col == "None" else color_col
        with c5:
            facet_col = st.selectbox("Facet (optional)", ["None"] + cat_cols, key="adhoc_facet")
            facet_col = None if facet_col == "None" else facet_col

    btn_col, reset_col = st.columns([1, 5])
    with btn_col:
        plot_clicked = st.button("Plot", type="primary", use_container_width=True)
    with reset_col:
        if st.button("Reset"):
            st.rerun()

# ---- Chart Output ----
_can_plot = x_col or (chart_type in ("Sunburst", "Treemap") and path_cols)
if plot_clicked and _can_plot:
    with st.spinner("Building chart..."):
        fig = build_chart(
            table_name=table_name,
            engine=engine,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            facet_col=facet_col,
            agg_func=agg_func,
            bins=bins,
            location_mode=location_mode,
            filters=adhoc_filters,
            z_col=z_col,
            size_col=size_col,
            path_cols=path_cols,
            top_n=int(top_n),
        )

    st.session_state.setdefault("adhoc_chart_history", [])

    # Build a descriptive title for history / report
    if chart_type in ("Sunburst", "Treemap"):
        chart_title = f"{chart_type} — {' > '.join(path_cols)}" if path_cols else chart_type
    elif chart_type in ("Pie", "Donut"):
        chart_title = f"{chart_type} — {x_col}" + (f" / {y_col}" if y_col else " (counts)")
    elif chart_type == "3D Scatter":
        chart_title = f"3D Scatter — {x_col} × {y_col} × {z_col}"
    elif chart_type in ("Multi-Line", "Area (Stacked)"):
        chart_title = f"{chart_type} — {x_col} × {y_col}"
        if color_col:
            chart_title += f" by {color_col}"
    else:
        chart_title = f"{chart_type} — {x_col}" + (f" × {y_col}" if y_col else "")
        if color_col:
            chart_title += f" by {color_col}"

    # Build filter dicts for serialisable history storage
    _filter_dicts = [
        {"column": f.column, "op": f.op, "value": f.value}
        for f in adhoc_filters
    ] if adhoc_filters else []
    _total = selected_ds["row_count"]

    st.session_state["adhoc_chart_history"].append(
        {
            "title": chart_title,
            "fig": fig,
            "filters": _filter_dicts,
            "dataset_name": selected_ds["display_name"],
            "total_rows": _total,
        }
    )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Add to Report"):
                add_to_report(
                    fig=fig,
                    title=chart_title,
                    filters=adhoc_filters,
                    dataset_name=selected_ds["display_name"],
                    total_rows=_total,
                )
                st.success(f"Added '{chart_title}' to report.")
        with btn2:
            # Plotly PNG download (requires kaleido)
            try:
                import io
                img_bytes = fig.to_image(format="png", width=1200, height=700)
                st.download_button(
                    "Download PNG",
                    data=img_bytes,
                    file_name=f"{chart_title.replace(' ', '_')}.png",
                    mime="image/png",
                )
            except Exception:
                st.caption("Install `kaleido` for PNG download.")

# ---- Session History ----
history = st.session_state.get("adhoc_chart_history", [])
if history:
    st.divider()
    st.subheader("Session History")
    for i, item in enumerate(reversed(history)):
        idx = len(history) - 1 - i
        h_col1, h_col2, h_col3 = st.columns([6, 1, 1])
        with h_col1:
            st.caption(f"{i + 1}. {item['title']}")
        with h_col2:
            if st.button("Add to report", key=f"hist_report_{idx}"):
                add_to_report(
                    fig=item["fig"],
                    title=item["title"],
                    filters=item.get("filters", []),
                    dataset_name=item.get("dataset_name", ""),
                    total_rows=item.get("total_rows"),
                )
                st.success("Added to report.")
        with h_col3:
            if st.button("✕", key=f"hist_rm_{idx}"):
                st.session_state["adhoc_chart_history"].pop(idx)
                st.rerun()
