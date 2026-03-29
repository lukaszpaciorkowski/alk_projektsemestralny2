"""
6_architecture.py — Architecture diagrams page for the Streamlit app.

Features:
- Regenerate Diagrams button (runs mmdc for all .mmd files)
- Radio: ER Diagram | Pipeline Flow | App Architecture
- Display rendered PNG via st.image
- Collapsible description
- View Source expander showing .mmd content
- Download PNG button
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.state import init_state

DIAGRAMS_DIR = Path("docs/diagrams")

DIAGRAM_INFO: dict[str, dict] = {
    "ER Diagram": {
        "mmd": DIAGRAMS_DIR / "er_diagram.mmd",
        "png": DIAGRAMS_DIR / "er_diagram.png",
        "description": (
            "The Entity-Relationship diagram shows the normalised 3NF database schema. "
            "The central `admissions` table links to `patients` (one patient : many encounters), "
            "`admission_types`, and `discharge_types` via foreign keys. "
            "Each admission can have multiple `medications` (unpivoted from 24 drug columns) "
            "and multiple `diagnosis_encounters` (positions 1-3), each referencing a code "
            "in `diagnoses_lookup`."
        ),
    },
    "Pipeline Flow": {
        "mmd": DIAGRAMS_DIR / "pipeline_flow.mmd",
        "png": DIAGRAMS_DIR / "pipeline_flow.png",
        "description": (
            "The pipeline flowchart shows the five-stage ETL process: "
            "(1) CSV download/upload, "
            "(2) validation (null_threshold, outlier_zscore), "
            "(3) load to SQLite 3NF schema, "
            "(4) SQL queries and statistical analysis, "
            "(5) Plotly/Matplotlib visualisation, "
            "(6) HTML + PDF report generation. "
            "Config parameters (shown in dashed annotations) control each stage."
        ),
    },
    "App Architecture": {
        "mmd": DIAGRAMS_DIR / "app_architecture.mmd",
        "png": DIAGRAMS_DIR / "app_architecture.png",
        "description": (
            "The application architecture diagram shows how the five Streamlit pages "
            "interact with the backend scripts and storage layer. "
            "Data Sources (P1) triggers the ingest/load scripts. "
            "Exploration (P2) and Analytics (P3) query the SQLite database via query_helpers.py. "
            "Analytics also drives figure generation (04_visualize.py). "
            "Reports (P4) assembles selected figures into HTML/PDF. "
            "Architecture (P5) renders the pre-generated Mermaid PNG files."
        ),
    },
}


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

render_sidebar()
init_state()

st.title("📐 Architecture")
st.markdown(
    "View the system architecture diagrams. PNG files are generated from Mermaid source "
    "via `mmdc` (requires `npm install -g @mermaid-js/mermaid-cli`)."
)

# ── Regenerate button ─────────────────────────────────────────────────────────
if st.button("🔄 Regenerate Diagrams", help="Re-render all Mermaid .mmd files to PNG"):
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        st.error(
            "**mmdc not found.** Install Mermaid CLI with:\n"
            "```\nnpm install -g @mermaid-js/mermaid-cli\n```"
        )
    else:
        rendered = 0
        errors: list[str] = []
        with st.spinner("Rendering diagrams…"):
            for name, info in DIAGRAM_INFO.items():
                mmd: Path = info["mmd"]
                png: Path = info["png"]
                if not mmd.exists():
                    errors.append(f"{name}: source file `{mmd}` not found")
                    continue
                result = subprocess.run(
                    [mmdc, "-i", str(mmd), "-o", str(png)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    rendered += 1
                else:
                    errors.append(
                        f"{name}: mmdc exited {result.returncode} — "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
        if rendered:
            st.success(f"✅ Rendered {rendered} diagram{'s' if rendered != 1 else ''}.")
        for err in errors:
            st.error(err)
        if rendered:
            st.rerun()

selected = st.radio(
    "Select diagram",
    options=list(DIAGRAM_INFO.keys()),
    horizontal=True,
)

info = DIAGRAM_INFO[selected]
png_path: Path = info["png"]
mmd_path: Path = info["mmd"]

st.divider()

# ---- Diagram Image ----
if png_path.exists():
    with open(png_path, "rb") as f:
        png_bytes = f.read()
    st.image(png_bytes, caption=selected, use_container_width=True)
    st.download_button(
        label=f"Download {selected} PNG",
        data=png_bytes,
        file_name=png_path.name,
        mime="image/png",
    )
else:
    st.info(
        f"PNG not generated yet — click **🔄 Regenerate Diagrams** above.\n\n"
        f"_(source: `{mmd_path}`)_"
    )

# ---- Description ----
with st.expander("Description", expanded=True):
    st.markdown(info["description"])

# ---- Mermaid Source ----
with st.expander("View Mermaid Source (.mmd)", expanded=False):
    if mmd_path.exists():
        mmd_content = mmd_path.read_text(encoding="utf-8")
        st.code(mmd_content, language="text")
        st.download_button(
            label="Download .mmd source",
            data=mmd_content.encode("utf-8"),
            file_name=mmd_path.name,
            mime="text/plain",
        )
    else:
        st.warning(f"Source file not found: `{mmd_path}`")
