"""
4_reports.py — Reports page for the Streamlit app.

Features:
- Show figures added to report from session state
- Up/down/remove controls
- Report metadata (title, author)
- Section checkboxes
- Preview
- Export PDF and HTML with st.download_button
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.sidebar import render_sidebar
from app.core.pipeline import DB_PATH, get_engine
from app.state import (
    clear_report,
    get_report_items,
    init_state,
    move_report_item,
    remove_report_item,
)

CONFIG_PATH = "config.json"

_OP_LABELS: dict[str, str] = {
    "eq": "=", "neq": "≠", "gte": "≥", "lte": "≤",
    "gt": ">", "lt": "<",
    "in": "IN", "nin": "NOT IN",
    "like": "contains", "isnull": "is null", "notnull": "not null",
}


def _filter_to_str(f: dict) -> str:
    """Return a human-readable string for a single filter dict."""
    col = f.get("column", "?")
    op_code = f.get("op", "")
    op_label = _OP_LABELS.get(op_code, op_code)
    val = f.get("value")

    if op_code in ("isnull", "notnull"):
        return f"{col} {op_label}"
    if op_code in ("in", "nin") and isinstance(val, list):
        vals = [str(v) for v in val[:5]]
        suffix = f", +{len(val) - 5} more" if len(val) > 5 else ""
        return f"{col} {op_label} [{', '.join(vals)}{suffix}]"
    if op_code == "like" and isinstance(val, str):
        return f"{col} contains '{val.strip('%')}'"
    return f"{col} {op_label} {val}"


def _filter_context_lines(item: dict) -> list[str]:
    """Return formatted context lines for a report item (empty list = no context)."""
    lines: list[str] = []
    if item.get("dataset_name"):
        lines.append(f"Dataset: {item['dataset_name']}")
    filters = item.get("filters") or []
    if filters:
        parts = " AND ".join(_filter_to_str(f) for f in filters)
        lines.append(f"Filters: {parts}")
    rc = item.get("row_count")
    tr = item.get("total_rows")
    if rc is not None and tr is not None:
        lines.append(f"Data: {rc:,} of {tr:,} rows")
    elif tr is not None and filters:
        lines.append(f"Total rows: {tr:,} (filtered subset used)")
    return lines


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fig_to_png_bytes(fig) -> bytes:
    """Convert plotly figure to PNG bytes."""
    try:
        return fig.to_image(format="png", width=900, height=500, scale=2)
    except Exception:
        # Fallback: return empty PNG placeholder bytes
        return b""


def _generate_html_report(
    title: str,
    author: str,
    items: list,
    sections: dict,
) -> str:
    """Build an HTML report from session report items."""
    figures_html = ""
    for item in items:
        fig_bytes = _fig_to_png_bytes(item["fig"])
        if fig_bytes:
            b64 = base64.b64encode(fig_bytes).decode("utf-8")
            ctx_lines = _filter_context_lines(item)
            ctx_html = "".join(
                f'<p class="filter-context">{line}</p>' for line in ctx_lines
            )
            figures_html += (
                f'<h3>{item["title"]}</h3>'
                f'{ctx_html}'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;margin-bottom:24px;" alt="{item["title"]}"/>'
            )

    sections_html = ""
    if sections.get("goal"):
        sections_html += """
        <h2>Goal</h2>
        <p>Identify factors associated with early hospital readmission (&lt;30 days)
        in diabetic patients using the Diabetes 130-US Hospitals dataset.</p>"""
    if sections.get("dataset"):
        sections_html += """
        <h2>Dataset</h2>
        <p>~101,766 inpatient encounters from 130 US hospitals (1999–2008).
        50 features including demographics, medications, diagnoses, and HbA1c.</p>"""
    if sections.get("methodology"):
        sections_html += """
        <h2>Methodology</h2>
        <p>ETL pipeline: ingestion (null/outlier removal), normalisation to 3NF SQLite,
        SQL aggregations, Plotly/Matplotlib visualisation, fpdf2/HTML reporting.</p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #1f618d; margin-top: 36px; }}
  h3 {{ color: #2e86c1; margin-bottom: 4px; }}
  img {{ display: block; }}
  .filter-context {{ font-size: 0.85em; color: #666; margin: 2px 0 8px; font-style: italic; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p><strong>Author:</strong> {author}</p>
<hr/>
{sections_html}
<h2>Figures</h2>
{figures_html if figures_html else "<p><em>No figures added.</em></p>"}
<hr/>
<p><em>Generated by Patient Data Analysis App — ALK Kozminski University</em></p>
</body>
</html>"""


_UNICODE_FONT_CANDIDATES: list[tuple[str, str, str]] = [
    # (regular, bold, italic) — tried in order; first complete set wins
    (
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    ),
    (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    ),
]

_CHAR_REPLACEMENTS: dict[str, str] = {
    "≥": ">=", "≤": "<=", "≠": "!=", "∈": "in", "∉": "not in",
    "—": "-", "–": "-", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"', "…": "...",
}


def _sanitize_for_pdf(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents for Helvetica fallback."""
    for src, dst in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _load_unicode_font(pdf) -> str:
    """
    Try to register a Unicode-capable font with fpdf2.
    Returns the registered font name, or empty string if none succeeded.
    """
    from pathlib import Path as _Path
    for regular, bold, italic in _UNICODE_FONT_CANDIDATES:
        if _Path(regular).exists() and _Path(bold).exists() and _Path(italic).exists():
            try:
                pdf.add_font("UniFont", "",  regular)
                pdf.add_font("UniFont", "B", bold)
                pdf.add_font("UniFont", "I", italic)
                return "UniFont"
            except Exception:
                continue
    return ""


def _generate_pdf_report(
    title: str,
    author: str,
    items: list,
    sections: dict,
) -> bytes:
    """Build a PDF report from session report items using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Try Unicode font first; fall back to Helvetica + sanitisation
    unicode_font = _load_unicode_font(pdf)
    base_font = unicode_font if unicode_font else "Helvetica"

    def _t(text: str) -> str:
        """Sanitise text only when the fallback font is in use."""
        return text if unicode_font else _sanitize_for_pdf(text)

    # Title
    pdf.set_font(base_font, "B", 18)
    pdf.set_text_color(26, 82, 118)
    pdf.cell(0, 12, _t(title), align="C")
    pdf.ln(8)
    pdf.set_font(base_font, "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, _t(f"Author: {author}"), align="C")
    pdf.ln(12)

    def h2(text: str) -> None:
        pdf.set_font(base_font, "B", 13)
        pdf.set_text_color(31, 97, 141)
        pdf.cell(0, 8, _t(text))
        pdf.ln(5)
        pdf.set_text_color(50, 50, 50)

    def body(text: str) -> None:
        pdf.set_font(base_font, "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, _t(text))
        pdf.ln(3)

    if sections.get("goal"):
        h2("Goal")
        body(
            "Identify factors associated with early hospital readmission (<30 days) "
            "in diabetic patients using the Diabetes 130-US Hospitals dataset."
        )

    if sections.get("dataset"):
        h2("Dataset")
        body(
            "~101,766 inpatient encounters from 130 US hospitals (1999-2008). "
            "50 features including demographics, medications, diagnoses, and HbA1c."
        )

    if sections.get("methodology"):
        h2("Methodology")
        body(
            "ETL pipeline: ingestion (null/outlier removal), normalisation to 3NF SQLite, "
            "SQL aggregations, Plotly/Matplotlib visualisation, fpdf2/HTML reporting."
        )

    h2("Figures")
    for item in items:
        fig_bytes = _fig_to_png_bytes(item["fig"])
        # Figure title
        pdf.set_font(base_font, "B", 10)
        pdf.set_text_color(46, 134, 193)
        pdf.cell(0, 7, _t(item["title"]))
        pdf.ln(3)
        # Filter context lines (italic, small, grey)
        ctx_lines = _filter_context_lines(item)
        if ctx_lines:
            pdf.set_font(base_font, "I", 8)
            pdf.set_text_color(120, 120, 120)
            for line in ctx_lines:
                pdf.cell(0, 4, _t(line))
                pdf.ln(4)
            pdf.ln(1)
        if fig_bytes:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(fig_bytes)
                tmp_path = tmp.name
            usable_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(tmp_path, x=pdf.l_margin, w=usable_w)
            pdf.ln(6)
            os.unlink(tmp_path)
        else:
            pdf.set_font(base_font, "I", 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 6, _t(f"{item['title']} - image could not be rendered."))
            pdf.ln(4)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

render_sidebar()

try:
    _engine = get_engine(DB_PATH)
except Exception:
    _engine = None

init_state(_engine)

st.title("📄 Reports")
st.markdown(
    "Compose a custom report from figures added in the Analytics page. "
    "Export as HTML or PDF."
)

items = get_report_items()

# ---- Report Items List ----
st.subheader("Report Contents")

if not items:
    st.info("No figures added yet. Go to the **Analytics** page and click 'Add to Report'.")
else:
    for i, item in enumerate(items):
        with st.container(border=True):
            col_title, col_up, col_down, col_remove = st.columns([5, 1, 1, 1])
            with col_title:
                st.markdown(f"**{i + 1}. {item['title']}**")
                for ctx_line in _filter_context_lines(item):
                    st.caption(ctx_line)
            with col_up:
                if st.button("▲", key=f"up_{i}", disabled=(i == 0)):
                    move_report_item(i, -1)
                    st.rerun()
            with col_down:
                if st.button("▼", key=f"dn_{i}", disabled=(i == len(items) - 1)):
                    move_report_item(i, 1)
                    st.rerun()
            with col_remove:
                if st.button("✕", key=f"rm_{i}"):
                    remove_report_item(i)
                    st.rerun()

    if st.button("Clear All", type="secondary"):
        clear_report()
        st.rerun()

st.divider()

# ---- Report Metadata ----
st.subheader("Report Settings")
meta_col1, meta_col2 = st.columns(2)
with meta_col1:
    report_title = st.text_input("Report Title", value="Diabetes Patient Data Analysis")
with meta_col2:
    report_author = st.text_input("Author", value="ALK Student")

st.markdown("**Include sections:**")
sc1, sc2, sc3 = st.columns(3)
with sc1:
    inc_goal = st.checkbox("Goal", value=True)
with sc2:
    inc_dataset = st.checkbox("Dataset", value=True)
with sc3:
    inc_methodology = st.checkbox("Methodology", value=True)

sections = {"goal": inc_goal, "dataset": inc_dataset, "methodology": inc_methodology}

st.divider()

# ---- Preview ----
if items:
    with st.expander("Preview (first figure)", expanded=False):
        st.subheader(items[0]["title"])
        for ctx_line in _filter_context_lines(items[0]):
            st.caption(ctx_line)
        st.plotly_chart(items[0]["fig"], use_container_width=True)

# ---- Export ----
st.subheader("Export")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    if st.button("Generate HTML Report", use_container_width=True):
        with st.spinner("Building HTML..."):
            html_content = _generate_html_report(
                report_title, report_author, items, sections
            )
        st.download_button(
            label="Download HTML",
            data=html_content.encode("utf-8"),
            file_name="report.html",
            mime="text/html",
            use_container_width=True,
        )

with exp_col2:
    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Building PDF..."):
            try:
                pdf_bytes = _generate_pdf_report(
                    report_title, report_author, items, sections
                )
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
