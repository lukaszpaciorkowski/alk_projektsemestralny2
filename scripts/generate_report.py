"""
generate_report.py - Generate HTML and PDF reports from the generic pipeline.

Works with the new app/core/pipeline.py architecture.
Reads figures from outputs/figures/ and dataset metadata from data/data.db.

Usage:
    python3 scripts/generate_report.py
"""

from __future__ import annotations

import base64
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import DB_PATH, get_engine, list_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/report")
FIGURES_DIR = Path("outputs/figures")
DIAGRAMS_DIR = Path("docs/diagrams")


def _img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.suffix.lstrip(".")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{data}"


def _build_datasets_table(datasets: list[dict]) -> str:
    rows = ""
    for ds in datasets:
        type_badge = (
            '<span style="background:#1a73e8;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em">diabetes</span>'
            if ds["dataset_type"] == "diabetes"
            else '<span style="background:#888;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em">generic</span>'
        )
        enrich = (
            '<span style="color:green">done</span>'
            if ds["enrichment_status"] == "done"
            else '<span style="color:#aaa">none</span>'
        )
        rows += f"""<tr>
            <td>{ds['display_name']}</td>
            <td>{type_badge}</td>
            <td>{ds['row_count']:,}</td>
            <td>{ds['col_count']}</td>
            <td>{enrich}</td>
            <td>{ds['uploaded_at'][:10]}</td>
        </tr>"""
    return f"""<table class="data-table">
        <thead><tr>
            <th>Dataset</th><th>Type</th><th>Rows</th><th>Columns</th>
            <th>Enrichment</th><th>Uploaded</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _figure_section(prefix: str, label: str) -> str:
    """Embed an HTML figure via iframe or show placeholder if missing."""
    figs = sorted(FIGURES_DIR.glob(f"{prefix}_*.html"))
    if not figs:
        return ""
    parts = [f"<h3>{label}</h3>"]
    for fig_path in figs:
        name = fig_path.stem.replace(f"{prefix}_", "").replace("_", " ").title()
        # Embed as scrollable iframe
        parts.append(
            f'<p style="font-weight:bold;margin:8px 0 2px">{name}</p>'
            f'<iframe src="../../outputs/figures/{fig_path.name}" '
            f'width="100%" height="480" style="border:1px solid #ddd;border-radius:6px;margin-bottom:12px"></iframe>'
        )
    return "\n".join(parts)


def generate_html_report(datasets: list[dict], output_path: Path) -> None:
    today = date.today().isoformat()
    datasets_table = _build_datasets_table(datasets)

    # Architecture diagram
    arch_img = ""
    arch_png = DIAGRAMS_DIR / "app_architecture.png"
    if arch_png.exists():
        arch_img = f'<img src="data-uri" style="max-width:100%" alt="Architecture">'
        arch_img = f'<img src="{_img_to_base64(arch_png)}" style="max-width:100%;border:1px solid #ddd;border-radius:6px" alt="App Architecture">'

    pipeline_img = ""
    pipeline_png = DIAGRAMS_DIR / "pipeline_flow.png"
    if pipeline_png.exists():
        pipeline_img = f'<img src="{_img_to_base64(pipeline_png)}" style="max-width:100%;border:1px solid #ddd;border-radius:6px" alt="Pipeline Flow">'

    er_img = ""
    er_png = DIAGRAMS_DIR / "er_diagram.png"
    if er_png.exists():
        er_img = f'<img src="{_img_to_base64(er_png)}" style="max-width:100%;border:1px solid #ddd;border-radius:6px" alt="ER Diagram">'

    # Analytics figure sections
    diabetes_section = _figure_section("diabetic_data_diabetes", "Diabetes-Specific Analytics")
    diabetic_generic = _figure_section("diabetic_data_generic", "Diabetic Data - Generic Analytics")
    pima_section = _figure_section("pima_diabetes_generic", "Pima Diabetes Dataset - Generic Analytics")
    heart_section = _figure_section("heart_disease_generic", "Heart Disease Dataset - Generic Analytics")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patient Data Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f7fa; color: #222; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
  h2 {{ color: #1a73e8; margin-top: 40px; }}
  h3 {{ color: #333; margin-top: 24px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 32px; }}
  .section {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.07);
              padding: 28px 32px; margin-bottom: 28px; }}
  .data-table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
  .data-table th {{ background: #1a73e8; color: #fff; padding: 8px 12px; text-align: left; }}
  .data-table td {{ padding: 7px 12px; border-bottom: 1px solid #e8ecf0; }}
  .data-table tr:hover td {{ background: #f0f4ff; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0; }}
  .stat-card {{ background: #f0f4ff; border-radius: 8px; padding: 18px 20px; text-align: center; }}
  .stat-card .value {{ font-size: 2em; font-weight: 700; color: #1a73e8; }}
  .stat-card .label {{ font-size: 0.85em; color: #555; margin-top: 4px; }}
  footer {{ text-align: center; color: #999; font-size: 0.8em; padding: 24px 0 8px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Patient Data Analysis Report</h1>
  <div class="meta">
    Generated: {today} &nbsp;|&nbsp; Project: ALK Kozminski University &nbsp;|&nbsp;
    Stack: Python 3.11 &middot; pandas &middot; plotly &middot; Streamlit &middot; SQLite
  </div>

  <div class="section">
    <h2>1. Project Overview</h2>
    <p>
      This report presents the results of a generic data pipeline that supports any CSV dataset
      with automatic type detection. The pipeline was applied to three patient-related datasets:
    </p>
    <ul>
      <li><strong>Diabetes 130-US Hospitals (1999-2008)</strong> - 101,766 inpatient encounters,
          50 features. UCI / Kaggle dataset. Special diabetes analytics enabled.</li>
      <li><strong>Pima Indians Diabetes</strong> - 768 records, 9 features. Baseline classification dataset.</li>
      <li><strong>Heart Disease (Cleveland UCI)</strong> - 303 records, 14 features. Cardiac risk factors.</li>
    </ul>
    <h3>Imported Datasets</h3>
    {datasets_table}
    <div class="stat-grid">
      <div class="stat-card">
        <div class="value">{sum(d['row_count'] for d in datasets):,}</div>
        <div class="label">Total Records</div>
      </div>
      <div class="stat-card">
        <div class="value">{len(datasets)}</div>
        <div class="label">Datasets</div>
      </div>
      <div class="stat-card">
        <div class="value">14</div>
        <div class="label">Analytics Functions</div>
      </div>
      <div class="stat-card">
        <div class="value">{len(list(FIGURES_DIR.glob('*.html')))}</div>
        <div class="label">Generated Charts</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>2. Architecture</h2>
    <p>
      The generic pipeline follows a single-database, single-import-path architecture.
      All datasets are stored in <code>data/data.db</code> in dynamically named tables.
      The <code>_datasets</code> registry tracks metadata for all imports.
    </p>
    <h3>Application Architecture</h3>
    {arch_img}
    <h3>Pipeline Flow</h3>
    {pipeline_img}
    <h3>Database Schema</h3>
    {er_img}
  </div>

  <div class="section">
    <h2>3. Methodology</h2>
    <h3>Type Detection</h3>
    <p>
      Column signature matching automatically identifies dataset type at upload.
      The diabetes type requires all of: <code>patient_nbr</code>, <code>readmitted</code>,
      <code>metformin</code>, <code>time_in_hospital</code>, <code>num_medications</code>, <code>diag_1</code>.
      All other datasets fall back to the generic type.
    </p>
    <h3>Enrichment (Diabetes only)</h3>
    <p>
      After import, the diabetes dataset is enriched:
      23 medication columns are unpivoted into a long-format <code>_meds</code> table (2.3M rows),
      and three diagnosis columns are decoded via ICD-9 lookup into a <code>_diag</code> table (305K rows).
    </p>
    <h3>Analytics Registry</h3>
    <p>
      All 8 generic analytics functions run on every dataset.
      6 diabetes-specific functions are additional gated by dataset type and enrichment status.
      Each function returns a <code>(DataFrame, Figure | None)</code> tuple.
    </p>
  </div>

  <div class="section">
    <h2>4. Diabetes Dataset Analytics</h2>
    {diabetes_section}
    {diabetic_generic}
  </div>

  <div class="section">
    <h2>5. Additional Datasets</h2>
    {pima_section}
    {heart_section}
  </div>

  <div class="section">
    <h2>6. Conclusions</h2>
    <ul>
      <li>The generic pipeline successfully detected and enriched the diabetes dataset automatically.</li>
      <li>Readmission rates vary significantly by age group, with the [70-80) cohort showing highest risk.</li>
      <li>HbA1c results above normal threshold correlate with higher readmission rates.</li>
      <li>Medication count positively correlates with length of stay, as expected for complex cases.</li>
      <li>The registry-driven analytics engine allows adding new dataset types and functions without
          modifying the UI or core pipeline.</li>
    </ul>
  </div>

  <footer>
    ALK Kozminski University &middot; Patient Data Analysis &middot; {today}
  </footer>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)


def generate_pdf_report(datasets: list[dict], output_path: Path) -> None:
    from fpdf import FPDF

    today = date.today().isoformat()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Patient Data Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {today}  |  ALK Kozminski University", ln=True, align="C")
    pdf.ln(8)

    # Section 1
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Project Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6,
        "This report presents results from a generic data pipeline applied to three patient datasets: "
        "Diabetes 130-US Hospitals (101,766 records), Pima Indians Diabetes (768 records), "
        "and Heart Disease Cleveland (303 records). The pipeline auto-detects dataset type via "
        "column signature matching and supports both generic and diabetes-specific analytics."
    )
    pdf.ln(4)

    # Datasets table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Imported Datasets", ln=True)
    col_widths = [60, 28, 22, 20, 28, 28]
    headers = ["Dataset", "Type", "Rows", "Cols", "Enrichment", "Uploaded"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(26, 115, 232)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    for ds in datasets:
        row = [
            ds["display_name"][:28],
            ds["dataset_type"],
            f"{ds['row_count']:,}",
            str(ds["col_count"]),
            ds["enrichment_status"],
            ds["uploaded_at"][:10],
        ]
        for w, v in zip(col_widths, row):
            pdf.cell(w, 6, v, border=1)
        pdf.ln()
    pdf.ln(6)

    # Section 2
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Architecture", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6,
        "Single SQLite database (data/data.db) stores all datasets in dynamically named ds_* tables. "
        "The _datasets registry table tracks all imports. Enrichment creates _meds and _diag satellite "
        "tables for diabetes datasets. The analytics registry (14 functions: 8 generic + 6 diabetes) "
        "drives the UI without any hardcoded dataset logic."
    )
    pdf.ln(4)

    # Embed architecture PNG
    arch_png = DIAGRAMS_DIR / "app_architecture.png"
    if arch_png.exists():
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Application Architecture", ln=True)
        pdf.image(str(arch_png), w=180)
        pdf.ln(4)

    pipeline_png = DIAGRAMS_DIR / "pipeline_flow.png"
    if pipeline_png.exists():
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Pipeline Flow", ln=True)
        pdf.image(str(pipeline_png), w=180)
        pdf.ln(4)

    er_png = DIAGRAMS_DIR / "er_diagram.png"
    if er_png.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Database Schema (ER Diagram)", ln=True)
        pdf.image(str(er_png), w=180)
        pdf.ln(4)

    # Section 3
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Methodology", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6,
        "Type Detection: Column signature matching identifies dataset type at upload. "
        "Diabetes requires: patient_nbr, readmitted, metformin, time_in_hospital, num_medications, diag_1.\n\n"
        "Enrichment (Diabetes only): 23 medication columns unpivoted to long format (2.34M rows). "
        "Three diagnosis columns decoded via ICD-9 lookup (305K rows).\n\n"
        "Analytics Registry: 8 generic functions run on all datasets; 6 diabetes-specific functions "
        "run only on the diabetes dataset and require enrichment_status=done. "
        "Every function returns (DataFrame, Figure | None)."
    )

    # Section 4 - Key findings
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Key Findings", ln=True)
    pdf.set_font("Helvetica", "", 10)
    findings = [
        "Readmission rates vary by age group: the [70-80) cohort shows the highest risk.",
        "HbA1c results above normal threshold correlate with higher readmission rates.",
        "Medication count positively correlates with length of stay.",
        "Top primary diagnosis codes include 428.0 (heart failure), 250.00 (diabetes mellitus), 276.6 (fluid disorders).",
        "Insulin is the most frequently prescribed medication in the diabetes dataset.",
        "Generic analytics on Pima and Heart Disease datasets reveal expected distributions.",
    ]
    for finding in findings:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, f"- {finding}")
    pdf.ln(4)

    # Section 5
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Conclusions", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6,
        "The generic pipeline successfully handles multiple dataset types from a single code path. "
        "Auto-detection eliminates manual dataset configuration. The analytics registry enables "
        "extensible analysis without UI changes. The architecture is ready for additional dataset types "
        "and analytics functions."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    logger.info("PDF report written to %s", output_path)


def main() -> None:
    engine = get_engine(DB_PATH)
    datasets = list_datasets(engine)

    if not datasets:
        logger.warning("No datasets found in %s - run the pipeline first", DB_PATH)
        sys.exit(1)

    logger.info("Found %d datasets", len(datasets))

    html_path = OUTPUT_DIR / "report.html"
    generate_html_report(datasets, html_path)

    try:
        pdf_path = OUTPUT_DIR / "report.pdf"
        generate_pdf_report(datasets, pdf_path)
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        pdf_path = None

    print(f"\nReports generated:")
    print(f"  HTML: {html_path.resolve()}")
    if pdf_path and pdf_path.exists():
        print(f"  PDF:  {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
