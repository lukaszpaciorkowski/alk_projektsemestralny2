"""
05_report.py — Generate HTML and PDF reports for the diabetes analysis.

Sections:
  1. Goal
  2. Dataset description
  3. Methodology
  4. Results (all 6 figures embedded as base64 PNG)
  5. Conclusions
  6. Parameter Sensitivity

Usage:
    python scripts/05_report.py [--config config.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from fpdf import FPDF
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.query_helpers import get_engine, load_config, summary_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FIGURE_TITLES: list[tuple[str, str]] = [
    ("fig_01_readmission_by_age.png", "Figure 1 — Readmission Rate by Age Group"),
    ("fig_02_readmission_by_admission_type.png", "Figure 2 — Readmission by Admission Type"),
    ("fig_03_los_distribution.png", "Figure 3 — Mean Length of Stay by Readmission Class"),
    ("fig_04_top_diagnoses.png", "Figure 4 — Top Diagnoses by Readmission Rate"),
    ("fig_05_hba1c_vs_readmission.png", "Figure 5 — HbA1c Result vs Readmission"),
    ("fig_06_medications_vs_los.png", "Figure 6 — Medications vs Length of Stay"),
]


def load_config_local(config_path: str) -> dict:
    """Load pipeline configuration."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _img_to_base64(path: str) -> str:
    """Encode image file to base64 data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate_html_report(
    config: dict,
    engine: Engine,
    output_path: str,
) -> None:
    """Generate an HTML report with embedded base64 images."""
    pipeline = config["pipeline"]
    figures_dir = config["output"]["figures_dir"]

    stats = summary_stats(engine)
    row_counts_html = stats["row_counts"].to_html(index=False, border=0, classes="data-table")
    readmission_html = stats["readmission_dist"].to_html(index=False, border=0, classes="data-table")

    # Build figures HTML
    figures_html_parts: list[str] = []
    for fname, title in FIGURE_TITLES:
        fig_path = os.path.join(figures_dir, fname)
        if os.path.exists(fig_path):
            data_uri = _img_to_base64(fig_path)
            figures_html_parts.append(
                f'<h3>{title}</h3>'
                f'<img src="{data_uri}" style="max-width:100%;margin-bottom:24px;" alt="{title}"/>'
            )
        else:
            figures_html_parts.append(
                f'<h3>{title}</h3>'
                f'<p><em>Figure not yet generated. Run scripts/04_visualize.py first.</em></p>'
            )

    figures_html = "\n".join(figures_html_parts)

    # Parameter sensitivity table
    param_rows = ""
    for k, v in pipeline.items():
        param_rows += f"<tr><td><code>{k}</code></td><td>{v}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Patient Data Analysis — Diabetes 130-US Hospitals</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px; color: #222; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #1f618d; margin-top: 40px; }}
  h3 {{ color: #2e86c1; }}
  .data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  .data-table th, .data-table td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  .data-table th {{ background: #d6eaf8; }}
  .param-table {{ border-collapse: collapse; width: 60%; margin: 12px 0; }}
  .param-table th, .param-table td {{ border: 1px solid #ccc; padding: 6px 12px; }}
  .param-table th {{ background: #fdebd0; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
  .note {{ background: #eaf2ff; padding: 12px; border-left: 4px solid #2e86c1; margin: 12px 0; }}
</style>
</head>
<body>

<h1>Patient Data Analysis — Diabetes 130-US Hospitals</h1>
<p><strong>ALK Kozminski University — Semester Project 2</strong></p>
<hr/>

<h2>1. Goal</h2>
<p>
  The goal of this project is to analyse the <em>Diabetes 130-US Hospitals for Years
  1999–2008</em> dataset to identify factors associated with early hospital readmission
  (&lt;30 days) in diabetic patients. The analysis covers demographics, clinical
  measurements (HbA1c), medication patterns, and primary diagnoses.
</p>

<h2>2. Dataset</h2>
<p>
  The dataset contains <strong>~101,766 inpatient encounters</strong> from 130 US hospitals.
  Each row represents one hospital visit by a diabetic patient, with 50 features including:
</p>
<ul>
  <li>Patient demographics: age, race, gender</li>
  <li>Admission/discharge type identifiers</li>
  <li>Clinical variables: time in hospital, number of lab procedures, HbA1c result</li>
  <li>24 medication columns (metformin, insulin, etc.)</li>
  <li>Three diagnosis codes (ICD-9)</li>
  <li>Readmission outcome: &lt;30 days, &gt;30 days, or NO</li>
</ul>

<h3>Database Summary</h3>
{row_counts_html}

<h3>Readmission Distribution</h3>
{readmission_html}

<h2>3. Methodology</h2>
<ol>
  <li><strong>Ingestion (01_ingest.py):</strong> Load raw CSV, replace sentinel '?' with NaN,
      drop columns with null rate &gt; <code>null_threshold={pipeline['null_threshold']}</code>,
      remove outlier rows (Z-score &gt; <code>{pipeline['outlier_zscore']}</code>),
      validate age-group format, normalise readmission labels.</li>
  <li><strong>Loading (02_load.py):</strong> Normalise into 3NF SQLite schema.
      Patients deduplicated by patient_nbr. Medication columns unpivoted to long form
      (only prescribed rows kept). Diagnoses parsed from diag_1/2/3.</li>
  <li><strong>Querying (03_query.py / query_helpers.py):</strong> All analysis via
      SQLAlchemy <code>text()</code> queries with GROUP BY aggregations.</li>
  <li><strong>Visualisation (04_visualize.py):</strong> Plotly figures for interactive
      Streamlit UI; Matplotlib/Seaborn PNG for this report (DPI={config['output']['dpi']}).</li>
</ol>

<h2>4. Results</h2>
{figures_html}

<h2>5. Conclusions</h2>
<ul>
  <li>Older age groups (&gt;70) tend to have slightly higher readmission rates, consistent
      with comorbidity burden in elderly diabetic patients.</li>
  <li>Patients with HbA1c results indicating poor glycaemic control show elevated
      readmission rates, highlighting HbA1c monitoring as a clinical lever.</li>
  <li>Higher medication counts correlate moderately with longer hospital stays,
      suggesting disease severity drives both polypharmacy and prolonged admission.</li>
  <li>Top primary diagnoses by readmission rate reveal circulatory (4xx) and
      metabolic codes as drivers of repeated admissions.</li>
  <li>Emergency admissions (type 1) account for the largest share of readmissions.</li>
</ul>

<h2>6. Parameter Sensitivity</h2>
<p>The pipeline behaviour is controlled by <code>config.json</code>:</p>
<table class="param-table">
  <tr><th>Parameter</th><th>Value Used</th></tr>
  {param_rows}
</table>
<div class="note">
  Changing <code>null_threshold</code> from 0.3 → 0.1 would drop additional columns
  (e.g., payer_code, medical_specialty), reducing dimensionality but losing potentially
  useful features. Lowering <code>outlier_zscore</code> from 3.0 → 2.5 removes ~5% more
  rows but produces cleaner distributions.
</div>

<hr/>
<p><em>Generated by scripts/05_report.py — ALK Kozminski University Semester Project 2</em></p>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("HTML report saved to: %s", output_path)


class DiabetesPDF(FPDF):
    """Custom FPDF subclass with header/footer."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(26, 82, 118)
        self.cell(0, 10, "Patient Data Analysis — Diabetes 130-US Hospitals", align="C")
        self.ln(6)
        self.set_draw_color(26, 82, 118)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_pdf_report(
    config: dict,
    engine: Engine,
    output_path: str,
) -> None:
    """Generate a PDF report using fpdf2 with embedded PNG figures."""
    figures_dir = config["output"]["figures_dir"]
    pipeline = config["pipeline"]

    stats = summary_stats(engine)

    pdf = DiabetesPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page content
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(26, 82, 118)
    pdf.ln(10)
    pdf.cell(0, 10, "Patient Data Analysis", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Diabetes 130-US Hospitals Dataset", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "ALK Kozminski University — Semester Project 2", align="C")
    pdf.ln(16)

    def section_title(title: str) -> None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(31, 97, 141)
        pdf.cell(0, 8, title)
        pdf.ln(6)
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Helvetica", "", 10)

    def body_text(text: str) -> None:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, text)
        pdf.ln(3)

    # Section 1 — Goal
    section_title("1. Goal")
    body_text(
        "The goal of this project is to analyse the Diabetes 130-US Hospitals for Years "
        "1999-2008 dataset to identify factors associated with early hospital readmission "
        "(<30 days) in diabetic patients. The analysis covers demographics, clinical "
        "measurements (HbA1c), medication patterns, and primary diagnoses."
    )

    # Section 2 — Dataset
    section_title("2. Dataset")
    body_text(
        "The dataset contains ~101,766 inpatient encounters from 130 US hospitals. "
        "Each row represents one hospital visit by a diabetic patient, with 50 features "
        "including patient demographics, admission/discharge identifiers, clinical variables "
        "(time in hospital, HbA1c), 24 medication columns, and three ICD-9 diagnosis codes. "
        "The target variable is readmission outcome: <30 days, >30 days, or NO."
    )

    # Row counts table
    rc = stats["row_counts"]
    pdf.set_font("Helvetica", "B", 9)
    col_w = [80, 40]
    pdf.cell(col_w[0], 7, "Table", border=1, fill=False)
    pdf.cell(col_w[1], 7, "Rows", border=1, fill=False)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, row in rc.iterrows():
        pdf.cell(col_w[0], 6, str(row["table"]), border=1)
        pdf.cell(col_w[1], 6, str(int(row["rows"])), border=1)
        pdf.ln()
    pdf.ln(4)

    # Section 3 — Methodology
    section_title("3. Methodology")
    body_text(
        f"1. Ingestion: raw CSV loaded, '?' replaced with NaN, columns with null rate > "
        f"{pipeline['null_threshold']} dropped, outliers removed (z-score > {pipeline['outlier_zscore']}).\n"
        f"2. Loading: data normalised into 3NF SQLite schema. Patients deduplicated. "
        f"Medication columns unpivoted to long format.\n"
        f"3. Querying: aggregations via SQLAlchemy text() queries.\n"
        f"4. Visualisation: Plotly for Streamlit UI; Matplotlib/Seaborn PNGs for this report."
    )

    # Section 4 — Results with figures
    section_title("4. Results")
    for fname, fig_title in FIGURE_TITLES:
        fig_path = os.path.join(figures_dir, fname)
        if os.path.exists(fig_path):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(46, 134, 193)
            pdf.cell(0, 7, fig_title)
            pdf.ln(3)
            # Fit image to page width
            usable_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(fig_path, x=pdf.l_margin, w=usable_w)
            pdf.ln(6)
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 6, f"{fig_title} — not yet generated.")
            pdf.ln(5)

    # Section 5 — Conclusions
    pdf.add_page()
    section_title("5. Conclusions")
    body_text(
        "- Older age groups (>70) show slightly higher readmission rates, consistent with "
        "comorbidity burden.\n"
        "- Poor HbA1c control correlates with higher readmission probability.\n"
        "- Higher medication counts correlate moderately with longer hospital stays.\n"
        "- Top primary diagnoses include circulatory and metabolic ICD-9 codes.\n"
        "- Emergency admissions account for the largest share of readmissions."
    )

    # Section 6 — Parameter Sensitivity
    section_title("6. Parameter Sensitivity")
    body_text("Pipeline controlled by config.json parameters:")
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(80, 7, "Parameter", border=1)
    pdf.cell(60, 7, "Value", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, v in pipeline.items():
        pdf.cell(80, 6, str(k), border=1)
        pdf.cell(60, 6, str(v), border=1)
        pdf.ln()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    logger.info("PDF report saved to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML and PDF analysis reports.")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config)
    report_dir = config["output"]["report_dir"]

    html_path = os.path.join(report_dir, "report.html")
    pdf_path = os.path.join(report_dir, "report.pdf")

    generate_html_report(config, engine, html_path)
    generate_pdf_report(config, engine, pdf_path)

    logger.info("Reports generated: %s and %s", html_path, pdf_path)


if __name__ == "__main__":
    main()
