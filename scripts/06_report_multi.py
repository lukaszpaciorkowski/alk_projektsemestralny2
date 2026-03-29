"""
06_report_multi.py - Generate a comprehensive multi-dataset PDF report.

Includes:
  1. Diabetes 130-US Hospitals analysis (6 figures)
  2. Heart Disease Cleveland (describe, correlation, distributions)
  3. Pima Indians Diabetes (describe, correlation, distributions)
  4. datasets_registry summary table

Usage:
    python scripts/06_report_multi.py [--config config.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from fpdf import FPDF
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_engine(config: dict):
    db_path = config["database"]["path"]
    return create_engine(f"sqlite:///{db_path}", echo=False)


class MultiReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(26, 82, 118)
        self.cell(0, 9, "Patient Data Analysis - Multi-Dataset Report", align="C")
        self.ln(5)
        self.set_draw_color(26, 82, 118)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} - ALK Kozminski University", align="C")


def section(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(31, 97, 141)
    pdf.cell(0, 8, title)
    pdf.ln(5)
    pdf.set_text_color(50, 50, 50)


def body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5, text)
    pdf.ln(3)


def add_figure(pdf: FPDF, fig_path: str, caption: str) -> None:
    if not os.path.exists(fig_path):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, f"{caption} - not found.")
        pdf.ln(4)
        return
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(46, 134, 193)
    pdf.cell(0, 7, caption)
    pdf.ln(2)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.image(fig_path, x=pdf.l_margin, w=usable_w)
    pdf.ln(6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config)
    figures_dir = config["output"]["figures_dir"]
    report_dir = config["output"]["report_dir"]
    pipeline = config["pipeline"]

    pdf = MultiReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title page ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(26, 82, 118)
    pdf.ln(10)
    pdf.cell(0, 12, "Patient Data Analysis", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Multi-Dataset Pipeline Report", align="C")
    pdf.ln(7)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 7, "ALK Kozminski University - Semester Project 2", align="C")
    pdf.ln(16)

    # ── Registry summary ───────────────────────────────────────────────────
    section(pdf, "1. Imported Datasets")
    body(pdf, "The following datasets were imported through the generic pipeline (app.core.pipeline.import_csv):")

    try:
        with engine.connect() as conn:
            reg_df = pd.read_sql(
                text("SELECT dataset_name, dataset_type, row_count, col_count, imported_at FROM datasets_registry ORDER BY imported_at"),
                conn,
            )
        pdf.set_font("Helvetica", "B", 9)
        col_widths = [55, 22, 22, 22, 60]
        headers = ["Dataset Name", "Type", "Rows", "Cols", "Imported At"]
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for _, row in reg_df.iterrows():
            pdf.cell(col_widths[0], 6, str(row["dataset_name"]), border=1)
            pdf.cell(col_widths[1], 6, str(row["dataset_type"]), border=1)
            pdf.cell(col_widths[2], 6, f"{int(row['row_count']):,}", border=1)
            pdf.cell(col_widths[3], 6, str(int(row["col_count"])), border=1)
            pdf.cell(col_widths[4], 6, str(row["imported_at"])[:19], border=1)
            pdf.ln()
        pdf.ln(6)
    except Exception as exc:
        body(pdf, f"Could not load registry: {exc}")

    # ── Section 2: Diabetes analysis ───────────────────────────────────────
    pdf.add_page()
    section(pdf, "2. Diabetes 130-US Hospitals Analysis")
    body(pdf,
         f"Dataset: ~89,991 inpatient encounters, 50 features. "
         f"Pipeline: null_threshold={pipeline['null_threshold']}, outlier_zscore={pipeline['outlier_zscore']}. "
         f"Loaded into 3NF SQLite schema (patients, admissions, medications, diagnoses).")

    diabetes_figs = [
        ("fig_01_readmission_by_age.png", "Figure 1 - Readmission Rate by Age Group"),
        ("fig_02_readmission_by_admission_type.png", "Figure 2 - Readmission by Admission Type"),
        ("fig_03_los_distribution.png", "Figure 3 - Mean Length of Stay by Readmission Class"),
        ("fig_04_top_diagnoses.png", "Figure 4 - Top Diagnoses by Readmission Rate"),
        ("fig_05_hba1c_vs_readmission.png", "Figure 5 - HbA1c Result vs Readmission"),
        ("fig_06_medications_vs_los.png", "Figure 6 - Medications vs Length of Stay"),
    ]
    for fname, caption in diabetes_figs:
        add_figure(pdf, os.path.join(figures_dir, fname), caption)

    # ── Section 3: Heart Disease ───────────────────────────────────────────
    pdf.add_page()
    section(pdf, "3. Heart Disease Cleveland Dataset (Generic)")
    body(pdf,
         "UCI Cleveland Heart Disease dataset. 303 patients, 14 features including age, sex, "
         "chest pain type, resting BP, cholesterol, and target (presence of heart disease). "
         "Imported as generic type - stored as flat table generic_heart_disease_cleveland.")

    heart_figs = [
        ("heart_disease_cleveland_describe.png", "Descriptive Statistics"),
        ("heart_disease_cleveland_correlation.png", "Correlation Matrix"),
        ("heart_disease_cleveland_distributions.png", "Feature Distributions"),
    ]
    for fname, caption in heart_figs:
        add_figure(pdf, os.path.join(figures_dir, fname), f"Heart Disease - {caption}")

    # ── Section 4: Pima Indians ────────────────────────────────────────────
    pdf.add_page()
    section(pdf, "4. Pima Indians Diabetes Dataset (Generic)")
    body(pdf,
         "Pima Indians Diabetes dataset from NIDDK. 768 female patients of Pima Indian heritage, "
         "9 features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
         "DiabetesPedigreeFunction, Age, Outcome. "
         "Imported as generic type - stored as flat table generic_pima_indians_diabetes.")

    pima_figs = [
        ("pima_indians_diabetes_describe.png", "Descriptive Statistics"),
        ("pima_indians_diabetes_correlation.png", "Correlation Matrix"),
        ("pima_indians_diabetes_distributions.png", "Feature Distributions"),
    ]
    for fname, caption in pima_figs:
        add_figure(pdf, os.path.join(figures_dir, fname), f"Pima Indians - {caption}")

    # ── Section 5: Methodology ────────────────────────────────────────────
    pdf.add_page()
    section(pdf, "5. Methodology - Generic Pipeline")
    body(pdf,
         "All datasets are imported through app.core.pipeline.import_csv() which:\n"
         "1. Reads any CSV file\n"
         "2. Auto-detects type: 'diabetes' if >=4/6 UCI signature columns found, else 'generic'\n"
         "3. For diabetes: runs full ETL (null drop, outlier removal, 3NF schema load)\n"
         "4. For generic: replaces '?' sentinels with NaN, stores as flat SQLite table\n"
         "5. Registers dataset in datasets_registry table\n\n"
         "Enrichment (enrich_dataset()) then runs:\n"
         "- Diabetes: 6 standard figures (readmission, HbA1c, LOS, diagnoses, medications)\n"
         "- Generic: descriptive statistics table, correlation heatmap, feature distributions")

    section(pdf, "6. Conclusions")
    body(pdf,
         "- The generic pipeline successfully handles heterogeneous patient datasets without\n"
         "  schema changes or code modifications.\n"
         "- Diabetes dataset: older patients (>70) show higher readmission; poor HbA1c control\n"
         "  correlates with readmission; emergency admissions dominate readmission counts.\n"
         "- Heart Disease: strong correlations between age, max heart rate, and ST depression.\n"
         "- Pima Indians: Glucose is the strongest predictor of diabetes outcome (correlation ~0.47).\n"
         "- The datasets_registry provides a unified audit trail across all imports.")

    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "report_multi.pdf")
    pdf.output(output_path)
    logger.info("Multi-dataset PDF report saved to: %s", output_path)
    print(f"\nPDF report: {output_path}")


if __name__ == "__main__":
    main()
