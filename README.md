# Patient Data Analysis — ALK Projekt Egzaminacyjny II

**Course:** Projekt Egzaminacyjny II · Kozminski University (ALK), Warsaw
**Dataset:** Diabetes 130-US Hospitals 1999–2008 (~100 000 patient encounters)
**Stack:** Python · SQLite · SQLAlchemy · pandas · plotly · Streamlit · fpdf2

---

## Project Goal

Analyze what patient and treatment factors are associated with **30-day hospital readmission** using a real-world clinical dataset of 100 000 inpatient encounters from 130 US hospitals.

The project demonstrates:
- Relational database design (3NF schema, 7 tables, indexed FKs)
- Reproducible ETL pipeline with documented parameters
- Statistical group analysis and 6 publication-quality visualizations
- Interactive Streamlit UI with PDF report export

---

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**

| Property | Value |
|----------|-------|
| Source | UCI ML Repository / Kaggle |
| Rows | ~100 000 patient encounters |
| Columns | 50+ (demographics, labs, medications, diagnoses, readmission) |
| License | CC0 Public Domain |

Download from:
- UCI: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Kaggle: `kaggle datasets download -d brandao/diabetes`

Place the file at `data/raw/diabetic_data.csv` before running the pipeline.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
#    data/raw/diabetic_data.csv

# 3. Run the full pipeline
python scripts/01_ingest.py    # validate + clean CSV
python scripts/02_load.py      # load into SQLite (3NF schema)
python scripts/03_query.py     # run SQL analyses
python scripts/04_visualize.py # generate all 6 figures
python scripts/05_report.py    # export HTML + PDF report

# 4. Launch the Streamlit app
streamlit run app/main.py

# Or run everything with make
make install
make pipeline
make run
```

---

## Pipeline

```
[1] INGEST          [2] LOAD              [3] QUERY
01_ingest.py   →    02_load.py       →    03_query.py
CSV validation      SQLite 3NF schema     GROUP BY / JOIN
null audit          medication unpivot    WINDOW functions
outlier removal     ICD-9 lookup          parameterized SQL

[4] VISUALIZE       [5] REPORT
04_visualize.py →   05_report.py
6 plotly figures    HTML + PDF
matplotlib/seaborn  fpdf2 embedding
outputs/figures/    outputs/report/
```

### Pipeline Parameters (`config.json`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `null_threshold` | `0.3` | Drop columns with >30% missing values |
| `outlier_zscore` | `3.0` | Remove rows where any numeric field exceeds z=3.0 |
| `age_bins` | `[0,30,50,70,100]` | Age group boundaries for readmission analysis |
| `readmission_binary` | `false` | Merge `<30`+`>30` into one positive class |
| `top_n_diagnoses` | `10` | Number of diagnoses shown in bar chart |
| `palette` | `"viridis"` | Matplotlib/seaborn color palette |

All parameters can be overridden via `--config path/to/config.json` on each script.

---

## Database Schema

Seven normalized tables (3NF):

```
patients            admissions              admission_types
-----------         ---------------         ---------------
patient_id (PK)     encounter_id (PK)       id (PK)
race                patient_id (FK)         description
gender              admission_type_id (FK)
age_group           discharge_type_id (FK)  discharge_types
                    time_in_hospital        ---------------
                    num_lab_procedures      id (PK)
                    num_medications         description
                    hba1c_result
                    readmission             diagnoses_lookup
                                            ----------------
medications                                 icd9_code (PK)
-----------                                 description
id (PK)             diagnosis_encounters    category
encounter_id (FK)   --------------------
drug_name           id (PK)
change_indicator    encounter_id (FK)
                    icd9_code (FK)
                    diagnosis_position
```

ER diagram source: `docs/diagrams/er_diagram.mmd`
DDL: `database/schema.sql`

---

## Streamlit UI

Five views, accessible from the sidebar:

| View | Purpose |
|------|---------|
| **📂 Data Sources** | Upload CSV, trigger ingest pipeline, view import log |
| **🔍 Data Exploration** | Browse DB tables, column histograms, summary stats |
| **📊 Analytics** | Filter by age/race/admission type, run 4 analysis models, add charts to report |
| **📄 Reports** | Assemble selected charts into a PDF or HTML report |
| **📐 Architecture** | View ER diagram, pipeline flow, and app architecture (Mermaid diagrams) |

Run: `streamlit run app/main.py`

---

## Analysis & Visualizations

Six figures produced by `04_visualize.py` and saved to `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `fig_01_readmission_by_age.png` | Readmission rate (%) by age group — bar chart |
| `fig_02_readmission_by_admission_type.png` | Readmission rate by admission type — stacked bar |
| `fig_03_los_distribution.png` | Length-of-stay distribution by readmission class — box plot |
| `fig_04_top_diagnoses.png` | Top 10 primary diagnoses by readmission rate — horizontal bar |
| `fig_05_hba1c_vs_readmission.png` | HbA1c test result vs. readmission — grouped bar |
| `fig_06_medications_vs_los.png` | Number of medications vs. time in hospital — scatter + regression |

---

## Repository Structure

```
alk_projektsemestralny2/
├── README.md
├── CLAUDE.md                  # AI assistant instructions
├── requirements.txt
├── config.json                # Pipeline parameters
├── Makefile                   # make install / pipeline / run / test / diagrams
├── data/
│   ├── raw/                   # Place diabetic_data.csv here (gitignored)
│   └── processed/             # Cleaned CSV output (gitignored)
├── database/
│   ├── schema.sql             # DDL — all 7 tables + indexes
│   └── create_db.py           # Schema creation script
├── scripts/
│   ├── 01_ingest.py           # Validate + clean raw CSV
│   ├── 02_load.py             # Load into SQLite (3NF, unpivot meds)
│   ├── 03_query.py            # Parameterized SQL analysis functions
│   ├── 04_visualize.py        # Generate 6 figures (plotly + seaborn)
│   ├── 05_report.py           # HTML + PDF report export
│   ├── ingest_helpers.py      # Validation logic (imported by tests)
│   ├── query_helpers.py       # Query functions (imported by UI)
│   └── visualize_helpers.py   # Chart helpers (imported by UI)
├── app/
│   ├── main.py                # Streamlit entry point (st.navigation)
│   ├── pages/
│   │   ├── 1_data_sources.py
│   │   ├── 2_exploration.py
│   │   ├── 3_analytics.py
│   │   ├── 4_reports.py
│   │   └── 5_architecture.py
│   ├── components/
│   │   ├── sidebar.py         # DB status indicator
│   │   └── charts.py          # Shared Plotly builders
│   └── state.py               # st.session_state helpers
├── notebooks/
│   └── analysis.ipynb         # Interactive analysis notebook
├── outputs/
│   ├── figures/               # PNG exports (gitignored)
│   └── report/                # HTML/PDF report (gitignored)
├── tests/
│   ├── test_validation.py     # 20 unit tests — ingest/validation logic
│   └── test_query.py          # 21 unit tests — SQL query functions
└── docs/
    ├── parameter_analysis.md  # Parameter sensitivity documentation
    └── diagrams/
        ├── er_diagram.mmd
        ├── pipeline_flow.mmd
        └── app_architecture.mmd
```

---

## Tests

```bash
make test
# or
pytest tests/ -v
```

41 unit tests covering:
- Null threshold column dropping
- Z-score outlier removal
- Age group validation
- Readmission binary conversion
- SQL query functions (in-memory SQLite)

---

## Architecture Diagrams

Mermaid source files in `docs/diagrams/`. Render to PNG with:

```bash
# Requires Node.js + mermaid-cli: npm install -g @mermaid-js/mermaid-cli
make diagrams
```

Pre-rendered PNGs are committed so the app works without Node.js installed.

---

## Report

After running the full pipeline, the report is at `outputs/report/report.html`.
Open it in any browser or use the **Reports** view in the Streamlit app to export PDF.
