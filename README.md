# Patient Data Analysis Platform

**Course:** Projekt Egzaminacyjny II · Kozminski University (ALK), Warsaw
**Stack:** Python 3.11+ · Streamlit · SQLAlchemy · SQLite · pandas · scikit-learn · plotly · fpdf2

A generic data analysis platform built around the **Diabetes 130-US Hospitals** dataset. Upload any CSV, explore it interactively, run 24 analytical methods, build dashboards, generate reports, and view live-generated architecture diagrams — all from a six-page Streamlit app.

---

## Features

| Feature | Details |
|---------|---------|
| **Generic CSV pipeline** | Upload any CSV → auto-type detection → SQLite storage → analysis |
| **9 datasets loaded** | Diabetes, Heart Disease (Cleveland), PIMA Indians, Eurostat hospital/mortality, OWID COVID-19, OWID causes of death, World Mortality |
| **24 analytical methods** | 17 generic + 6 diabetes-specific + geographic summary |
| **14 chart types** | Bar, Line, Scatter, Box, Histogram, Heatmap, Choropleth, Pie, Donut, Multi-Line, Area (Stacked), 3D Scatter, Sunburst, Treemap |
| **Interactive filtering** | Column-level filters (=, !=, ≥, ≤, IN, LIKE, IS NULL) on every page |
| **PDF/HTML reports** | Compose figures from any analysis; filter context embedded; Unicode font support |
| **Persistent reports** | Report queue stored in SQLite `_reports` table — survives page refreshes |
| **Live architecture diagrams** | ER diagram, pipeline flowchart, app architecture — generated from live DB/code via Mermaid |
| **214 passing tests** | pytest, in-memory SQLite fixtures |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app/main.py --server.address 0.0.0.0

# 3. Import data
#    - Open Data Sources → upload any CSV
#    - Or place data/raw/diabetic_data.csv and run:
python scripts/01_ingest.py
python scripts/02_load.py
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| UI | Streamlit 1.50+ |
| Database | SQLite via SQLAlchemy 2.0 |
| Data | pandas, numpy |
| Statistics | scipy, scikit-learn |
| Visualisation | plotly, matplotlib, seaborn |
| Reporting | fpdf2 (PDF), HTML with base64 images |
| Diagrams | Mermaid CLI (`mmdc`) |
| Testing | pytest |

---

## Application Pages

| Page | Purpose |
|------|---------|
| **📂 Data Sources** | Upload CSV, trigger import pipeline, view import log, run enrichment |
| **🔍 Data Exploration** | Browse tables with filters, column stats, mini histograms |
| **📈 Dashboards** | Build interactive charts from any dataset — 14 chart types |
| **📊 Analytics** | 24 analytical methods with parameter controls and filter support |
| **📄 Reports** | Compose figures into PDF/HTML reports with filter context |
| **📚 Documentation** | Live architecture diagrams + full analytics method reference |

---

## Analytics Methods

### 🌐 Generic (work on any dataset)

| Method | Description |
|--------|-------------|
| Descriptive Statistics | pandas `.describe(include='all')` |
| Correlation Matrix | Pearson / Spearman / Kendall heatmap |
| Value Counts | Frequency table for categorical columns |
| Group By / Aggregate | Group + aggregate with mean/sum/count/min/max/median |
| Cross-tabulation | Pivot table of two categorical columns |
| Distribution | Histogram with KDE overlay |
| Null Analysis | Bar chart of null % per column |
| Data Types | Column type summary with cardinality |
| Principal Component Analysis | PCA biplot with loading arrows, component pair selector |
| Outlier Detection | Single-variable (histogram + strip) or two-variable (scatter); Z-score / IQR / Isolation Forest |
| Chi-Square Test | Test of independence with Cramér's V |
| Two-Group Comparison | T-test or Mann-Whitney U |
| Multi-Group Comparison | ANOVA or Kruskal-Wallis (auto-selected) |
| Normality Test | Shapiro-Wilk (n≤5000) or KS test |
| K-Means Clustering | Clustering with silhouette score and elbow plot |
| Feature Importance | Random Forest feature importance |
| Time Series Trend | Line chart with rolling mean overlay |
| Geographic Summary | Choropleth world map by aggregated value |

### 🧬 Diabetes-Specific

| Method | Description |
|--------|-------------|
| Readmission Rate by Group | Readmission % by any categorical column |
| HbA1c vs Readmission | Readmission rate by HbA1c test result |
| Top Diagnoses by Readmission | Top N ICD-9 diagnoses ranked by readmission rate *(requires enrichment)* |
| Medication Frequency | Most prescribed medications *(requires enrichment)* |
| Length of Stay by Readmission | Mean/median/min/max LOS by readmission class |
| Medications vs LOS | Scatter: number of medications vs mean length of stay |

---

## Chart Types (Dashboards)

Bar · Line · Scatter · Box · Histogram · Heatmap · Choropleth Map · Pie · Donut · Multi-Line · Area (Stacked) · 3D Scatter · Sunburst · Treemap

---

## Primary Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**

- ~101,766 inpatient encounters from 130 US hospitals
- 50 features: demographics, diagnoses, medications, lab results, readmission status
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) / Kaggle (`kaggle datasets download -d brandao/diabetes`)
- Place at `data/raw/diabetic_data.csv`

---

## Project Structure

```
alk_projektsemestralny2/
├── app/
│   ├── main.py                    # Streamlit entry point
│   ├── state.py                   # Session state + DB-persisted report helpers
│   ├── views/
│   │   ├── 1_data_sources.py
│   │   ├── 2_exploration.py
│   │   ├── 3_dashboards.py
│   │   ├── 4_analytics.py
│   │   ├── 5_reports.py
│   │   └── 6_documentation.py
│   ├── components/
│   │   ├── chart_builder.py       # 14 chart types
│   │   ├── filter_panel.py        # Reusable column-level filter UI
│   │   └── sidebar.py
│   └── core/
│       ├── pipeline.py            # CSV import, type detection, SQLite storage
│       ├── query.py               # Parameterised queries, Filter dataclass
│       ├── registry.py            # Analytics function registry (24 methods)
│       ├── reports.py             # DB persistence for report items
│       ├── introspect.py          # Live Mermaid diagram generation
│       ├── type_detector.py       # Dataset type inference
│       └── analytics/
│           ├── generic.py         # 18 generic analytics functions
│           └── diabetes.py        # 6 diabetes-specific functions
├── database/
│   ├── schema.sql                 # SQLite DDL (normalised 3NF schema + _datasets + _reports)
│   └── create_db.py
├── scripts/                       # Numbered ETL pipeline scripts
├── tests/
│   └── test_integration.py        # 214 tests
├── docs/diagrams/                 # Mermaid .mmd + rendered .png files
├── data/raw/                      # Source CSVs (gitignored)
├── config.json                    # Pipeline parameters
├── requirements.txt
└── Makefile
```

---

## Architecture Diagrams

Three diagrams are generated live from the database schema and codebase via `app/core/introspect.py`:

- **ER Diagram** — all tables, columns (up to 10 per table), and logical FK relationships
- **Pipeline Flow** — real dataset names and row counts from `_datasets`
- **App Architecture** — pages, components, core modules with live function counts

Rendered to PNG at 4800×3600 effective resolution via `mmdc` (Mermaid CLI).
Saved to `docs/diagrams/` — open the **Documentation** page and click **Regenerate Diagrams**.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Using the venv
.venv/bin/python -m pytest tests/ -q
```

**214 tests** covering: CSV ingestion validation, SQLite pipeline, all 24 analytics functions, chart builders, filter logic, report persistence, and outlier detection modes.

---

## Configuration (`config.json`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `null_threshold` | `0.84` | Drop columns with >84% missing values |
| `outlier_zscore` | `3.0` | Remove rows where any numeric field exceeds z=3.0 |
| `readmission_binary` | `false` | Merge `<30`+`>30` into one positive class |
| `top_n_diagnoses` | `10` | Default diagnoses shown in diabetes analyses |
