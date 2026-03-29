# ALK Projekt Semestralny 2

Exam project for studies at Kozminski University (ALK).

## Project Context
- **Course:** Semester Project 2
- **University:** Kozminski University (ALK), Warsaw
- **Repo:** https://github.com/lukaszpaciorkowski/alk_projektsemestralny2
- **Topic:** Patient Data Analysis — Diabetes 130-US Hospitals Dataset

## Stack
- **Language:** Python 3.11+
- **Database:** SQLite via SQLAlchemy 2.0
- **UI:** Streamlit 1.35
- **Data Processing:** pandas, numpy, scipy
- **Visualization:** plotly, matplotlib, seaborn
- **Reporting:** fpdf2 (PDF), HTML
- **Testing:** pytest

## Dataset
UCI / Kaggle: Diabetes 130-US Hospitals for Years 1999–2008
- ~100,000 inpatient encounters
- 50 features: demographics, diagnoses, medications, lab results
- Source: `data/raw/diabetic_data.csv` (download from Kaggle)

## Key Commands
```bash
# Install dependencies
make install

# Download dataset (requires kaggle CLI configured)
kaggle datasets download -d jimschacko/airlines-dataset-to-predict-a-delay -p data/raw/

# Run full pipeline
make pipeline

# Generate Mermaid diagrams (requires mmdc)
make diagrams

# Start Streamlit app
make run

# Run tests
make test
```

## Quick Start
1. `pip install -r requirements.txt`
2. Place `diabetic_data.csv` in `data/raw/`
3. `python scripts/01_ingest.py`
4. `python scripts/02_load.py`
5. `streamlit run app/main.py`

## Project Structure
- `scripts/` — numbered ETL pipeline scripts
- `app/` — Streamlit multi-page application
- `database/` — SQL schema and DB creation
- `docs/diagrams/` — Mermaid architecture diagrams
- `tests/` — pytest unit tests
- `outputs/` — generated figures and reports
