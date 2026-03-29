.PHONY: diagrams run test install pipeline

install:
	pip install -r requirements.txt

diagrams:
	mmdc -i docs/diagrams/er_diagram.mmd -o docs/diagrams/er_diagram.png -t neutral
	mmdc -i docs/diagrams/pipeline_flow.mmd -o docs/diagrams/pipeline_flow.png -t neutral
	mmdc -i docs/diagrams/app_architecture.mmd -o docs/diagrams/app_architecture.png -t neutral

run:
	streamlit run app/main.py

test:
	pytest tests/ -v

pipeline:
	python scripts/01_ingest.py
	python scripts/02_load.py
	python scripts/03_query.py
	python scripts/04_visualize.py
	python scripts/05_report.py
