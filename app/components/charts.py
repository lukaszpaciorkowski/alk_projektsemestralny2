"""
charts.py — Shared Plotly figure builder wrappers for the Streamlit app.

These functions wrap 04_visualize.py figure generators, injecting config
values automatically.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import plotly.graph_objects as go
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.visualize_helpers import (
    fig_01_readmission_by_age,
    fig_02_readmission_by_admission_type,
    fig_03_los_distribution,
    fig_04_top_diagnoses,
    fig_05_hba1c_vs_readmission,
    fig_06_medications_vs_los,
    generate_all_figures,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.json"


def _load_config() -> dict:
    """Load config.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_readmission_by_age(engine: Engine) -> go.Figure:
    """Build Fig 1: Readmission Rate by Age Group."""
    config = _load_config()
    return fig_01_readmission_by_age(
        engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
    )


def build_readmission_by_admission_type(engine: Engine) -> go.Figure:
    """Build Fig 2: Readmission by Admission Type."""
    config = _load_config()
    return fig_02_readmission_by_admission_type(
        engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
    )


def build_los_distribution(engine: Engine) -> go.Figure:
    """Build Fig 3: Length of Stay Distribution."""
    config = _load_config()
    return fig_03_los_distribution(
        engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
    )


def build_top_diagnoses(engine: Engine) -> go.Figure:
    """Build Fig 4: Top Diagnoses by Readmission Rate."""
    config = _load_config()
    return fig_04_top_diagnoses(
        engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
        top_n=config["pipeline"]["top_n_diagnoses"],
    )


def build_hba1c_vs_readmission(engine: Engine) -> go.Figure:
    """Build Fig 5: HbA1c vs Readmission."""
    config = _load_config()
    return fig_05_hba1c_vs_readmission(
        engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
    )


def build_medications_vs_los(engine: Engine) -> go.Figure:
    """Build Fig 6: Medications vs Length of Stay."""
    config = _load_config()
    return fig_06_medications_vs_los(
        engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
    )


def build_all_figures(engine: Engine) -> dict[str, go.Figure]:
    """Generate all 6 figures at once."""
    config = _load_config()
    return generate_all_figures(
        engine=engine,
        figures_dir=config["output"]["figures_dir"],
        dpi=config["output"]["dpi"],
        palette=config["pipeline"]["palette"],
        top_n=config["pipeline"]["top_n_diagnoses"],
    )
