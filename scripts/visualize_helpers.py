"""
visualize_helpers.py — Re-exports all figure functions from 04_visualize.py.

This module provides a stable import path for the Streamlit app and other
scripts that need figure generation functions.
"""

from __future__ import annotations

# We import by loading the module via importlib to handle the numeric prefix
import importlib.util
import sys
from pathlib import Path

_mod_path = Path(__file__).parent / "04_visualize.py"
_spec = importlib.util.spec_from_file_location("_04_visualize", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fig_01_readmission_by_age = _mod.fig_01_readmission_by_age
fig_02_readmission_by_admission_type = _mod.fig_02_readmission_by_admission_type
fig_03_los_distribution = _mod.fig_03_los_distribution
fig_04_top_diagnoses = _mod.fig_04_top_diagnoses
fig_05_hba1c_vs_readmission = _mod.fig_05_hba1c_vs_readmission
fig_06_medications_vs_los = _mod.fig_06_medications_vs_los
generate_all_figures = _mod.generate_all_figures

__all__ = [
    "fig_01_readmission_by_age",
    "fig_02_readmission_by_admission_type",
    "fig_03_los_distribution",
    "fig_04_top_diagnoses",
    "fig_05_hba1c_vs_readmission",
    "fig_06_medications_vs_los",
    "generate_all_figures",
]
