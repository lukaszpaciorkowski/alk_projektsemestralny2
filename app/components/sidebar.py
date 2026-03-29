"""
sidebar.py — Shared sidebar component for the Streamlit app.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.json"


def _load_config() -> dict:
    """Load config.json, return empty dict on error."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _get_db_status(config: dict) -> tuple[bool, int, str]:
    """
    Check database connectivity and row count.

    Returns:
        Tuple of (is_connected: bool, row_count: int, db_path: str).
    """
    db_path = config.get("database", {}).get("path", "data/diabetes.db")
    if not Path(db_path).exists():
        return False, 0, db_path

    try:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM admissions")).fetchone()
            count = int(row[0]) if row else 0
        return True, count, db_path
    except Exception as exc:
        logger.warning("DB status check failed: %s", exc)
        return False, 0, db_path


def render_sidebar() -> None:
    """Render the sidebar with database status and project info."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/"
            "ALK_logo.svg/120px-ALK_logo.svg.png",
            width=80,
        )
        st.markdown("### Patient Data Analysis")
        st.markdown("*Diabetes 130-US Hospitals*")
        st.divider()

        config = _load_config()
        connected, row_count, db_path = _get_db_status(config)

        st.markdown("#### Database Status")
        if connected:
            st.success(f"Connected — {row_count:,} encounters")
        else:
            st.error("Not connected")
            st.caption(f"Expected at: `{db_path}`")
            st.caption("Run the pipeline on the **Data Sources** page.")

        st.divider()
        st.markdown("#### Pipeline Config")
        pipeline = config.get("pipeline", {})
        if pipeline:
            st.caption(f"null_threshold: **{pipeline.get('null_threshold', '—')}**")
            st.caption(f"outlier_zscore: **{pipeline.get('outlier_zscore', '—')}**")
            st.caption(f"top_n_diagnoses: **{pipeline.get('top_n_diagnoses', '—')}**")
            st.caption(f"palette: **{pipeline.get('palette', '—')}**")
        else:
            st.caption("config.json not found.")

        st.divider()
        st.markdown(
            "<small>ALK Kozminski University<br/>Semester Project 2</small>",
            unsafe_allow_html=True,
        )
