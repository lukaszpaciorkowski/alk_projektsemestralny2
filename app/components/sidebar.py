"""
sidebar.py — Shared sidebar component for the Streamlit app.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from sqlalchemy import text

from app.core.pipeline import DB_PATH, get_engine
from app.core.type_detector import dataset_type_icon, dataset_type_label

logger = logging.getLogger(__name__)


def _get_db_status() -> tuple[bool, int, str, str | None]:
    """
    Check database connectivity.

    Returns:
        (is_connected, dataset_count, db_path, active_dataset_display)
    """
    db_file = Path(DB_PATH)
    if not db_file.exists():
        return False, 0, DB_PATH, None

    try:
        engine = get_engine(DB_PATH)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM _datasets")
            ).fetchone()
            count = int(row[0]) if row else 0
        return True, count, DB_PATH, None
    except Exception as exc:
        logger.warning("DB status check failed: %s", exc)
        return False, 0, DB_PATH, None


def render_sidebar() -> None:
    """Render the sidebar with DB status, active dataset badge, and project info."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/"
            "ALK_logo.svg/120px-ALK_logo.svg.png",
            width=80,
        )
        st.markdown("### Patient Data Analysis")
        st.markdown("*Generic Data Pipeline*")
        st.divider()

        # DB Status
        connected, dataset_count, db_path, _ = _get_db_status()
        st.markdown("#### Database")
        if connected:
            st.success(f"Connected — {dataset_count} dataset(s)")
            st.caption(f"`{db_path}`")
        else:
            st.error("Not connected")
            st.caption(f"Expected: `{db_path}`")
            st.caption("Import a dataset on **Data Sources**.")

        # Active Dataset Badge
        active_name = st.session_state.get("active_dataset_name")
        active_type = st.session_state.get("active_dataset_type", "generic")
        if active_name:
            st.divider()
            st.markdown("#### Active Dataset")
            icon = dataset_type_icon(active_type)
            label = dataset_type_label(active_type)
            st.markdown(f"{icon} **{active_name}**")
            st.caption(f"Type: {label}")
            enrich_status = st.session_state.get("active_enrichment_status", "none")
            if active_type != "generic":
                enrich_badge = (
                    "✅ enriched"
                    if enrich_status == "done"
                    else "⚠ not enriched"
                )
                st.caption(enrich_badge)

        st.divider()
        st.markdown(
            "<small>ALK Kozminski University<br/>Semester Project 2</small>",
            unsafe_allow_html=True,
        )
