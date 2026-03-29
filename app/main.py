"""
main.py — Streamlit multi-page application entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Patient Data Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("app/pages/1_data_sources.py", title="Data Sources", icon="📂"),
    st.Page("app/pages/2_exploration.py", title="Data Exploration", icon="🔍"),
    st.Page("app/pages/3_analytics.py", title="Analytics", icon="📊"),
    st.Page("app/pages/4_reports.py", title="Reports", icon="📄"),
    st.Page("app/pages/5_architecture.py", title="Architecture", icon="📐"),
]

pg = st.navigation(pages)
pg.run()
