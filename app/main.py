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
    st.Page("views/1_data_sources.py",  title="Data Sources",     icon="📂"),
    st.Page("views/2_exploration.py",   title="Data Exploration",  icon="🔍"),
    st.Page("views/3_dashboards.py",    title="Dashboards",        icon="📈", url_path="dashboards"),
    st.Page("views/4_analytics.py",     title="Analytics",         icon="📊"),
    st.Page("views/7_pipelines.py",     title="Pipelines",         icon="🔧", url_path="pipelines"),
    st.Page("views/5_reports.py",       title="Reports",           icon="📄"),
    st.Page("views/6_documentation.py", title="Documentation",     icon="📚"),
]

pg = st.navigation(pages)
pg.run()
