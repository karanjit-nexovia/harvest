"""Harvest: say what you want your vision model to learn, get a clean, checked training dataset.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Harvest", layout="wide", initial_sidebar_state="expanded")
from harvest import config, ui_audit, ui_landing  # noqa: E402

ui_landing.gate()   # the launch page, until the visitor picks what to train
ui_audit.sidebar_brand()
page = st.sidebar.radio("Page", ["Build dataset", "Audit report"], label_visibility="collapsed", key="nav")
if st.sidebar.button("← Home", use_container_width=True):
    st.query_params.clear()
    st.session_state["entered"] = False
    st.session_state["seen_loader"] = True
    st.rerun()
st.sidebar.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.sidebar.caption("Demo mode (no keys)" if config.MOCK else
                   "Connected · VAST Data · NVIDIA Cosmos3-Reason · Weights & Biases")

if page == "Build dataset":
    from harvest import ui_flow
    ui_flow.page()
else:
    from harvest import ui_report
    ui_report.page()
