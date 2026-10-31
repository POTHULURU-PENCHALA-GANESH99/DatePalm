"""
Date Palm White Scale Disease Classification - Streamlit Application
Advanced ML-Powered Disease Detection System
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from src.styles import CUSTOM_CSS
from src.model_loader import load_models
from src.ui_components import render_header, render_sidebar
from src.pages import (
    render_home_page,
    render_dataset_page,
    render_preprocessing_page,
    render_prediction_page,
    render_performance_page
)

st.set_page_config(
    page_title="Date Palm Disease Classification System",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'models' not in st.session_state:
        st.session_state.models = None
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.models is None or st.session_state.extractor is None:
        with st.spinner("Loading models..."):
            models, extractor, summary = load_models()
            if models is not None:
                st.session_state.models = models
                st.session_state.extractor = extractor
                st.session_state.summary = summary
    
    render_header()
    render_sidebar()
    
    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'dataset':
        render_dataset_page()
    elif st.session_state.page == 'preprocessing':
        render_preprocessing_page()
    elif st.session_state.page == 'predictions':
        render_prediction_page()
    elif st.session_state.page == 'performance':
        render_performance_page()


if __name__ == "__main__":
    main()
