"""
UI Components for Date Palm Disease Classification
"""

import streamlit as st


def render_header():
    """Render application header"""
    st.markdown("""
        <div class="main-header">
            <h1>Date Palm White Scale Disease Classification</h1>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("### Navigation")
        
        if st.button("Problem Statement", key="nav_home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("Dataset Information", key="nav_dataset", use_container_width=True):
            st.session_state.page = 'dataset'
            st.rerun()
        
        if st.button("Data Preprocessing", key="nav_preprocessing", use_container_width=True):
            st.session_state.page = 'preprocessing'
            st.rerun()
        
        if st.button("Model Training", key="nav_performance", use_container_width=True):
            st.session_state.page = 'performance'
            st.rerun()
        
        if st.button("Predictions", key="nav_predictions", use_container_width=True):
            st.session_state.page = 'predictions'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### System Status")
        
        if st.session_state.models is not None:
            st.success("Models Loaded")
            st.metric("Models Available", len(st.session_state.models))
        else:
            st.error("Models Not Loaded")
        
        if st.session_state.extractor is not None:
            st.success("Feature Extractor Ready")
        else:
            st.error("Feature Extractor Missing")
        
        st.markdown("---")
        
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classes", "4")
            st.metric("Features", "89")
        with col2:
            st.metric("Models", len(st.session_state.models) if st.session_state.models else 0)
            st.metric("Predictions", len(st.session_state.prediction_history))
