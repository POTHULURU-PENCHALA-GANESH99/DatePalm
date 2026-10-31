"""
Model Loading for Date Palm Disease Classification
"""

import pickle
from pathlib import Path
import streamlit as st


@st.cache_resource
def load_models():
    """Load trained models and feature extractor"""
    try:
        models_dir = Path('models')
        
        if not models_dir.exists():
            return None, None, None
        
        models = {}
        
        model_files = {
            'SVM': 'svm_paper_exact.pkl',
            'KNN': 'knn_paper_exact.pkl',
            'Random Forest': 'random_forest_paper_exact.pkl',
            'LightGBM': 'lightgbm_paper_exact.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = models_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
        
        extractor_path = models_dir / 'feature_extractor.pkl'
        if extractor_path.exists():
            with open(extractor_path, 'rb') as f:
                extractor = pickle.load(f)
        else:
            extractor = None
        
        summary_path = models_dir / 'results_summary_paper_exact.pkl'
        if summary_path.exists():
            with open(summary_path, 'rb') as f:
                summary = pickle.load(f)
        else:
            summary = None
        
        return models, extractor, summary
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None
