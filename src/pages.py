"""
Page Rendering Functions for Date Palm Disease Classification
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import time
import os
import random
from pathlib import Path
import glob
import cv2


CLASS_NAMES = [
    'Healthy',
    'Low Infestation (Stage 1)',
    'Medium Infestation (Stage 2)',
    'High Infestation (Stage 3)'
]

CLASS_COLORS = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']


def preprocess_image(image):
    """Preprocess image for prediction"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array


def predict_image(image, model_name):
    """Make prediction on image"""
    if not st.session_state.models or not st.session_state.extractor:
        return None, None
    
    img_array = preprocess_image(image)
    
    features = st.session_state.extractor.extract_features(img_array)
    features = features.reshape(1, -1)
    
    model_data = st.session_state.models[model_name]
    model = model_data['model']
    scaler = model_data['scaler']
    
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction, probabilities


def render_home_page():
    """Render home page"""
    st.title("Problem Statement")
    try:
        palm_img = Image.open("Gemini_Generated_Image_yu1wsiyu1wsiyu1w.png")
        st.image(palm_img, width=400)
    except:
        pass
    
    st.markdown("""
    ### Problem Statement
    
    Date palm white scale disease is a serious agricultural threat affecting date palm plantations worldwide. 
    Early and accurate detection is crucial for effective treatment and preventing crop loss. However, traditional 
    manual diagnosis poses several critical challenges:
    
    **High Stakes** – Misdiagnosis can lead to delayed treatment or unnecessary chemical applications, directly 
    impacting crop yield and farmer income.
    
    **Time-Intensive Analysis** – Manual examination of palm leaflets requires significant expertise and time 
    from agricultural specialists.
    
    **Diagnostic Variability** – Different experts may interpret the same leaflet differently, leading to 
    inconsistent diagnoses.
    
    **Large Scale Processing** – Agricultural facilities need to process numerous samples efficiently without 
    compromising accuracy.
    
    ---
    
    ### Project Objective
    
    This research aims to develop an **AI-powered date palm disease classification system** that leverages 
    **machine learning algorithms and advanced feature extraction techniques** to accurately detect and classify 
    white scale disease into four stages. The proposed approach includes:
    
    **Multi-Stage Classification** – Accurately classify disease into Healthy, Stage 1 (Low), Stage 2 (Medium), 
    and Stage 3 (High) infestation levels.
    
    **Comprehensive Feature Extraction** – Extract 89 features combining GLCM texture features (80) and HSV 
    color moments (9) for thorough analysis.
    
    **Multiple ML Models** – Compare performance of SVM, KNN, Random Forest, and LightGBM classifiers.
    
    **High Accuracy Detection** – Achieve reliable classification to support agricultural decision-making.
    
    **Scalable Solution** – Design for integration into agricultural monitoring systems and mobile applications.
    
    This approach enhances the accuracy and reliability of date palm disease detection, enabling agricultural 
    professionals to **make faster, more confident diagnostic decisions** and ultimately improve crop management.
    
    ---
    
    ### Machine Learning Models Used
    
    The proposed system incorporates advanced machine learning techniques:
    
    **Support Vector Machine (SVM)** – High-performance classification with optimized hyperparameters.
    
    **K-Nearest Neighbors (KNN)** – Instance-based learning for pattern recognition.
    
    **Random Forest** – Ensemble method combining multiple decision trees.
    
    **LightGBM** – Gradient boosting framework for efficient and accurate predictions.
    
    ---
    
    ### Disclaimer
    
    This system is designed to **assist agricultural professionals** and should **not be used as the sole 
    basis for treatment decisions**. Always consult with qualified agricultural experts for crop management.
    """)


def render_dataset_page():
    """Render dataset information page"""
    st.title("Dataset Information")
    
    st.markdown("""
    ### Dataset Overview
    
    The dataset used in this project follows the methodology from the research paper on Date Palm White 
    Scale Disease classification.
    
    **Total Images**: Images of date palm leaflets at various disease stages
    
    **Classes**: 4 disease stages
    - Healthy (No disease present)
    - Stage 1: Low Infestation (Early detection)
    - Stage 2: Medium Infestation (Treatment recommended)
    - Stage 3: High Infestation (Urgent treatment required)
    
    **Image Format**: RGB color images of date palm leaflets
    
    **Resolution**: Resized to 224×224 pixels for processing
    
    **Split**: 80% training, 20% testing
    
    ---
    
    ### Feature Extraction
    
    The system extracts **89 features** from each image:
    
    #### GLCM Texture Features (80 features)
    - **Distances**: 4 different pixel distances (1, 3, 5, 7)
    - **Angles**: 4 different angles (0°, 45°, 90°, 135°)
    - **Properties**: 5 texture properties per distance-angle combination
      - Contrast
      - Dissimilarity
      - Homogeneity
      - Energy
      - Correlation
    - **Total**: 4 distances × 4 angles × 5 properties = 80 features
    
    #### HSV Color Moment Features (9 features)
    - **Channels**: 3 HSV color channels (Hue, Saturation, Value)
    - **Moments**: 3 statistical moments per channel
      - Mean
      - Standard Deviation
      - Skewness
    - **Total**: 3 channels × 3 moments = 9 features
    
    **Grand Total**: 80 + 9 = 89 features
    
    ---
    
    ### Data Preprocessing Pipeline
    
    #### Image Loading & Resizing
    - Load RGB images
    - Resize to 224×224 pixels
    - Normalize pixel values to [0, 1] range
    
    #### Feature Extraction
    - Extract GLCM texture features from grayscale conversion
    - Extract HSV color moments from color channels
    - Combine all features into 89-dimensional feature vector
    
    #### Feature Scaling
    - Apply StandardScaler normalization
    - Ensure features are on similar scales
    - Improve model convergence and performance
    
    ---
    
    ### Sample Images
    
    Below are random samples from each disease stage in the dataset:
    """)
    
    if 'sample_seed' not in st.session_state:
        st.session_state.sample_seed = 0
    
    if st.button("Refresh Samples", type="primary", use_container_width=True):
        st.session_state.sample_seed += 1
        st.rerun()
    
    random.seed(st.session_state.sample_seed)
    
    data_dir = Path('data')
    
    if data_dir.exists():
        healthy_dir = data_dir / 'healthy'
        ws_dir = data_dir / 'white scale'
        
        healthy_images = list(healthy_dir.glob('*.jpg')) if healthy_dir.exists() else []
        ws_images = list(ws_dir.glob('*.jpg')) if ws_dir.exists() else []
        
        ws_stage1 = [img for img in ws_images if 'wsphase1' in img.name.lower()]
        ws_stage2 = [img for img in ws_images if 'wsphase2' in img.name.lower()]
        ws_stage3 = [img for img in ws_images if 'wsstage3' in img.name.lower()]
        
        st.markdown("#### Healthy Leaflets")
        if healthy_images:
            sample_healthy = random.sample(healthy_images, min(4, len(healthy_images)))
            cols = st.columns(4)
            for idx, img_path in enumerate(sample_healthy):
                with cols[idx]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=f"Healthy {idx+1}")
                    except:
                        pass
        else:
            st.info("No healthy images found in dataset.")
        
        st.markdown("#### Stage 1 - Low Infestation")
        if ws_stage1:
            sample_stage1 = random.sample(ws_stage1, min(4, len(ws_stage1)))
            cols = st.columns(4)
            for idx, img_path in enumerate(sample_stage1):
                with cols[idx]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=f"Stage 1 - {idx+1}")
                    except:
                        pass
        else:
            st.info("No Stage 1 images found in dataset.")
        
        st.markdown("#### Stage 2 - Medium Infestation")
        if ws_stage2:
            sample_stage2 = random.sample(ws_stage2, min(4, len(ws_stage2)))
            cols = st.columns(4)
            for idx, img_path in enumerate(sample_stage2):
                with cols[idx]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=f"Stage 2 - {idx+1}")
                    except:
                        pass
        else:
            st.info("No Stage 2 images found in dataset.")
        
        st.markdown("#### Stage 3 - High Infestation")
        if ws_stage3:
            sample_stage3 = random.sample(ws_stage3, min(4, len(ws_stage3)))
            cols = st.columns(4)
            for idx, img_path in enumerate(sample_stage3):
                with cols[idx]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=f"Stage 3 - {idx+1}")
                    except:
                        pass
        else:
            st.info("No Stage 3 images found in dataset.")
    else:
        st.warning("Dataset directory not found. Please ensure images are in the 'data' folder.")
    
    st.markdown("---")
    
    st.markdown("""
    ### Dataset Statistics
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Features", "89")
    
    with col2:
        st.metric("GLCM Features", "80")
    
    with col3:
        st.metric("Color Features", "9")
    
    with col4:
        st.metric("Classes", "4")


def render_preprocessing_page():
    """Render data preprocessing page with interactive before/after images"""
    st.title("Data Preprocessing")
    
    st.markdown("""
    <div class="info-box">
        <p>
            Explore the data preprocessing pipeline used in this project. See how raw images are transformed 
            through various preprocessing steps to prepare them for feature extraction and model training.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Preprocessing Pipeline Overview")
    
    st.markdown("""
    <div class="preprocessing-container">
        <h4>Step-by-Step Transformation Process</h4>
        <p>Each image undergoes a series of preprocessing steps to ensure consistency and optimal feature extraction:</p>
        <ol>
            <li><b>Image Loading</b> - Load original RGB image from dataset</li>
            <li><b>Resizing</b> - Standardize all images to 224×224 pixels</li>
            <li><b>Normalization</b> - Scale pixel values to [0, 1] range</li>
            <li><b>Color Space Conversion</b> - Convert to HSV for color feature extraction</li>
            <li><b>Grayscale Conversion</b> - Convert to grayscale for texture analysis</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Interactive Preprocessing Demonstration")
    
    data_dir = Path('data')
    
    if data_dir.exists():
        healthy_dir = data_dir / 'healthy'
        ws_dir = data_dir / 'white scale'
        
        healthy_images = list(healthy_dir.glob('*.jpg')) if healthy_dir.exists() else []
        ws_images = list(ws_dir.glob('*.jpg')) if ws_dir.exists() else []
        
        all_images = healthy_images + ws_images
        
        if all_images:
            if 'preprocessing_image' not in st.session_state:
                st.session_state.preprocessing_image = random.choice(all_images)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Load Random Image", use_container_width=True):
                    st.session_state.preprocessing_image = random.choice(all_images)
                    st.rerun()
            
            with col2:
                preprocessing_step = st.selectbox(
                    "Select Preprocessing Step",
                    ["Original", "Resized (224×224)", "Normalized", "HSV Color Space", "Grayscale"]
                )
            
            st.markdown("---")
            
            try:
                original_img = Image.open(st.session_state.preprocessing_image).convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Before")
                    st.image(original_img, use_container_width=True, caption=f"Original Image ({original_img.size[0]}×{original_img.size[1]})")
                    st.info(f"**Image:** {st.session_state.preprocessing_image.name}")
                
                with col2:
                    st.markdown("#### After")
                    
                    if preprocessing_step == "Original":
                        processed_img = original_img
                        st.image(processed_img, use_container_width=True, caption="Original RGB Image")
                        st.success("No preprocessing applied - showing original image")
                    
                    elif preprocessing_step == "Resized (224×224)":
                        processed_img = original_img.resize((224, 224))
                        st.image(processed_img, use_container_width=True, caption="Resized to 224×224 pixels")
                        st.success(f"Resized from {original_img.size[0]}×{original_img.size[1]} to 224×224")
                    
                    elif preprocessing_step == "Normalized":
                        resized_img = original_img.resize((224, 224))
                        img_array = np.array(resized_img) / 255.0
                        processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
                        st.image(processed_img, use_container_width=True, caption="Normalized (pixel values 0-1)")
                        st.success("Pixel values normalized to [0, 1] range")
                    
                    elif preprocessing_step == "HSV Color Space":
                        resized_img = original_img.resize((224, 224))
                        img_array = np.array(resized_img)
                        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                        processed_img = Image.fromarray(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
                        st.image(processed_img, use_container_width=True, caption="HSV Color Space Representation")
                        st.success("Converted to HSV for color moment extraction")
                    
                    elif preprocessing_step == "Grayscale":
                        resized_img = original_img.resize((224, 224))
                        gray_img = resized_img.convert('L')
                        st.image(gray_img, use_container_width=True, caption="Grayscale for Texture Analysis")
                        st.success("Converted to grayscale for GLCM texture features")
                
                st.markdown("---")
                
                st.markdown("### Preprocessing Details")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>224×224</h3>
                        <p>Standard Size</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>[0, 1]</h3>
                        <p>Normalized Range</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>89</h3>
                        <p>Features Extracted</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.warning("No images found in dataset directory.")
    else:
        st.warning("Dataset directory not found. Please ensure images are in the 'data' folder.")
    
    st.markdown("---")
    
    st.markdown("### Feature Extraction After Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### GLCM Texture Features (80)
        Extracted from **grayscale** images:
        - Contrast
        - Dissimilarity
        - Homogeneity
        - Energy
        - Correlation
        
        Computed at 4 distances × 4 angles = 16 combinations
        Each combination yields 5 properties = **80 features**
        """)
    
    with col2:
        st.markdown("""
        #### HSV Color Moments (9)
        Extracted from **HSV color space**:
        - Mean (H, S, V)
        - Standard Deviation (H, S, V)
        - Skewness (H, S, V)
        
        3 channels × 3 moments = **9 features**
        
        **Total: 89 features** ready for ML models
        """)


def render_prediction_page():
    """Render predictions page"""
    st.title("Predictions")
    
    if st.session_state.models is None or st.session_state.extractor is None:
        st.error("Models not loaded. Please restart the application.")
        return
    
    st.markdown("""
    <div class="info-box">
        <p>
            Upload a date palm leaflet image to get real-time disease classification predictions
            using trained machine learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Selection")
    model_name = st.selectbox(
        "Select Model",
        list(st.session_state.models.keys()),
        label_visibility="collapsed"
    )
    
    if st.session_state.summary and model_name in st.session_state.summary:
        accuracy = st.session_state.summary[model_name]['accuracy'] * 100
        st.info(f"**Using {model_name}** ({accuracy:.2f}% accuracy)")
    
    st.markdown("### Upload Image")
    
    uploaded_file = st.file_uploader("Choose a date palm leaflet image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### Image Info")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**Format:** {uploaded_file.type}")
        
        st.markdown("---")
        
        if st.button("Predict Disease Stage", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preprocessing image...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            status_text.text("Extracting features...")
            progress_bar.progress(50)
            time.sleep(0.3)
            
            status_text.text("Making prediction...")
            progress_bar.progress(75)
            time.sleep(0.3)
            
            prediction, probabilities = predict_image(image, model_name)
            
            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            progress_bar.empty()
            status_text.empty()
            
            if prediction is not None:
                st.session_state.prediction_history.append({
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'model': model_name
                })
                
                st.success("Prediction Complete!")
                
                st.markdown("### Prediction Result")
                
                predicted_class = CLASS_NAMES[prediction]
                predicted_color = CLASS_COLORS[prediction]
                confidence = probabilities[prediction]
                
                st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, {predicted_color} 0%, {predicted_color}dd 100%);">
                        <h2>{predicted_class}</h2>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">
                            Confidence: {confidence:.1%}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{model_name}</h3>
                        <p>Model Used</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{confidence*100:.1f}%</h3>
                        <p>Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Stage {prediction}</h3>
                        <p>Classification</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Probability Distribution")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=CLASS_NAMES,
                        y=probabilities,
                        text=[f'{p:.1%}' for p in probabilities],
                        textposition='auto',
                        marker_color=CLASS_COLORS
                    )
                ])
                
                fig.update_layout(
                    title="Classification Probabilities",
                    xaxis_title="Class",
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Recommendations")
                
                recommendations = {
                    0: {
                        "title": "Healthy Leaflet",
                        "action": "Continue regular monitoring and preventive care.",
                        "color": "#2ecc71"
                    },
                    1: {
                        "title": "Early Detection",
                        "action": "Monitor closely. Chemical treatment may not be necessary yet.",
                        "color": "#f39c12"
                    },
                    2: {
                        "title": "Action Required",
                        "action": "Chemical treatment recommended to prevent further progression.",
                        "color": "#e67e22"
                    },
                    3: {
                        "title": "Urgent Action",
                        "action": "Immediate chemical treatment required! Tree health at risk.",
                        "color": "#e74c3c"
                    }
                }
                
                rec = recommendations[prediction]
                
                st.markdown(f"""
                    <div style="background: {rec['color']}15; border-left: 5px solid {rec['color']}; 
                         padding: 1.5rem; border-radius: 8px;">
                        <h3 style="color: {rec['color']}; margin: 0;">{rec['title']}</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                            <b>Recommended Action:</b> {rec['action']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)


def render_performance_page():
    """Render model performance page"""
    st.title("Model Training")
    
    if not st.session_state.summary:
        st.error("No performance data available")
        return
    
    st.markdown("### Overall Performance")
    
    df = pd.DataFrame(st.session_state.summary).T
    df['Accuracy (%)'] = (df['accuracy'] * 100).round(2)
    df['Precision'] = df['precision'].round(4)
    df['Recall'] = df['recall'].round(4)
    df['F1-Score'] = df['f1_score'].round(4)
    df['Training Time (s)'] = df['train_time'].round(2)
    
    st.dataframe(
        df[['Accuracy (%)', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)']],
        use_container_width=True
    )
    
    best_model = df['accuracy'].idxmax()
    best_acc = df.loc[best_model, 'Accuracy (%)']
    
    st.success(f"**Best Model:** {best_model} with {best_acc:.2f}% accuracy")
    
    st.markdown("### Model Comparison")
    
    fig = go.Figure()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for metric, name in zip(metrics, metric_names):
        fig.add_trace(go.Bar(
            name=name,
            x=list(df.index),
            y=df[metric],
            text=[f'{v:.3f}' for v in df[metric]],
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Performance Radar Chart")
    
    fig_radar = go.Figure()
    
    for model in df.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                df.loc[model, 'accuracy'] * 100,
                df.loc[model, 'precision'] * 100,
                df.loc[model, 'recall'] * 100,
                df.loc[model, 'f1_score'] * 100
            ],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Multi-Metric Performance Comparison",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Training Time vs Accuracy")
    
    fig_scatter = go.Figure()
    
    colors_map = {
        'SVM': '#6366f1',
        'KNN': '#8b5cf6',
        'Random Forest': '#06b6d4',
        'LightGBM': '#f59e0b'
    }
    
    for model in df.index:
        fig_scatter.add_trace(go.Scatter(
            x=[df.loc[model, 'train_time']],
            y=[df.loc[model, 'accuracy'] * 100],
            mode='markers+text',
            name=model,
            text=[model],
            textposition='top center',
            marker=dict(
                size=20,
                color=colors_map.get(model, '#6366f1'),
                line=dict(width=2, color='white')
            )
        ))
    
    fig_scatter.update_layout(
        title="Training Efficiency: Accuracy vs Training Time",
        xaxis_title="Training Time (seconds)",
        yaxis_title="Accuracy (%)",
        yaxis_range=[90, 100],
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Metrics Heatmap")
    
    heatmap_data = df[['accuracy', 'precision', 'recall', 'f1_score']].T
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values * 100,
        x=heatmap_data.columns,
        y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        text=np.round(heatmap_data.values * 100, 2),
        texttemplate='%{text:.2f}%',
        textfont={"size": 12},
        colorscale='Viridis',
        colorbar=dict(title="Score (%)"),
    ))
    
    fig_heatmap.update_layout(
        title="Performance Metrics Heatmap",
        xaxis_title="Model",
        yaxis_title="Metric",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Comparison with Paper Results")
    
    paper_results = {
        'SVM': 0.9829,
        'LightGBM': 0.9752,
        'KNN': 0.9690,
        'Random Forest': 0.9542
    }
    
    comparison_data = []
    for model in df.index:
        your_acc = df.loc[model, 'accuracy']
        paper_acc = paper_results.get(model, 0)
        diff = (your_acc - paper_acc) * 100
        
        comparison_data.append({
            'Model': model,
            'Your Accuracy': f"{your_acc*100:.2f}%",
            'Paper Accuracy': f"{paper_acc*100:.2f}%",
            'Difference': f"{diff:+.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Your Results',
        x=[row['Model'] for _, row in comparison_df.iterrows()],
        y=[float(row['Your Accuracy'].strip('%')) for _, row in comparison_df.iterrows()],
        marker_color='#6366f1',
        text=[row['Your Accuracy'] for _, row in comparison_df.iterrows()],
        textposition='auto',
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='Paper Results',
        x=[row['Model'] for _, row in comparison_df.iterrows()],
        y=[float(row['Paper Accuracy'].strip('%')) for _, row in comparison_df.iterrows()],
        marker_color='#8b5cf6',
        text=[row['Paper Accuracy'] for _, row in comparison_df.iterrows()],
        textposition='auto',
    ))
    
    fig_comparison.update_layout(
        title="Your Results vs Paper Benchmark",
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        barmode='group',
        height=400,
        template='plotly_white',
        yaxis_range=[90, 100]
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Model Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{best_acc:.2f}%</h3>
            <p>Best Accuracy</p>
            <small>{best_model}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fastest_model = df['train_time'].idxmin()
        fastest_time = df.loc[fastest_model, 'train_time']
        st.markdown(f"""
        <div class="metric-card">
            <h3>{fastest_time:.2f}s</h3>
            <p>Fastest Training</p>
            <small>{fastest_model}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_precision = df['precision'].max()
        best_prec_model = df['precision'].idxmax()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{best_precision:.4f}</h3>
            <p>Best Precision</p>
            <small>{best_prec_model}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_recall = df['recall'].max()
        best_recall_model = df['recall'].idxmax()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{best_recall:.4f}</h3>
            <p>Best Recall</p>
            <small>{best_recall_model}</small>
        </div>
        """, unsafe_allow_html=True)
