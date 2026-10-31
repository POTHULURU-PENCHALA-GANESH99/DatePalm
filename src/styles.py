"""
Styles for Date Palm Disease Classification Streamlit App
"""

CUSTOM_CSS = """
<style>
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    /* Global text size improvements */
    .main .block-container {
        font-size: 1.1rem;
    }
    
    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    h4 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    p, li, .stMarkdown {
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
    }
    
    .stDataFrame {
        font-size: 1.05rem !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.4);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem !important;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.3rem !important;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3);
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.4rem !important;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem !important;
        opacity: 0.9;
    }
    
    .metric-card small {
        font-size: 0.95rem !important;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-card h2 {
        margin: 0;
        font-size: 3rem !important;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-card p {
        font-size: 1.4rem !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-left: 4px solid #6366f1;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .info-box p {
        font-size: 1.15rem !important;
        margin: 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-weight: 700;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
    }
    
    .preprocessing-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .preprocessing-container h4 {
        font-size: 1.5rem !important;
        margin-bottom: 1rem;
    }
    
    .preprocessing-container p, .preprocessing-container li {
        font-size: 1.1rem !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg, [data-testid="stSidebar"] {
        font-size: 1.05rem !important;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 1.4rem !important;
    }
    
    [data-testid="stSidebar"] .stMetric label {
        font-size: 1rem !important;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
    
    /* File uploader text */
    .stFileUploader label {
        font-size: 1.15rem !important;
    }
    
    /* Selectbox text */
    .stSelectbox label {
        font-size: 1.15rem !important;
    }
    
    /* Success/Error/Warning/Info messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        font-size: 1.1rem !important;
    }
</style>
"""
