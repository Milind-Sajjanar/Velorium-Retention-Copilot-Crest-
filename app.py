import streamlit as st
import pandas as pd
import joblib
import shap
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
import os

# ==========================================
# 1. ESSENTIAL CLASS DEFINITION
# ==========================================

class NeuralNetworkPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model = None
        self.named_steps = {'preprocessor': preprocessor}

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X).astype('float32')
        prob_pos = self.model.predict(X_trans, verbose=0)
        return np.hstack([1 - prob_pos, prob_pos])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

# ==========================================
# 2. PAGE CONFIGURATION & VISUAL STYLING
# ==========================================
st.set_page_config(page_title="Velorium HRMS", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    /* 1. BACKGROUNDS */
    .stApp { background-color: #f4f8fb; }
    
    /* 2. TEXT COLORS - FORCE BLACK */
    h1, h2, h3, h4, h5, p, div, span, label { color: #000000 !important; }
    
    /* 3. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #e3f2fd !important;
        border-right: 1px solid #bbdefb;
    }
    
    /* 4. CARDS */
    .card {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        border: 1px solid #d1d9e6;
        margin-bottom: 20px;
    }
    
    /* 5. RISK ALERT BOX */
    .risk-alert {
        background-color: #e3f2fd; 
        padding: 20px;
        border-radius: 10px;
        border: 3px solid #000000;
        text-align: center;
    }
    .risk-high-text { color: #d50000 !important; font-size: 28px; font-weight: 900; margin: 0; }
    .risk-med-text { color: #ef6c00 !important; font-size: 28px; font-weight: 900; margin: 0; }
    .risk-safe-text { color: #2e7d32 !important; font-size: 28px; font-weight: 900; margin: 0; }
    
    .prob-text { color: #000000 !important; font-size: 18px; font-weight: 700; margin-top: 5px; }

    /* 6. BUTTONS */
    .stButton>button {
        width: 100%;
        background-color: #ffffff !important; 
        color: #000000 !important;            
        border-radius: 8px;
        font-weight: 700;
        font-size: 18px;
        border: 2px solid #000000;
        padding: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { 
        background-color: #f0f0f0 !important; 
        border-color: #333333;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'full_strategy' not in st.session_state:
    st.session_state.full_strategy = ""
if 'target_employee_data' not in st.session_state:
    st.session_state.target_employee_data = None

# ==========================================
# 3. LOAD RESOURCES (SMART DIAGNOSTIC VERSION)
# ==========================================

@st.cache_resource
def load_resources():
    # 1. Check if files exist first
    if not os.path.exists('velorium_data.csv'):
        st.error("❌ Error: 'velorium_data.csv' is missing from the GitHub repository. Please upload it.")
        return None, None
    if not os.path.exists('velorium_models_dict.pkl'):
        st.error("❌ Error: 'velorium_models_dict.pkl' is missing from the GitHub repository. Please upload it.")
        return None, None
        
    try:
        data = pd.read_csv('velorium_data.csv')
        models_dict = joblib.load('velorium_models_dict.pkl')
        
        # 2. Check if the pickled file is a dictionary containing the model, or just the model itself
        if isinstance(models_dict, dict):
            if 'Logistic Regression' in models_dict:
                model = models_dict['Logistic Regression']
            else:
                st.error(f"❌ Error: 'Logistic Regression' key not found. Available keys are: {list(models_dict.keys())}")
                return None, None
        else:
            # If it's not a dict, assume the file IS the model
            model = models_dict
            
        return data, model
    except Exception as e:
        st.error(f"❌ Python Error during file loading: {str(e)}")
        return None, None

df, pipeline = load_resources()

if df is None or pipeline is None:
    st.warning("⚠️ Application paused. Please fix the file errors above to continue.")
    st.stop()

# ==========================================
# 4. SIDEBAR SETUP
# ==========================================
if os.path.exists("client_logo.png"):
    st.sidebar.image("client_logo.png", use_container_width=True)
else:
    st.sidebar.markdown("### Velorium Technologies")

st.sidebar.caption("HR Management System (v3.1)")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate:", ["📊 Exec Dashboard", "👥 Employee Directory", "🤖 Retention Copilot"])

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ System Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    try:
        llm_model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        llm_model = genai.GenerativeModel('gemini-pro')

st.sidebar.markdown("---")
if os.path.exists("consultancy_logo.png"):
    c1, c2, c3 = st.sidebar.columns([1, 2, 1])
    with c2:
        st.image("consultancy_logo.png", use_container_width=True)
else:
    st.sidebar.markdown(
        "<div style='text-align: center; font-size: 12px;'>Powered by <b>AnalytIQ</b></div>", unsafe_allow_html=True)

# ==========================================
# 5. PAGE LOGIC
# ==========================================

if page == "📊 Exec Dashboard":
    st.markdown("## Executive Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="card"><h4>Headcount</h4><h2>4,500</h2><p style="color:green !important">↑ 12</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="card"><h4>Attrition</h4><h2>19.2%</h2><p style="color:red !important">↑ 1.2%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="card"><h4>Positions</h4><h2>42</h2><p style="color:grey !important">Open</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(
            '<div class="card"><h4>Avg Tenure</h4><h2>4.2y</h2><p style="color:green !important">Stable</p></div>', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ec/World_map_blank_without_borders.svg",
             caption="Global Operations Map")

elif page == "👥 Employee Directory":
    st.markdown("## Employee Directory")
    safe_df = df.drop(columns=['Attrition'], errors='ignore').head(50)
    st.dataframe(safe_df, use_container_width=True)

elif page == "🤖 Retention Copilot":
    st.markdown("## 🤖 Velorium Retention Copilot")
    st.markdown(
        "Predictive analytics module powered by **AnalytIQ Intelligence Engine**.")

    # --- INPUT CARD ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        emp_id = st.number_input("Enter Employee ID", min_value=1, value=23)
    with c2:
        st.write("")
        st.write("")
        manual_mode = st.checkbox("Register New Employee")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- DATA LOGIC (PERSISTENT) ---
    if not manual_mode:
        # EXISTING EMPLOYEE LOGIC
        if st.button("🔎 Search Database"):
            row = df[df['Employee_ID'] == emp_id].copy()
            if not row.empty:
                st.session_state.target_employee_data = row  # SAVE TO MEMORY
                st.success("✅ Employee Record Found")
            else:
                st.session_state.target_employee_data = None
                st.warning("⚠️ Employee ID not found.")
    else:
        # MANUAL REGISTRATION FORM
        st.markdown("### 📝 New Employee Registration")
        with st.form("reg_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                dept = st.selectbox("Department", df['Department'].unique())
                role = st.selectbox("Job Role", df['Job_Role'].unique())
                age = st.number_input("Age", 20, 60, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                income = st.number_input("Monthly Income", 2000, 20000, 5000)
            with col2:
                ot = st.selectbox("Overtime", ["Yes", "No"])
                wlb = st.slider("Work Life Balance", 1, 4, 3)
                tenure = st.number_input("Years at Company", 0, 40, 5)
