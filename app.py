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
# 3. LOAD RESOURCES
# ==========================================


@st.cache_resource
def load_resources():
    try:
        data = pd.read_csv('velorium_data.csv')
        models_dict = joblib.load('velorium_models_dict.pkl')
        model = models_dict['Logistic Regression']
        return data, model
    except:
        return None, None


df, pipeline = load_resources()

if df is None:
    st.error("❌ Critical Error: Files missing.")
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
                since_promo = st.number_input(
                    "Years Since Promotion", 0, 20, 2)
                satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
            with col3:
                manager_rel = st.slider("Relationship w/ Manager", 1, 4, 3)
                projects = st.number_input("Project Count", 1, 10, 4)
                hours = st.number_input("Avg Hours/Week", 30, 80, 45)
                marital = st.selectbox(
                    "Marital Status", df['Marital_Status'].unique())
                distance = st.number_input("Distance From Home", 1, 100, 10)

            submitted = st.form_submit_button("💾 Register")

            if submitted:
                new_data = df.iloc[0:1].copy()
                new_data['Employee_ID'] = emp_id
                new_data['Department'] = dept
                new_data['Job_Role'] = role
                new_data['Age'] = age
                new_data['Gender'] = gender
                new_data['Monthly_Income'] = income
                new_data['Overtime'] = ot
                new_data['Work_Life_Balance'] = wlb
                new_data['Years_at_Company'] = tenure
                new_data['Years_Since_Last_Promotion'] = since_promo
                new_data['Job_Satisfaction'] = satisfaction
                new_data['Relationship_with_Manager'] = manager_rel
                new_data['Project_Count'] = projects
                new_data['Average_Hours_Worked_Per_Week'] = hours
                new_data['Marital_Status'] = marital
                new_data['Distance_From_Home'] = distance

                st.session_state.target_employee_data = new_data  # SAVE TO MEMORY
                st.success(f"New Employee #{emp_id} Registered Temporarily.")

    # --- SHOW DATA IF LOADED ---
    if st.session_state.target_employee_data is not None:
        target_row = st.session_state.target_employee_data

        # Display the data (Clean View)
        with st.expander("📂 View Current Profile Data", expanded=False):
            st.dataframe(target_row.drop(
                columns=['Attrition'], errors='ignore'), hide_index=True)

        # --- RUN ANALYSIS BUTTON (NOW CONNECTED TO PERSISTENT DATA) ---
        if st.button("🚀 RUN RISK ANALYSIS"):
            st.session_state.chat_history = []
            st.session_state.full_strategy = ""

            if not api_key:
                st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Crunching numbers..."):
                    # Pipeline
                    prep_step = pipeline.named_steps['preprocessor']
                    model_step = pipeline.named_steps['classifier']
                    X_in = target_row.drop(
                        columns=['Employee_ID', 'Attrition'], errors='ignore')
                    X_proc = prep_step.transform(X_in)
                    prob = model_step.predict_proba(X_proc)[0][1]

                    # SHAP
                    try:
                        X_bg = prep_step.transform(
                            df.drop(columns=['Employee_ID', 'Attrition'], errors='ignore').head(50))
                        explainer = shap.LinearExplainer(model_step, X_bg)
                        shap_values = explainer.shap_values(X_proc)
                        vals = shap_values[0] if not isinstance(
                            shap_values, list) else shap_values
                    except:
                        vals = np.zeros(X_proc.shape[1])

                    # Top 5
                    try:
                        feats = prep_step.get_feature_names_out()
                    except:
                        feats = [f"F{i}" for i in range(len(vals))]

                    shap_df = pd.DataFrame({'Feature': feats, 'Impact': vals})
                    shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                    top_5 = shap_df.sort_values(
                        'Abs_Impact', ascending=False).head(5)

                    # Risk Label Logic
                    if prob > 0.75:
                        risk_label = "HIGH RISK (! Attention Needed)"
                    elif prob > 0.50:
                        risk_label = "RISK (Monitor)"
                    else:
                        risk_label = "SAFE (Stable)"

                    # Strategy Prompt
                    drivers_txt = ", ".join(
                        [f"{r['Feature']}" for i, r in top_5.iterrows()])
                    role_val = target_row['Job_Role'].values[0]

                    strategy_prompt = f"""
                    You are a Senior HR Consultant.
                    Subject: Employee #{emp_id} ({role_val}). 
                    Risk Assessment: {risk_label} ({prob:.1%}).
                    Key Drivers: {drivers_txt}.
                    Write a comprehensive "Retention Strategy Document" in Markdown.
                    Structure:
                    ### 🚨 Risk Assessment
                    ### 🔍 Root Cause Analysis
                    ### 📅 30-Day Retention Plan
                    """

                    try:
                        resp = llm_model.generate_content(strategy_prompt)
                        strategy_text = resp.text
                    except:
                        strategy_text = "AI Error: Could not generate strategy."

                    # Save Analysis State
                    st.session_state.analysis_data = {
                        "id": emp_id,
                        "role": role_val,
                        "prob": prob,
                        "top_5": top_5,
                        "drivers_txt": drivers_txt,
                        "risk_label": risk_label
                    }
                    st.session_state.full_strategy = strategy_text

    # --- RESULTS DISPLAY ---
    if st.session_state.analysis_data:
        data = st.session_state.analysis_data

        st.divider()
        res_c1, res_c2 = st.columns([1, 1.5])

        with res_c1:
            # 3-TIER RISK LOGIC
            if data['prob'] > 0.75:
                st.markdown(f"""
                <div class="risk-alert" style="border-color: #d50000;">
                    <h3 class="risk-high-text">HIGH RISK</h3>
                    <p class="risk-high-text" style="font-size: 20px;">(! Attention Needed)</p>
                    <p class="prob-text">{data['prob']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            elif data['prob'] > 0.50:
                st.markdown(f"""
                <div class="risk-alert" style="border-color: #ef6c00;">
                    <h3 class="risk-med-text">RISK</h3>
                    <p class="prob-text">{data['prob']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-alert" style="border-color: #2e7d32;">
                    <h3 class="risk-safe-text">SAFE</h3>
                    <p class="prob-text">{data['prob']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

        with res_c2:
            st.markdown("### Why? (Top Drivers)")
            top_5_view = data['top_5'].copy()
            top_5_view['Feature'] = top_5_view['Feature'].str.replace(
                'cat__', '').str.replace('num__', '').str.replace('ord__', '')

            def fmt_arrow(
                x): return "⬆️ Increases Risk" if x > 0 else "⬇️ Decreases Risk"
            top_5_view['Effect'] = top_5_view['Impact'].apply(fmt_arrow)
            top_5_view['SHAP Value'] = top_5_view['Impact'].round(4)
            st.dataframe(top_5_view[['Feature', 'Effect', 'SHAP Value']],
                         hide_index=True, use_container_width=True)

        # STRATEGY
        if st.session_state.full_strategy:
            st.divider()
            st.markdown("### 🧠 Comprehensive Retention Strategy")
            st.markdown(st.session_state.full_strategy)

        # ACTION CENTER
        st.divider()
        st.markdown("### ⚡ Action Center")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([3, 1])
        with col_act1:
            action_type = st.selectbox("Select Document to Draft:",
                                       ["Choose an action...",
                                        "📧 Manager to HR Email (Alert)",
                                        "📜 Policy Exception Request",
                                        "💬 1:1 Script (Verbatim)"])
        with col_act2:
            st.write("")
            st.write("")
            gen_draft = st.button("Generate Draft")

        if gen_draft and action_type != "Choose an action...":
            with st.spinner("Drafting..."):
                draft_prompt = f"""
                Context: Employee #{data['id']}, Status: {data['risk_label']} ({data['prob']:.1%}).
                Drivers: {data['drivers_txt']}.
                Task: Write a {action_type}. 
                Tone: Professional.
                """
                try:
                    draft_resp = llm_model.generate_content(draft_prompt)
                    st.success(f"Draft Generated: {action_type}")
                    st.markdown(draft_resp.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # CHAT
        st.divider()
        st.markdown("### 💬 Copilot Chat")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about salary, workload..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_prompt = f"""
                    Context: Employee #{data['id']}, Status: {data['risk_label']}, Drivers: {data['drivers_txt']}.
                    Question: {prompt}
                    Answer concisely.
                    """
                    try:
                        chat_resp = llm_model.generate_content(chat_prompt)
                        st.markdown(chat_resp.text)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": chat_resp.text})
                    except Exception as e:
                        st.error(f"AI Error: {e}")
