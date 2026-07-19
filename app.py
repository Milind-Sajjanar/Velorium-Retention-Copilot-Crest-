import streamlit as st
import pandas as pd
import joblib
import shap
from openai import OpenAI
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
# 1b. MODEL-AGNOSTIC HELPERS
# ==========================================
# These let the rest of the app treat "Logistic Regression" / "XGBoost"
# (real sklearn Pipelines) and "Neural Network" (our custom wrapper)
# the same way, instead of assuming a fixed named_steps layout.

def get_preprocessor(pipeline):
    """Return the fitted preprocessor regardless of pipeline type."""
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        return pipeline.named_steps["preprocessor"]
    if hasattr(pipeline, "preprocessor"):
        return pipeline.preprocessor
    raise ValueError("Could not locate a preprocessor on this pipeline.")


def get_classifier(pipeline):
    """Return the underlying sklearn/xgboost classifier, or None for the NN wrapper."""
    if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
        return pipeline.named_steps["classifier"]
    return None


def compute_shap(pipeline, model_choice, prep_step, X_proc, X_bg):
    """Pick the right SHAP explainer for the active model type."""
    classifier = get_classifier(pipeline)

    if model_choice == "Logistic Regression":
        explainer = shap.LinearExplainer(classifier, X_bg)
        shap_values = explainer.shap_values(X_proc)
    elif model_choice == "XGBoost":
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_proc)
    else:
        # Neural Network: model-agnostic explainer, keep the background
        # small since KernelExplainer is expensive.
        background = X_bg[:20]
        predict_fn = lambda x: pipeline.model.predict(x, verbose=0).flatten()
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_proc, nsamples=100)

    vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    return np.array(vals).flatten()


# ==========================================
# 2. PAGE CONFIGURATION & VISUAL STYLING
# ==========================================
st.set_page_config(page_title="Velorium HRMS", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    /* 1. BACKGROUNDS */
    .stApp { background-color: #f4f8fb; }

    /* 2. TEXT COLORS */
    h1, h2, h3, h4, h5 {
        color: #000000 !important;
    }

    p, label {
        color: #000000 !important;
    }

    /* Force ALL markdown-rendered content (bullet lists, bold, links, etc.)
       to be visible. Without this, <li> text inherits a light/white color
       from Streamlit's theme and disappears on the light background. */
    [data-testid="stMarkdownContainer"] * {
        color: #000000 !important;
    }

    /* Inline code spans (e.g. `Feature_Name` from AI responses) default to
       a dark background in Streamlit. Combined with the black text forced
       above, that made them unreadable black-on-black. Give them a light
       background so the black text is visible again. */
    [data-testid="stMarkdownContainer"] code {
        background-color: #eef2f7 !important;
        color: #000000 !important;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #cbd5e1;
    }

    /* 3. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #e3f2fd !important;
        border-right: 1px solid #bbdefb;
    }
    /* SELECT BOX FIX */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px;
        border: 1px solid #b0bec5;
    }
    
    div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    div[role="listbox"] {
        background-color: #ffffff !important;
    }
    
    div[role="option"] {
        color: #000000 !important;
    }
    
    div[role="option"]:hover {
        background-color: #e3f2fd !important;
    }

    /* 4. CARDS */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #d1d9e6;
        margin-bottom: 20px;
        transition: box-shadow 0.15s ease-in-out;
    }
    .card:hover {
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }

    /* 5. RISK ALERT BOX */
    .risk-alert {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 14px;
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
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 700;
        font-size: 18px;
        border: 2px solid #000000;
        padding: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #1565c0 !important;
        color: white !important;
        border-color: #333333;
    }

    /* 7. ACTIVE MODEL BADGE */
    .model-badge {
        display: inline-block;
        background-color: #000000;
        color: #ffffff !important;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 10px;
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
        data = pd.read_csv("velorium_data.csv")

        models = {
            "Logistic Regression": joblib.load("logistic_regression.pkl"),
            "XGBoost": joblib.load("xgboost.pkl"),
            "Neural Network": joblib.load("neural_network.pkl")
        }

        return data, models

    except Exception as e:
        st.error(f"Error Loading Resources:\n{e}")
        return None, None


df, models = load_resources()

if df is None or models is None:
    st.error("❌ Critical Error: Required files are missing. Check that "
              "velorium_data.csv and all three .pkl files are in the app directory.")
    st.stop()

# ==========================================
# 4. SIDEBAR SETUP
# ==========================================
if os.path.exists("client_logo.png"):
    st.sidebar.image("client_logo.png", use_container_width=True)
else:
    st.sidebar.markdown("### Velorium Technologies")

st.sidebar.caption("HR Management System (v3.2)")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate:", ["📊 Exec Dashboard", "👥 Employee Directory", "🤖 Retention Copilot"])

st.sidebar.divider()
st.sidebar.markdown("### 🧠 Prediction Model")
model_choice = st.sidebar.selectbox("Active model", list(models.keys()))
pipeline = models[model_choice]
st.sidebar.caption(f"Currently scoring with **{model_choice}**.")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ System Settings")

# Prefer a key from environment variables (Cloud Run, Docker, etc.), then
# st.secrets (Streamlit Cloud), then fall back to manual entry.
default_key = os.environ.get("GROQ_API_KEY", "")
if not default_key:
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""

api_key = st.sidebar.text_input("Groq API Key", type="password", value=default_key)
if default_key:
    st.sidebar.caption("✅ Using key from app secrets.")

GROQ_MODEL_PRIMARY = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"


def call_gemini(api_key: str, prompt: str) -> str:
    """Calls the Groq API (name kept for minimal downstream changes)."""
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception:
        resp = client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


# No separate client object needed up front — call_gemini() builds its own
# Groq client from the api_key each time it's called. We just track
# whether a key has been provided.
groq_ready = bool(api_key)

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

    # --- Normalize Attrition into a boolean flag (handles Yes/No, 1/0, True/False) ---
    def _is_attrited(val):
        if pd.isna(val):
            return False
        return str(val).strip().lower() in ("yes", "1", "true", "left")

    dash_df = df.copy()
    dash_df['_Attrited'] = dash_df['Attrition'].apply(_is_attrited) if 'Attrition' in dash_df.columns else False

    # --- FILTERS ---
    st.markdown("#### Filters")
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        dept_options = sorted(dash_df['Department'].dropna().unique()) if 'Department' in dash_df.columns else []
        dept_filter = st.multiselect("Department", dept_options, default=dept_options)
    with fc2:
        role_options = sorted(dash_df['Job_Role'].dropna().unique()) if 'Job_Role' in dash_df.columns else []
        role_filter = st.multiselect("Job Role", role_options, default=role_options)
    with fc3:
        st.write("")
        st.write("")
        if st.button("🔄 Reset"):
            st.session_state.selected_kpi = None
            st.rerun()

    if dept_filter and 'Department' in dash_df.columns:
        dash_df = dash_df[dash_df['Department'].isin(dept_filter)]
    if role_filter and 'Job_Role' in dash_df.columns:
        dash_df = dash_df[dash_df['Job_Role'].isin(role_filter)]

    if dash_df.empty:
        st.warning("⚠️ No employees match the selected filters.")
        st.stop()

    # --- LIVE KPIs ---
    headcount = len(dash_df)
    attrition_rate = dash_df['_Attrited'].mean() * 100
    avg_tenure = dash_df['Years_at_Company'].mean() if 'Years_at_Company' in dash_df.columns else None

    if 'selected_kpi' not in st.session_state:
        st.session_state.selected_kpi = None

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="card"><h4>Headcount</h4><h2>{headcount:,}</h2></div>', unsafe_allow_html=True)
        if st.button("🔍 View breakdown", key="kpi_head", use_container_width=True):
            st.session_state.selected_kpi = "headcount"
    with k2:
        st.markdown(f'<div class="card"><h4>Attrition</h4><h2>{attrition_rate:.1f}%</h2></div>', unsafe_allow_html=True)
        if st.button("🔍 View breakdown", key="kpi_attr", use_container_width=True):
            st.session_state.selected_kpi = "attrition"
    with k3:
        st.markdown('<div class="card"><h4>Positions</h4><h2>N/A</h2>'
                     '<p style="color:grey !important">Not in current dataset</p></div>', unsafe_allow_html=True)
        if st.button("🔍 Why N/A?", key="kpi_pos", use_container_width=True):
            st.session_state.selected_kpi = "positions"
    with k4:
        tenure_display = f"{avg_tenure:.1f}y" if avg_tenure is not None else "N/A"
        st.markdown(f'<div class="card"><h4>Avg Tenure</h4><h2>{tenure_display}</h2></div>', unsafe_allow_html=True)
        if st.button("🔍 View breakdown", key="kpi_tenure", use_container_width=True):
            st.session_state.selected_kpi = "tenure"

    # --- CHARTS ---
    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Attrition Rate by Department")
        if 'Department' in dash_df.columns:
            dept_stats = dash_df.groupby('Department')['_Attrited'].mean().mul(100).sort_values(ascending=False)
            st.bar_chart(dept_stats)
        else:
            st.info("No 'Department' column found in the dataset.")
    with cc2:
        st.markdown("#### Attrition Rate by Tenure Band")
        if 'Years_at_Company' in dash_df.columns:
            bins = [0, 1, 3, 5, 10, 100]
            band_labels = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]
            dash_df['_TenureBand'] = pd.cut(dash_df['Years_at_Company'], bins=bins, labels=band_labels, right=False)
            tenure_stats = dash_df.groupby('_TenureBand', observed=True)['_Attrited'].mean().mul(100)
            st.bar_chart(tenure_stats)
        else:
            st.info("No 'Years_at_Company' column found in the dataset.")

    # --- DRILL-DOWN DETAIL (from clicking a KPI card) ---
    if st.session_state.selected_kpi:
        st.divider()
        st.markdown("### 🔎 Detail View")
        sel = st.session_state.selected_kpi

        if sel == "headcount":
            if 'Department' in dash_df.columns:
                st.dataframe(dash_df['Department'].value_counts().rename("Headcount"), use_container_width=True)
            else:
                st.write(f"Total employees in current filter: **{headcount:,}**")

        elif sel == "attrition":
            if 'Department' in dash_df.columns:
                st.dataframe(
                    (dash_df.groupby('Department')['_Attrited'].mean() * 100).round(1)
                    .sort_values(ascending=False).rename("Attrition %"),
                    use_container_width=True
                )
            else:
                st.write(f"Overall attrition rate: **{attrition_rate:.1f}%**")

        elif sel == "positions":
            st.info("Open headcount requisitions aren't part of this employee-level dataset. "
                     "To make this KPI live, add an 'Open_Positions' source (e.g. a separate "
                     "recruiting CSV keyed by Department) and I can wire it in.")

        elif sel == "tenure":
            if 'Years_at_Company' in dash_df.columns:
                st.bar_chart(dash_df['Years_at_Company'].value_counts().sort_index())
            else:
                st.write("No 'Years_at_Company' column found in the dataset.")

        if st.button("✖️ Close detail view"):
            st.session_state.selected_kpi = None
            st.rerun()

    st.divider()
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ec/World_map_blank_without_borders.svg",
             caption="Global Operations Map")

elif page == "👥 Employee Directory":
    st.markdown("## Employee Directory")
    safe_df = df.drop(columns=['Attrition'], errors='ignore').head(50)
    st.dataframe(safe_df, use_container_width=True)

elif page == "🤖 Retention Copilot":
    st.markdown(f'<span class="model-badge">MODEL: {model_choice.upper()}</span>', unsafe_allow_html=True)
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

            if not groq_ready:
                st.error("⚠️ Please enter a valid Groq API Key in the sidebar.")
            else:
                with st.spinner(f"Crunching numbers with {model_choice}..."):
                    X_in = target_row.drop(
                        columns=['Employee_ID', 'Attrition'], errors='ignore')

                    # --- Prediction (works for sklearn Pipelines AND the NN wrapper) ---
                    try:
                        prob = pipeline.predict_proba(X_in)[0][1]
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        st.stop()

                    # --- SHAP explanation, model-specific ---
                    vals = None
                    feats = None
                    try:
                        prep_step = get_preprocessor(pipeline)
                        X_bg = prep_step.transform(
                            df.drop(columns=['Employee_ID', 'Attrition'], errors='ignore').head(50))
                        X_proc = prep_step.transform(X_in)
                        vals = compute_shap(pipeline, model_choice, prep_step, X_proc, X_bg)
                        feats = prep_step.get_feature_names_out()
                    except Exception as e:
                        st.warning(f"⚠️ SHAP explanation unavailable for this model ({e}). "
                                   "Showing prediction without feature attribution.")
                        n_feats = X_in.shape[1]
                        vals = np.zeros(n_feats)
                        feats = X_in.columns.tolist()

                    if len(feats) != len(vals):
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
                        strategy_text = call_gemini(api_key, strategy_prompt)
                    except Exception as e:
                        strategy_text = f"⚠️ AI Error: Could not generate strategy ({e})."

                    # Save Analysis State
                    st.session_state.analysis_data = {
                        "id": emp_id,
                        "role": role_val,
                        "prob": prob,
                        "top_5": top_5,
                        "drivers_txt": drivers_txt,
                        "risk_label": risk_label,
                        "model_used": model_choice,
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
            st.caption(f"Scored using **{data.get('model_used', 'Unknown')}**")

        with res_c2:
            st.markdown("### Why? (Top Drivers)")
            top_5_view = data['top_5'].copy()
            top_5_view['Feature'] = top_5_view['Feature'].astype(str).str.replace(
                'cat__', '').str.replace('num__', '').str.replace('ord__', '')

            def fmt_arrow(x):
                return "⬆️ Increases Risk" if x > 0 else "⬇️ Decreases Risk"

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
            if not groq_ready:
                st.error("⚠️ Please enter a valid Groq API Key in the sidebar.")
            else:
                with st.spinner("Drafting..."):
                    draft_prompt = f"""
                    Context: Employee #{data['id']}, Status: {data['risk_label']} ({data['prob']:.1%}).
                    Drivers: {data['drivers_txt']}.
                    Task: Write a {action_type}.
                    Tone: Professional.
                    """
                    try:
                        draft_text = call_gemini(api_key, draft_prompt)
                        st.success(f"Draft Generated: {action_type}")
                        st.markdown(draft_text)
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
            if not groq_ready:
                st.error("⚠️ Please enter a valid Groq API Key in the sidebar.")
            else:
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
                            chat_text = call_gemini(api_key, chat_prompt)
                            st.markdown(chat_text)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": chat_text})
                        except Exception as e:
                            st.error(f"AI Error: {e}")
