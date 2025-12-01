import streamlit as st
import pandas as pd
import numpy as np

# Import หน้า Tab ที่แยกไว้ (เพิ่ม tab_survival)
from tabs import tab_data, tab_table1, tab_diag, tab_corr, tab_logit, tab_survival

# --- CONFIGURATION ---
st.set_page_config(page_title="Medical Stat Tool", layout="wide")
st.title("🏥 Medical Statistical Tool")

# --- INITIALIZE STATE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'var_meta' not in st.session_state: st.session_state.var_meta = {}

# --- SIDEBAR (ยังคงไว้ใน app.py เพราะเป็น Global Control) ---
st.sidebar.title("MENU")
st.sidebar.header("1. Data Management")

# Example Data Generator
if st.sidebar.button("📄 Load Example Data"):
    np.random.seed(42)
    n = 200 # Increased n for better significance
    
    # 1. Group Variable (0=Standard Care, 1=New Drug) - Use 0/1 for computation
    group = np.random.choice([0, 1], n, p=[0.5, 0.5]) 

    # 2. Predictors (Age, BMI, Sex)
    age = np.random.normal(60, 10, n).astype(int)
    sex = np.random.choice([0, 1], n)
    bmi = np.random.normal(27, 4, n).round(1)
    
    # 3. Hypertension (Categorical, significantly different between groups for Table 1)
    p_hyper = np.where(group == 0, 0.65, 0.3) # Higher prevalence in Standard Care (Group 0)
    hypertension = np.random.binomial(1, p_hyper)
    
    # 4. Risk_Score (Continuous, significantly different between groups, good for Diagnostic/Table 1)
    # Group 0 (Standard) has higher score, and correlates with age
    risk_score_base = np.where(group == 0, 6 + np.random.normal(0, 1.5), 4 + np.random.normal(0, 1.5))
    risk_score = risk_score_base + 0.05 * age # Introduce slight age effect
    risk_score = risk_score.round(2)
    
    # 5. Outcome_Disease (Binary, significant prediction from Risk_Score and Group - Logistic Regression)
    # Logit: -4 + 0.8 * Risk_Score - 1.2 * Group (Group 1 lowers the risk)
    log_p = 1 / (1 + np.exp(-(-4 + 0.8 * risk_score - 1.2 * group))) 
    outcome_disease = np.random.binomial(1, log_p)
    
    # 6. Correlation Variable: Inflammation_Marker (correlated with BMI - Correlation Tab)
    inflammation_marker = 5 + 0.8 * bmi + np.random.normal(0, 1.0, n)
    inflammation_marker = inflammation_marker.round(1)

    # 7. Survival Variables (Time and Event) - Significant difference between groups (Survival Analysis)
    # Standard Care (0) has shorter survival (scale 150), New Drug (1) has longer survival (scale 400)
    scale_param = np.where(group == 0, 150, 400)
    time_days = np.random.exponential(scale=scale_param, size=n)
    time_days = time_days.clip(min=1, max=1000).astype(int) # Max follow-up 1000 days
    
    # Event: Probability of event (death) is higher for Group 0 and shorter times
    event_prob_base = np.where(group == 0, 0.8, 0.4)
    event_prob = event_prob_base - 0.0003 * time_days
    event_prob = event_prob.clip(min=0.1, max=0.9) 
    event_death = np.random.binomial(1, event_prob)
    
    # Create DataFrame and Metadata
    data = {
        'ID': range(1, n+1),
        # Convert group to string for the DataFrame as in original code
        'Group_Treatment': np.where(group == 0, 'Standard Care', 'New Drug'), 
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Hypertension': hypertension, # 0/1
        'Risk_Score': risk_score, # Continuous
        'Inflammation_Marker': inflammation_marker, # Continuous
        'Outcome_Disease': outcome_disease, # 0/1
        'Time_Days': time_days, # Continuous
        'Event_Death': event_death # 0/1 (Status)
    }
    
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {
        'Sex': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}},
        'Hypertension': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}},
        'Outcome_Disease': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}},
        'Event_Death': {'type':'Categorical', 'map':{0:'Censored', 1:'Event (Death)'}} # NEW: สำหรับ Survival Analysis
    }
    
    st.sidebar.success(f"Loaded {n} Example Patients!")
    st.experimental_rerun()

# File Uploader
upl = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if upl:
    try:
        if upl.name.endswith('.csv'): st.session_state.df = pd.read_csv(upl)
        else: st.session_state.df = pd.read_excel(upl)
        st.sidebar.success("File Uploaded!")
    except Exception as e: st.sidebar.error(f"Error: {e}")

# Variable Settings (Metadata)
if st.session_state.df is not None:
    st.sidebar.header("2. Settings")
    cols = st.session_state.df.columns.tolist()
    s_var = st.sidebar.selectbox("Edit Var:", ["Select..."] + cols)
    if s_var != "Select...":
        meta = st.session_state.var_meta.get(s_var, {})
        n_type = st.sidebar.radio("Type:", ['Auto-detect', 'Categorical', 'Continuous'], 
                                  index=['Auto-detect', 'Categorical', 'Continuous'].index(meta.get('type','Auto-detect')))
        st.sidebar.markdown("Labels (0=No):")
        map_txt = st.sidebar.text_area("Map", value="\n".join([f"{k}={v}" for k,v in meta.get('map',{}).items()]), height=80)
        
        if st.sidebar.button("💾 Save"):
            new_map = {}
            for line in map_txt.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    try: 
                        k=k.strip()
                        if k.replace('.','',1).isdigit(): k = float(k) if '.' in k else int(k)
                        new_map[k] = v.strip()
                    except: pass
            if s_var not in st.session_state.var_meta: st.session_state.var_meta[s_var]={}
            st.session_state.var_meta[s_var]['type'] = n_type
            st.session_state.var_meta[s_var]['map'] = new_map
            st.sidebar.success("Saved!")
            st.experimental_rerun()

# ==========================================
# 2. MAIN AREA
# ==========================================
if st.session_state.df is not None:
    df = st.session_state.df

    # Create Tabs (เพิ่ม Survival Analysis)
    t0, t1, t2, t3, t4, t5 = st.tabs([
        "📄 Raw Data", 
        "📋 Baseline Table 1", 
        "🔬 Diagnostic Test",
        "🔗 Correlation",
        "📊 Logistic Regression",
        "⏳ Survival Analysis" # <--- NEW TAB
    ])

    # Call Modules
    with t0:
        st.session_state.df = tab_data.render(df) 
        
    with t1:
        tab_table1.render(df, st.session_state.var_meta)

    with t2:
        tab_diag.render(df, st.session_state.var_meta)

    with t3:
        tab_corr.render(df)

    with t4:
        tab_logit.render(df, st.session_state.var_meta)

    with t5: # <--- NEW RENDER CALL
        tab_survival.render(df, st.session_state.var_meta)
        
else:
    st.info("👈 Please load example data or upload a file to start.")
    st.markdown("""
### ✨ All Statistical Features:

1.  **Raw Data Management:**
    * Load/Import data files (CSV, Excel, etc.).
    * Performs **automatic data type detection** and preliminary structure checking.

2.  **Baseline Characteristics (Table 1):**
    * Auto-generated summary table (Mean ± SD / Median IQR, Count %).
    * **Automated P-value Selection:** Automatically selects the correct statistical test (**t-test, Mann-Whitney U, Chi-square, Fisher's Exact, etc.**) based on variable type and distribution.

3.  **Diagnostic Test & Statistics:**
    * **ROC Curve Analysis:** Evaluates test performance using AUC and identifies the Optimal Cut-off Point.
    * **Chi-square & Risk Analysis:** Generates a structured Contingency Table and calculates Risk Ratios and Odds Ratios (OR).

4.  **Continuous Correlation (Pearson/Spearman):**
    * Measures **Linear** (Pearson) or **Monotonic** (Spearman) association between two continuous variables.

5.  **Binary Logistic Regression:**
    * Univariate & Multivariate Analysis.
    * Calculates **Odds Ratio (OR)** and **Adjusted Odds Ratio (AOR)**, controlling for confounding variables.
    """)
    
# ==========================================
# 3. GLOBAL CSS (Cleanup)
# ==========================================

# 🟢 NEW: Inject CSS to hide the default Streamlit footer (Keep this part)
st.markdown("""
<style>
/* โค้ดสำหรับซ่อน Streamlit footer เดิม (ยังจำเป็นต้องมี) */
footer {
    visibility: hidden;
    height: 0px;
}
footer:after {
    content: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<hr style="margin-top: 20px; margin-bottom: 10px; border-color: var(--border-color); opacity: 0.5;">
<div style='text-align: center; font-size: 0.8em; color: var(--text-color); opacity: 0.8;'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit; font-weight:bold;">NTWKKM</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
</div>
""", unsafe_allow_html=True)
