import streamlit as st
import pandas as pd
import numpy as np

# Import หน้า Tab ที่แยกไว้ (ต้องสร้างโฟลเดอร์ tabs และไฟล์ __init__.py ก่อนนะ)
from tabs import tab_data, tab_table1, tab_diag, tab_corr, tab_logit

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
    np.random.seed(42); n = 150
    data = {
        'ID': range(1, n+1),
        'Group_Treatment': np.random.choice(['Standard Care', 'New Drug'], n),
        'Age': np.random.normal(60, 12, n).astype(int),
        'Sex': np.random.choice([0, 1], n),
        'BMI': np.random.normal(25, 4, n).round(1),
        'Hypertension': np.random.binomial(1, 0.4, n),
        'Risk_Score': np.random.normal(5, 2, n).round(2)
    }
    # Logistic Prob
    p = 1 / (1 + np.exp(-(data['Risk_Score'] - 6)*0.8))
    data['Outcome_Disease'] = np.random.binomial(1, p)
    
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {
        'Sex': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}},
        'Hypertension': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}},
        'Outcome_Disease': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}}
    }
    st.sidebar.success("Loaded!")
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

# 🟢 NEW: เพิ่มเครดิตที่ส่วนท้ายของ Sidebar (เฉพาะเมื่อมีการโหลดข้อมูลแล้ว)

# ==========================================
# 2. MAIN AREA
# ==========================================
if st.session_state.df is not None:
    df = st.session_state.df

    # Create Tabs
    t0, t1, t2, t3, t4 = st.tabs([
        "📄 Raw Data", 
        "📋 Baseline Table 1", 
        "🔬 Diagnostic Test",
        "🔗 Correlation",
        "📊 Logistic Regression" 
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

    st.session_state.markdown("""
    <hr style="margin-top: 20px; margin-bottom: 10px; border-color: var(--border-color);">
    <div style='text-align: center; font-size: 0.7em; color: var(--text-color);'>
      &copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit
    </div>
    """, unsafe_allow_html=True)

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
    
st.session_state.markdown("""
<hr style="margin-top: 20px; margin-bottom: 10px; border-color: var(--border-color);">
<div style='text-align: center; font-size: 0.7em; color: var(--text-color);'>
  &copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit
</div>
""", unsafe_allow_html=True)

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
