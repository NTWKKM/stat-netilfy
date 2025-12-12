import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit.components.v1 as components 

# ==========================================
# 1. CONFIG & LOADING SCREEN KILLER (Must be First)
# ==========================================
st.set_page_config(
    page_title="Medical Stat Tool", 
    layout="wide", 
    menu_items={
        'Get Help': 'https://ntwkkm.github.io/infos/stat_manual.html',
        # 🟢 แก้จุดที่ 1: เปลี่ยนลิงก์เป็น GitHub Issues ของคุณ (หรือลบบรรทัดนี้ทิ้งถ้ายังไม่มี)
        'Report a bug': "https://github.com/NTWKKM/stat-netilfy/issues", 
    }
)

st.title("🏥 Medical Statistical Tool")

# 🟢 แก้จุดที่ 2: ใช้ try-catch เพื่อความปลอดภัย (Safe Loader Removal)
components.html("""
<script>
    try {
        var loader = window.parent && window.parent.document
            ? window.parent.document.getElementById('loading-screen')
            : null;
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(function() {
                loader.style.display = 'none';
            }, 500);
        }
    } catch (e) {
        // no-op (ถ้ามี error ก็ปล่อยผ่าน เว็บจะไม่พัง)
        console.log("Loader removal error: " + e);
    }
</script>
""", height=0)

# ==========================================
# 2. ค่อยเริ่ม IMPORT MODULES (จุดเสี่ยง Error)
# ==========================================
try:
    # พยายาม import ไฟล์ ถ้าไฟล์ไหนมีปัญหามันจะแจ้งเตือนให้เห็นบนหน้าจอแทนการหมุนค้าง
    from tabs import tab_data, tab_table1, tab_diag, tab_corr, tab_logit, tab_survival, tab_psm, tab_adv_survival
except (KeyboardInterrupt, SystemExit):
    raise
except Exception as e:
    st.exception(e)
    st.stop()

# --- INITIALIZE STATE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'var_meta' not in st.session_state: st.session_state.var_meta = {}

# --- SIDEBAR (ยังคงไว้ใน app.py เพราะเป็น Global Control) ---
st.sidebar.title("MENU")
st.sidebar.header("1. Data Management")

# Example Data Generator
if st.sidebar.button("📄 Load Example Data"):
    np.random.seed(42)
    n = 500 # 🟢 เพิ่ม N เป็น 500 เพื่อให้ p-value significant ง่ายขึ้น
    
    # --- 1. Baseline Characteristics (Predictors) ---
    # Age: Normal Dist
    age = np.random.normal(60, 12, n).astype(int)
    
    # Sex: 0/1 (Balanced)
    sex = np.random.binomial(1, 0.5, n)
    
    # BMI: Normal Dist
    bmi = np.random.normal(25, 4, n).round(1)
    
    # Hypertension (Confounder): สัมพันธ์กับ Age และ BMI อย่างชัดเจน (Logistic)
    # logit = -10 + 0.1*Age + 0.1*BMI
    p_hyp = 1 / (1 + np.exp(-( -10 + 0.1*age + 0.15*bmi )))
    hypertension = np.random.binomial(1, p_hyp)

    # --- 2. Treatment Assignment (Selection Bias for PSM) ---
    # กำหนดให้ Group สัมพันธ์กับทุกตัวแปร (Age, Sex, BMI, HT) เพื่อให้ Table 1 Significant (Imbalanced)
    # และเพื่อให้ PSM มีหน้าที่ในการแก้ Bias นี้
    logit_treat = -3 + 0.05*age + 0.5*sex + 0.1*bmi + 1.2*hypertension
    p_treat = 1 / (1 + np.exp(-logit_treat))
    group = np.random.binomial(1, p_treat) 
    
    # --- 3. Outcome & Risk Score ---
    # Risk Score: สร้างให้ต่างกันชัดเจนระหว่างกลุ่ม (T-test Sig)
    # Group 1 (New Drug) ควรจะมี Risk Score ต่ำกว่า Group 0 (Standard) หรือกลับกัน
    base_score = np.where(group == 1, 3.5, 6.0) # Mean ต่างกันเยอะ
    risk_score = base_score + np.random.normal(0, 1.5, n) + 0.02*age
    risk_score = risk_score.round(2)
    
    # Outcome Disease (Binary): สัมพันธ์กับ Risk Score และ Group อย่างชัดเจน (Logistic/Chi2 Sig)
    # ใส่ effect ของ Hypertension เข้าไปด้วยเพื่อให้ Chi-Square (HT vs Outcome) significant
    logit_outcome = -4 + 0.8*risk_score - 1.5*group + 1.0*hypertension
    p_outcome = 1 / (1 + np.exp(-logit_outcome))
    outcome_disease = np.random.binomial(1, p_outcome)
    
    # --- 4. Correlation Variable ---
    # Inflammation Marker: สัมพันธ์กับ BMI แบบ Linear (Pearson r สูง)
    inflammation_marker = 10 + 1.5 * bmi + np.random.normal(0, 5, n)
    inflammation_marker = inflammation_marker.round(1)

    # --- 5. Survival Analysis ---
    # Time: Group 1 อยู่ได้นานกว่า Group 0 ชัดเจน (Log-rank Sig)
    # Scale (Mean survival time): Group 0=200 days, Group 1=500 days
    scale_param = np.where(group == 0, 200, 500)
    time_days = np.random.exponential(scale=scale_param, size=n)
    time_days = time_days.clip(min=1, max=1800).astype(int)
    
    # Event: สัมพันธ์กับ Time (ยิ่งนานยิ่งตายน้อยลง? หรือตัด Censored)
    # ให้ Group 0 ตายเยอะกว่า (Event rate สูง)
    p_event = np.where(group == 0, 0.7, 0.3) 
    event_death = np.random.binomial(1, p_event)
    
    # For Time-Dependent Cox (Structure only)
    time_start = np.zeros(n, dtype=int)
    time_stop = time_days # ใช้เวลาตายเป็นจุดสิ้นสุด
    
    # --- 6. Reliability & Agreement ---
    # Cohen's Kappa: Dr A vs Dr B (High Agreement)
    diag_dr_a = np.random.binomial(1, 0.4, n)
    diag_dr_b = diag_dr_a.copy()
    # Flip 5% of data to create minor disagreement (High Kappa)
    mismatch_idx = np.random.choice(n, int(0.05*n), replace=False)
    diag_dr_b[mismatch_idx] = 1 - diag_dr_b[mismatch_idx]
    
    # ICC: Machine 1 vs Machine 2 (High Correlation)
    sbp_m1 = np.random.normal(130, 15, n).round(0)
    sbp_m2 = sbp_m1 + np.random.normal(0, 2, n) # Noise น้อยมาก
    sbp_m2 = sbp_m2.round(0)

    # Create DataFrame
    data = {
        'ID': range(1, n+1),
        'Group_Treatment': np.where(group == 0, 'Standard Care', 'New Drug'), 
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Hypertension': hypertension, 
        'Risk_Score': risk_score, 
        'Inflammation_Marker': inflammation_marker, 
        'Outcome_Disease': outcome_disease,
        # Survival Cols
        'Time_Start': time_start, # เพิ่มเพื่อรองรับ Time Cox Tab
        'Time_Stop': time_stop,
        'Event_Death': event_death,
        # Diag/Rel Cols
        'Diagnosis_Dr_A': diag_dr_a,
        'Diagnosis_Dr_B': diag_dr_b,
        'SBP_Machine_1': sbp_m1,
        'SBP_Machine_2': sbp_m2
    }
    
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {
        'Sex': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}},
        'Hypertension': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}},
        'Outcome_Disease': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}},
        'Event_Death': {'type':'Categorical', 'map':{0:'Censored', 1:'Event (Death)'}},
        'Diagnosis_Dr_A': {'type':'Categorical', 'map':{0:'Negative', 1:'Positive'}},
        'Diagnosis_Dr_B': {'type':'Categorical', 'map':{0:'Negative', 1:'Positive'}}
    }
    
    st.sidebar.success(f"Loaded {n} Example Patients!")
    st.rerun()

# File Uploader
upl = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if upl:
    try:
        if upl.name.endswith('.csv'): st.session_state.df = pd.read_csv(upl)
        else: st.session_state.df = pd.read_excel(upl)
        st.sidebar.success("File Uploaded!")
    except Exception as e: st.sidebar.error(f"Error: {e}")
        
if st.sidebar.button("⚠️ Reset All Data", type="primary"):
    st.session_state.clear()
    st.rerun()

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
            st.rerun()

# ==========================================
# 2. MAIN AREA
# ==========================================
if st.session_state.df is not None:
    df = st.session_state.df 

    # 🟢 FIX 2: เพิ่ม Tab t7 สำหรับ Advanced Survival Analysis
    t0, t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📄 Raw Data", 
        "📋 Baseline Table 1", 
        "🔬 Diagnostic Test", 
        "🔗 Correlation",
        "📊 Logistic Regression",
        "⏳ Survival Analysis",
        "⚖️ Propensity Score",
        "📈 Time Cox Regs" # 🟢 New Tab
    ])

    # Call Modules
    with t0:
        st.session_state.df = tab_data.render(df) 
        custom_na = st.session_state.get('custom_na_list', [])
        df_clean = tab_data.get_clean_data(st.session_state.df, custom_na)

    with t1:
        tab_table1.render(df_clean, st.session_state.var_meta)
    with t2:
        tab_diag.render(df_clean, st.session_state.var_meta)
    with t3:
        tab_corr.render(df_clean)
    with t4:
        tab_logit.render(df_clean, st.session_state.var_meta)
    with t5:
        tab_survival.render(df_clean, st.session_state.var_meta)
    with t6:
        tab_psm.render(df_clean, st.session_state.var_meta)
    with t7:
        tab_adv_survival.render(df_clean, st.session_state.var_meta)
        
else:
    st.info("👈 Please load example data or upload a file to start.")
    st.markdown("""
### ✨ All Statistical Features:
1.  **Raw Data Management**
2.  **Baseline Characteristics (Table 1)**
3.  **Diagnostic Test & Statistics**
4.  **Continuous Correlation**
5.  **Binary Logistic Regression**
6.  **Survival Analysis**
7.  **Propensity Score Matching**
8.  **Time-Dependent Cox Regression (New!)**
    """)
    
# ==========================================
# 3. GLOBAL CSS (Cleanup)
# ==========================================

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
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit; font-weight:bold;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
</div>
""", unsafe_allow_html=True)
