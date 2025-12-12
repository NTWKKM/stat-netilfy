import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit.components.v1 as components 

import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit.components.v1 as components 

# ==========================================
# 1. ย้าย CONFIG และ LOADING SCREEN KILLER มาไว้บนสุด
# ==========================================
st.set_page_config(page_title="Medical Stat Tool", layout="wide", menu_items={
        'Get Help': 'https://ntwkkm.github.io/infos/stat_manual.html',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
    })

st.title("🏥 Medical Statistical Tool")

# สั่งปิด Loading Screen ทันที เพื่อให้ถ้ามี Error ด้านล่าง เราจะยังเห็นข้อความ Error ได้
components.html("""
<script>
    var loader = window.parent.document.getElementById('loading-screen');
    if (loader) {
        loader.style.opacity = '0';
        setTimeout(function() {
            loader.style.display = 'none';
        }, 500);
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
    n = 300 # 🟢 ปรับเพิ่ม n เล็กน้อยเพื่อให้ PSM เห็นผลชัดขึ้น
    
    # 🟢 1. Predictors (สร้างตัวแปรพื้นฐานก่อน เพื่อใช้กำหนด Bias)
    age = np.random.normal(60, 10, n).astype(int)
    sex = np.random.choice([0, 1], n)
    bmi = np.random.normal(27, 4, n).round(1)
    
    # 🟢 2. Confounder: Hypertension (ขึ้นอยู่กับ Age และ BMI)
    prob_hyp = 1 / (1 + np.exp(-( -3 + 0.05*age + 0.1*bmi )))
    hypertension = np.random.binomial(1, prob_hyp)

    # 🟢 3. Group Variable (Treatment Selection Bias สำหรับ PSM)
    # คนที่อายุเยอะและมีความดัน มีโอกาสได้ยาใหม่ (Group 1) มากกว่า -> เกิด Selection Bias
    prob_treat = 1 / (1 + np.exp(-( -2 + 0.04*age + 1.5*hypertension )))
    group = np.random.binomial(1, prob_treat) 

    # 4. Risk_Score (Continuous, significantly different between groups)
    # Group 0 (Standard) has higher score
    risk_score_base = np.where(group == 0, 6 + np.random.normal(0, 1.5), 4 + np.random.normal(0, 1.5))
    risk_score = risk_score_base + 0.05 * age 
    risk_score = risk_score.round(2)
    
    # 5. Outcome_Disease (Binary)
    log_p = 1 / (1 + np.exp(-(-4 + 0.8 * risk_score - 1.2 * group))) 
    outcome_disease = np.random.binomial(1, log_p)
    
    # 6. Correlation Variable: Inflammation_Marker
    inflammation_marker = 5 + 0.8 * bmi + np.random.normal(0, 1.0, n)
    inflammation_marker = inflammation_marker.round(1)

    # 7. Survival Variables (Time and Event)
    scale_param = np.where(group == 0, 150, 400)
    time_days = np.random.exponential(scale=scale_param, size=n)
    time_days = time_days.clip(min=1, max=1000).astype(int) 
    
    event_prob_base = np.where(group == 0, 0.8, 0.4)
    event_prob = event_prob_base - 0.0003 * time_days
    event_prob = event_prob.clip(min=0.1, max=0.9) 
    event_death = np.random.binomial(1, event_prob)
    
    # 8. Agreement Variables (Cohen's Kappa)
    diag_dr_a = np.random.binomial(1, 0.3, n)
    diag_dr_b = diag_dr_a.copy()
    num_mismatch = int(0.12 * n) 
    mismatch_idx = np.random.choice(n, num_mismatch, replace=False)
    diag_dr_b[mismatch_idx] = 1 - diag_dr_b[mismatch_idx] 

    # 9. Reliability Variables (ICC)
    sbp_machine_1 = np.random.normal(120, 15, n).round(0)
    sbp_machine_2 = sbp_machine_1 + np.random.normal(2, 3, n) 
    sbp_machine_2 = sbp_machine_2.round(0)

    # Create DataFrame and Metadata
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
        'Time_Days': time_days, 
        'Event_Death': event_death,
        'Diagnosis_Dr_A': diag_dr_a,
        'Diagnosis_Dr_B': diag_dr_b,
        'SBP_Machine_1': sbp_machine_1,
        'SBP_Machine_2': sbp_machine_2
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

    with t1: tab_table1.render(df_clean, st.session_state.var_meta)
    with t2: tab_diag.render(df_clean, st.session_state.var_meta)
    with t3: tab_corr.render(df_clean)
    with t4: tab_logit.render(df_clean, st.session_state.var_meta)
    with t5: tab_survival.render(df_clean, st.session_state.var_meta)
    with t6: tab_psm.render(df_clean, st.session_state.var_meta)
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
