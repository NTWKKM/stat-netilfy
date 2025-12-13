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
    np.random.seed(999) # Fixed seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง
    n = 600 # เพิ่ม N เป็น 600 เพื่อให้ P-value เห็นผลชัดเจน
    
    # --- 1. Demographics & Confounders (Table 1 & PSM) ---
    # Age: Normal distribution
    age = np.random.normal(55, 12, n).astype(int).clip(20, 90)
    
    # Sex: Binary (0=Female, 1=Male)
    sex = np.random.binomial(1, 0.55, n)
    
    # BMI: Normal distribution
    bmi = np.random.normal(24, 4, n).round(1)
    
    # Comorbidity (Binary): สัมพันธ์กับ Age และ BMI (เป็น Confounder)
    logit_comorb = -5 + 0.05*age + 0.1*bmi
    p_comorb = 1 / (1 + np.exp(-logit_comorb))
    comorbidity = np.random.binomial(1, p_comorb)

    # --- 2. Treatment Assignment (Selection Bias for PSM) ---
    # กำหนดให้คนที่มี Comorbidity หรือ Age มาก มีโอกาสได้ยาใหม่ (Group 1) มากกว่า
    # สิ่งนี้ทำให้ Baseline Table 1 ไม่เท่ากัน (Bias) -> ต้องแก้ด้วย PSM
    logit_treat = -2 + 1.5*comorbidity - 0.02*age
    p_treat = 1 / (1 + np.exp(-logit_treat))
    group = np.random.binomial(1, p_treat) 

    # --- 3. Outcome & Survival ---
    # Survival: Group 1 (New Drug) อยู่ได้นานกว่า (Protective Effect)
    # Hazard Ratio: Group=0.5 (ลดเสี่ยง), Comorbidity=1.5 (เพิ่มเสี่ยง)
    lambda_base = 0.02
    hazard = lambda_base * np.exp(0.4*comorbidity - 0.8*group)
    surv_time = np.random.exponential(1/hazard)
    
    # Censoring
    censor_time = np.random.uniform(0, 100, n)
    time_obs = np.minimum(surv_time, censor_time).round(1)
    event_death = (surv_time <= censor_time).astype(int) # 1=Dead, 0=Censored

    # --- 4. Diagnostic Test (สำคัญ: ปรับให้มี Sens/Spec) ---
    # Gold Standard: โรคจริง
    gold_std = np.random.binomial(1, 0.3, n) # ความชุก 30%
    
    # Rapid Test: สร้างให้ตรงกับ Gold Standard ประมาณ 85-90%
    # ถ้าเป็นโรค (1) ตรวจเจอ (1) 85% (Sensitivity)
    # ถ้าไม่เป็นโรค (0) ตรวจเจอ (1) 10% (False Positive -> Spec 90%)
    prob_test = np.where(gold_std==1, 0.85, 0.10)
    rapid_test = np.random.binomial(1, prob_test)
    
    # Inter-rater (Kappa): Dr.A vs Dr.B
    # ให้ Dr.A วินิจฉัยใกล้เคียงความจริง
    dr_a = np.where(gold_std==1, np.random.binomial(1, 0.9, n), np.random.binomial(1, 0.1, n))
    # ให้ Dr.B เห็นตรงกับ Dr.A 90% (High Agreement)
    agree_noise = np.random.binomial(1, 0.90, n)
    dr_b = np.where(agree_noise==1, dr_a, 1-dr_a)

    # --- 5. Correlation ---
    lab_alb = np.random.normal(3.5, 0.5, n).round(2)
    # Ca สัมพันธ์กับ Alb
    lab_ca = 2 + 1.5*lab_alb + np.random.normal(0, 0.3, n)
    lab_ca = lab_ca.round(2)

    # Create DataFrame
    data = {
        'ID': range(1, n+1),
        'Group_Treatment': group, 
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Comorbidity': comorbidity,
        # Survival
        'Time_Months': time_obs,
        'Status_Death': event_death,
        # Diagnostic
        'Gold_Standard': gold_std,
        'Rapid_Test': rapid_test,
        'Diagnosis_Dr_A': dr_a,
        'Diagnosis_Dr_B': dr_b,
        # Correlation
        'Lab_Albumin': lab_alb,
        'Lab_Calcium': lab_ca,
        # For Time Cox (Placeholder)
        'T_Start': np.zeros(n, dtype=int),
        'T_Stop': time_obs
    }
    
    st.session_state.df = pd.DataFrame(data)
    
    # Set Metadata (ให้แสดงผลสวยๆ อัตโนมัติ)
    st.session_state.var_meta = {
        'Group_Treatment': {'type':'Categorical', 'map':{0:'Standard Care', 1:'New Drug'}},
        'Sex': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}},
        'Comorbidity': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}},
        'Status_Death': {'type':'Categorical', 'map':{0:'Censored', 1:'Dead'}},
        'Gold_Standard': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}},
        'Rapid_Test': {'type':'Categorical', 'map':{0:'Negative', 1:'Positive'}},
        'Diagnosis_Dr_A': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}},
        'Diagnosis_Dr_B': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}}
    }
    
    st.sidebar.success(f"Loaded {n} Example Patients! Ready for all tabs.")
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
