import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import hashlib
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIG & LOADING SCREEN KILLER (Must be First)
# ==========================================
st.set_page_config(
    page_title="Medical Stat Tool", 
    layout="wide", 
    menu_items={
        'Get Help': 'https://ntwkkm.github.io/pl/infos/stat_manual.html',
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
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {}
# 🟢 เพิ่ม State สำหรับการติดตามไฟล์ที่อัปโหลด (เพื่อป้องกันการโหลดซ้ำ)
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
    
# --- SIDEBAR (ยังคงไว้ใน app.py เพราะเป็น Global Control) ---
st.sidebar.title("MENU")
st.sidebar.header("1. Data Management")

# Example Data Generator
if st.sidebar.button("📄 Load Example Data"):
    np.random.seed(999) # Fixed seed
    n = 600 
    
    # --- 1. Demographics & Confounders ---
    age = np.random.normal(55, 12, n).astype(int).clip(20, 90)
    sex = np.random.binomial(1, 0.55, n)
    bmi = np.random.normal(24, 4, n).round(1).clip(10, 60)
    
    # Comorbidity
    logit_comorb = -5 + 0.05*age + 0.1*bmi
    p_comorb = 1 / (1 + np.exp(-logit_comorb))
    comorbidity = np.random.binomial(1, p_comorb)

    # --- 2. Treatment Assignment (Selection Bias) ---
    logit_treat = -2 + 1.5*comorbidity - 0.02*age
    p_treat = 1 / (1 + np.exp(-logit_treat))
    group = np.random.binomial(1, p_treat) 

    # --- 3. Survival Outcome ---
    lambda_base = 0.02
    hazard = lambda_base * np.exp(0.4*comorbidity - 0.8*group)
    surv_time = np.random.exponential(1/hazard)
    
    censor_time = np.random.uniform(0, 100, n)
    time_obs = np.minimum(surv_time, censor_time).round(1)
    time_obs = np.maximum(time_obs, 0.1) # avoid zero/invalid durations
    event_death = (surv_time <= censor_time).astype(int)

    # --- 4. Logistic Regression Outcome [NEW] ---
    # Outcome: Cured (1=หาย, 0=ไม่หาย)
    # ปัจจัย: ยาใหม่ (Group 1) ช่วยให้หายง่ายขึ้น, แต่อายุมากและโรคประจำตัวทำให้หายยาก
    logit_cure = 0.5 + 1.2*group - 0.03*age - 0.8*comorbidity
    p_cure = 1 / (1 + np.exp(-logit_cure))
    outcome_cured = np.random.binomial(1, p_cure)

    # --- 5. Diagnostic Test ---
    gold_std = np.random.binomial(1, 0.3, n)
    
    # Rapid Test (Continuous)
    rapid_test_val = np.where(gold_std==1, 
                              np.random.normal(55, 15, n), 
                              np.random.normal(35, 10, n))
    rapid_test_val = np.maximum(rapid_test_val, 0).round(1)
    
    # Kappa Raters
    dr_a = np.where(gold_std==1, np.random.binomial(1, 0.9, n), np.random.binomial(1, 0.1, n))
    agree_noise = np.random.binomial(1, 0.85, n)
    dr_b = np.where(agree_noise==1, dr_a, 1-dr_a)

    # --- 6. Correlation ---
    lab_alb = np.random.normal(3.5, 0.5, n).round(2)
    lab_ca = 2 + 1.5*lab_alb + np.random.normal(0, 0.3, n)
    lab_ca = lab_ca.round(2)

    # --- 7. ICC Data ---
    icc_rater1 = np.random.normal(50, 10, n).round(1)
    icc_rater2 = icc_rater1 + np.random.normal(0, 3, n)
    icc_rater2 = icc_rater2.round(1)

    # Create DataFrame
    data = {
        'ID': range(1, n+1),
        'Group_Treatment': group, 
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Comorbidity': comorbidity,
        # Logistic Outcome [NEW]
        'Outcome_Cured': outcome_cured,
        # Survival
        'Time_Months': time_obs,
        'Status_Death': event_death,
        # Diagnostic
        'Gold_Standard': gold_std,
        'Rapid_Test_Score': rapid_test_val, 
        'Diagnosis_Dr_A': dr_a,
        'Diagnosis_Dr_B': dr_b,
        # Correlation
        'Lab_Albumin': lab_alb,
        'Lab_Calcium': lab_ca,
        # ICC
        'ICC_Rater1': icc_rater1,
        'ICC_Rater2': icc_rater2,
        # Time Cox
        'T_Start': np.zeros(n, dtype=float),
        'T_Stop': time_obs.astype(float)
    }
    
    st.session_state.df = pd.DataFrame(data)
    
    # Set Metadata
    st.session_state.var_meta = {
        'Group_Treatment': {'type':'Categorical', 'map':{0:'Standard Care', 1:'New Drug'}},
        'Sex': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}},
        'Comorbidity': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}},
        'Outcome_Cured': {'type':'Categorical', 'map':{0:'Not Cured', 1:'Cured'}}, # Added Metadata
        'Status_Death': {'type':'Categorical', 'map':{0:'Censored', 1:'Dead'}},
        'Gold_Standard': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}},
        'Diagnosis_Dr_A': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}},
        'Diagnosis_Dr_B': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}},
        # 🟢 Initialize continuous variables explicitly too, matching the file upload logic
        'Age': {'type': 'Continuous', 'label': 'Age', 'map': {}},
        'BMI': {'type': 'Continuous', 'label': 'BMI', 'map': {}},
        'Time_Months': {'type': 'Continuous', 'label': 'Time (Months)', 'map': {}},
        'Rapid_Test_Score': {'type': 'Continuous', 'label': 'Rapid Test Score', 'map': {}},
        'Lab_Albumin': {'type': 'Continuous', 'label': 'Albumin (g/dL)', 'map': {}},
        'Lab_Calcium': {'type': 'Continuous', 'label': 'Calcium (mg/dL)', 'map': {}},
        'ICC_Rater1': {'type': 'Continuous', 'label': 'ICC Rater 1', 'map': {}},
        'ICC_Rater2': {'type': 'Continuous', 'label': 'ICC Rater 2', 'map': {}},
        'T_Start': {'type': 'Continuous', 'label': 'Time Start', 'map': {}},
        'T_Stop': {'type': 'Continuous', 'label': 'Time Stop', 'map': {}},
    }
    st.session_state.uploaded_file_name = "Example Data" # Mark as loaded example data
    
    st.sidebar.success(f"Loaded {n} Example Patients! (Includes Logistic Outcome)")
    st.rerun()
    
# File Uploader
upl = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if upl:
    try:
        data_bytes = upl.getvalue()
        
        # 🟢 แก้ไข: ใช้ hashlib.md5 เพื่อให้ได้ค่า hash ที่คงที่ (Deterministic)
        # เดิม: file_sig = (upl.name, hash(data_bytes))
        file_sig = (upl.name, hashlib.md5(data_bytes).hexdigest())
        
        if st.session_state.get('uploaded_file_sig') != file_sig:
            if upl.name.endswith('.csv'):
                new_df = pd.read_csv(io.BytesIO(data_bytes))
            else:
                new_df = pd.read_excel(io.BytesIO(data_bytes))
            
            st.session_state.df = new_df
            st.session_state.uploaded_file_name = upl.name
            st.session_state.uploaded_file_sig = file_sig
            st.session_state.var_meta = {} # Reset meta for new file
            
            # 🟢 REQUIRED FIX: Add default metadata for new continuous/categorical variables
            current_meta = {}
            for col in new_df.columns:
                # Determine type automatically
                if pd.api.types.is_numeric_dtype(new_df[col]):
                    current_meta[col] = {'type': 'Continuous', 'label': col, 'map': {}}
                else:
                    current_meta[col] = {'type': 'Categorical', 'label': col, 'map': {}}

            st.session_state.var_meta = current_meta
            st.sidebar.success("File Uploaded and Metadata Initialized!")
            st.rerun() # Rerun to update the main page and sidebar controls
        
        else:
            st.sidebar.info("File already loaded.")
            
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, ImportError) as e:  
        st.sidebar.error(f"Error: {e}")
        st.session_state.df = None
        st.session_state.uploaded_file_name = None
        st.session_state.uploaded_file_sig = None

if st.sidebar.button("⚠️ Reset All Data", type="primary"):
    st.session_state.clear()
    st.rerun()

# Variable Settings (Metadata)
if st.session_state.df is not None:
    st.sidebar.header("2. Settings")
    cols = st.session_state.df.columns.tolist()
    
    # 🟢 Use a default value of 'Auto-detect' if the key doesn't exist, which is safer
    auto_detect_meta = {c: st.session_state.var_meta.get(c, {'type': 'Auto-detect', 'map': {}}).get('type', 'Auto-detect') for c in cols}
    
    s_var = st.sidebar.selectbox("Edit Var:", ["Select..."] + cols)
    if s_var != "Select...":
        # Ensure metadata for s_var exists before accessing
        if s_var not in st.session_state.var_meta:
            # Fallback to auto-detect if metadata is missing (shouldn't happen with fix, but safer)
            is_numeric = pd.api.types.is_numeric_dtype(st.session_state.df[s_var]) if s_var in st.session_state.df.columns else False
            initial_type = 'Continuous' if is_numeric else 'Categorical'
            st.session_state.var_meta[s_var] = {'type': initial_type, 'label': s_var, 'map': {}}

        meta = st.session_state.var_meta.get(s_var, {})
        
        # Determine current type for radio button display
        current_type = meta.get('type', 'Auto-detect')
        if current_type == 'Auto-detect':
            is_numeric = pd.api.types.is_numeric_dtype(st.session_state.df[s_var]) if s_var in st.session_state.df.columns else False
            current_type = 'Continuous' if is_numeric else 'Categorical'

        n_type = st.sidebar.radio("Type:", ['Categorical', 'Continuous'], 
                                  index=['Categorical', 'Continuous'].index(current_type))
                                  
        st.sidebar.markdown("Labels (0=No):")
        map_txt = st.sidebar.text_area("Map", value="\n".join([f"{k}={v}" for k,v in meta.get('map',{}).items()]), height=80)
        
        if st.sidebar.button("💾 Save"):
            new_map = {}
            for line in map_txt.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    try:
                        k = k.strip()
                        # Try numeric parse (supports negatives, floats, sci-notation)
                        try:
                            k_num = float(k)
                            k = int(k_num) if k_num.is_integer() else k_num
                        except ValueError:
                            pass
                        new_map[k] = v.strip()
                    except (TypeError, ValueError) as e:
                        st.sidebar.warning(f"Skipping invalid map line '{line}': {e}")
            
            # Ensure the key exists
            if s_var not in st.session_state.var_meta: 
                st.session_state.var_meta[s_var] = {}
            
            # Update meta
            st.session_state.var_meta[s_var]['type'] = n_type
            st.session_state.var_meta[s_var]['map'] = new_map
            st.session_state.var_meta[s_var].setdefault('label', s_var)  # don't clobber existing labels
            
            st.sidebar.success("Saved!")
            st.rerun()

# ==========================================
# 2. MAIN AREA
# ==========================================
if st.session_state.df is not None:
    df = st.session_state.df 

    # 🟢 FIX 2: เพิ่ม Tab t7 สำหรับ Advanced Survival Analysis (Time Cox Regs)
    t0, t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📄 Raw Data", 
        "📋 Baseline Table 1", 
        "🔬 Diagnostic Test", 
        "🔗 Correlation",
        "📊 Logistic Regression",
        "⏳ Survival Analysis",
        "⚖️ Propensity Score",
       # "📈 Time Cox Regs" # 🟢 New Tab
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
 #   with t7:
 #       tab_adv_survival.render(df_clean, st.session_state.var_meta)
        
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
