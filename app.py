import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit.components.v1 as components # 🟢 1. ต้อง Import ตัวนี้เพิ่ม

# Import หน้า Tab ที่แยกไว้ (เพิ่ม tab_survival)
from tabs import tab_data, tab_table1, tab_diag, tab_corr, tab_logit, tab_survival

# --- CONFIGURATION ---
st.set_page_config(page_title="Medical Stat Tool", layout="wide",menu_items={
        'Get Help': 'https://ntwkkm.github.io/infos/stat_manual.html',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
    })
st.title("🏥 Medical Statistical Tool")

# 🟢 FIX: วาง components.html ไว้ที่นี่ (หลัง set_page_config และ st.title)
components.html("""
<script>
    // window.parent คือการสั่งให้ทะลุ Iframe ของ Component ออกไปที่หน้าหลัก (index.html)
    var loader = window.parent.document.getElementById('loading-screen');
    if (loader) {
        loader.style.opacity = '0'; // สั่งให้จางลง
        setTimeout(function() {
            loader.style.display = 'none'; // แล้วซ่อนถาวร
        }, 500);
    }
</script>
""", height=0) # height=0 เพื่อไม่ให้กินพื้นที่หน้าจอ

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
    
    # 🟢 8. Agreement Variables (Cohen's Kappa) -- เพิ่มส่วนนี้ --
    # สร้างผลวินิจฉัยของหมอ A (สุ่มมาเลย)
    diag_dr_a = np.random.binomial(1, 0.3, n) # ความชุกโรค 30%
    
    # สร้างผลหมอ B โดย Copy จาก A มาก่อน (เพื่อให้ Agreement สูง)
    diag_dr_b = diag_dr_a.copy()
    
    # สร้างความไม่ตรงกัน (Disagreement) ประมาณ 10-15% ของเคสทั้งหมด
    # สุ่มเลือก Index มาจำนวนหนึ่ง แล้วสลับค่า (0->1, 1->0)
    num_mismatch = int(0.12 * n) # 12% Error rate
    mismatch_idx = np.random.choice(n, num_mismatch, replace=False)
    diag_dr_b[mismatch_idx] = 1 - diag_dr_b[mismatch_idx] # Flip ค่า

    # 🟢 9. Reliability Variables (ICC) -- เพิ่มส่วนนี้ --
    # จำลองการวัด SBP (Systolic Blood Pressure) จาก 2 เครื่อง
    sbp_machine_1 = np.random.normal(120, 15, n).round(0)
    
    # Machine 2: มีความคลาดเคลื่อนเล็กน้อย (Noise) + Systematic Bias (เช่น วัดได้สูงกว่าจริง 2 mmHg)
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
        # 🟢 ใส่ข้อมูล Agreement ลง DataFrame
        'Diagnosis_Dr_A': diag_dr_a,
        'Diagnosis_Dr_B': diag_dr_b
        # 🟢 เพิ่มตัวแปร ICC ลง DataFrame
        'SBP_Machine_1': sbp_machine_1,
        'SBP_Machine_2': sbp_machine_2
    }
    
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {
        'Sex': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}},
        'Hypertension': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}},
        'Outcome_Disease': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}},
        'Event_Death': {'type':'Categorical', 'map':{0:'Censored', 1:'Event (Death)'}},
        # 🟢 เพิ่ม Meta Map ให้แสดงผลสวยๆ
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
        
# 🟢 แก้ไข: ลบพารามิเตอร์ icon ออก และใส่ Emoji ในชื่อปุ่มแทน
# คุณสามารถเลือก Emoji ได้ตามชอบ เช่น "🗑️", "⚠️", "❌", "♻️"
if st.sidebar.button("⚠️ Reset All Data", type="primary"):
    # ล้างค่าทุกอย่างใน Session State
    st.session_state.clear()
    st.rerun()
    # หมายเหตุ: ถ้าใช้ Streamlit เวอร์ชันใหม่ แนะนำให้เปลี่ยนเป็น st.rerun()

# Example Data Generator (โค้ดเดิมต่อจากนี้)

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
    df = st.session_state.df  # นี่คือ Raw Data (มี 'abc' ปนอยู่)

    t0, t1, t2, t3, t4, t5 = st.tabs([
        "📄 Raw Data", 
        "📋 Baseline Table 1", 
        "🔬 Diagnostic Test", 
        "🔗 Correlation",
        "📊 Logistic Regression",
        "⏳ Survival Analysis"
    ])

    # Call Modules
    with t0:
        # 🟢 Tab 0: จัดการ Raw Data -> ส่งกลับมาอัปเดต session_state (ยังเป็น Raw)
        st.session_state.df = tab_data.render(df) 
        
        # 🟢 Generate Clean Data for Analysis: สร้างข้อมูลสะอาดเตรียมไว้ให้ Tab อื่น
        # ดึง custom_na_list ที่ตั้งค่าไว้มาใช้ด้วย
        custom_na = st.session_state.get('custom_na_list', [])
        df_clean = tab_data.get_clean_data(st.session_state.df, custom_na)

    # 🟢 Tab 1-5: ใช้ df_clean (ที่แปลง Text ผิดๆ เป็น NaN แล้ว) ในการคำนวณ
    # ทำให้กราฟไม่พัง และ Raw Data ไม่หาย
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
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit; font-weight:bold;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
</div>
""", unsafe_allow_html=True)
