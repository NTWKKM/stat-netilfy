import streamlit as st
import pandas as pd
import numpy as np
# Import logic ทั้งหมด
from logic import process_data_and_generate_html
import diag_test 
import table_one # 🟢 Import ไฟล์ใหม่

st.set_page_config(page_title="Medical Stat Tool", layout="wide")

st.title("🏥 Medical Statistical Tool")

# --- GLOBAL DATA STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {} 

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
# 🟢 เพิ่มเมนูที่ 3
page = st.sidebar.radio("Go to:", [
    "1. Data & Logistic Regression", 
    "2. Diagnostic Test (ROC / Chi2)",
    "3. Baseline Characteristics (Table 1)"
])

st.sidebar.markdown("---")
st.sidebar.header("Data Management")

# --- DATA INPUT (Shared) ---
if st.sidebar.button("📄 Load Example Data"):
    # (ใช้ Data เดิมได้เลยครับ ละไว้เพื่อความกระชับ)
    data = {
        'age': [25, 28, 30, 35, 40, 42, 45, 22, 29, 33, 50, 55, 52, 58, 60, 62, 51, 59, 54, 57, 65, 70, 72, 75, 80, 82, 85, 78, 88, 90] * 2,
        'sex': [0, 1] * 30,
        'score_test': [1.2, 1.5, 1.1, 2.0, 2.5, 3.1, 4.0, 1.0, 1.8, 2.2, 5.0, 5.5, 4.8, 6.0, 6.5, 5.2, 4.9, 6.1, 5.3, 5.8, 8.0, 8.5, 9.0, 7.5, 9.2, 9.5, 9.8, 8.8, 9.9, 9.1] * 2,
        'outcome_disease': [0,0,0,0,0, 0,0,0,0,0, 1,1,0,1,0, 0,1,0,1,0, 1,1,1,1,1, 1,1,1,1,1] * 2,
        'group_treatment': [0,0,1,1,0, 0,1,1,0,1] * 6 # เพิ่มตัวแปรกลุ่ม
    }
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {
        'sex': {'type': 'Categorical', 'map': {0:'Female', 1:'Male'}},
        'outcome_disease': {'type': 'Categorical', 'map': {0:'Healthy', 1:'Disease'}},
        'group_treatment': {'type': 'Categorical', 'map': {0:'Placebo', 1:'Drug A'}}
    }
    st.sidebar.success("Loaded!")

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded_file)
        else: st.session_state.df = pd.read_excel(uploaded_file)
    except: st.sidebar.error("Load failed")

# 🟢 MAIN LOGIC
if st.session_state.df is not None:
    df = st.session_state.df
    all_cols = df.columns.tolist()

    # -----------------------------------------------
    # PAGE 1: LOGISTIC REGRESSION
    # -----------------------------------------------
    if page == "1. Data & Logistic Regression":
        st.subheader("Data Review")
        st.data_editor(df, num_rows="dynamic", use_container_width=True, height=300)
        
        st.subheader("Logistic Regression Analysis")
        target = st.selectbox("Select Outcome (Y)", all_cols)
        if st.button("🚀 Run Logistic Regression"):
             html = process_data_and_generate_html(df, target, var_meta=st.session_state.var_meta)
             st.components.v1.html(html, height=600, scrolling=True)

    # -----------------------------------------------
    # PAGE 2: DIAGNOSTIC TEST
    # -----------------------------------------------
    elif page == "2. Diagnostic Test (ROC / Chi2)":
        st.header("🔬 Diagnostic Test & Statistics")
        tab1, tab2, tab3 = st.tabs(["📊 Descriptive", "🎲 Chi-Square", "📈 ROC Curve"])
        
        with tab1:
            col_desc = st.selectbox("Select Variable:", all_cols)
            if col_desc: st.table(diag_test.calculate_descriptive(df, col_desc))
                
        with tab2:
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Var 1:", all_cols, key='chi1')
            v2 = c2.selectbox("Var 2:", all_cols, key='chi2')
            if st.button("Run Chi-Square"):
                tab_res, msg = diag_test.calculate_chi2(df, v1, v2)
                st.write(msg)
                if tab_res is not None: st.dataframe(tab_res)

        with tab3:
            rc1, rc2 = st.columns(2)
            truth = rc1.selectbox("Gold Standard (0/1):", all_cols, key='roc_truth')
            score = rc2.selectbox("Test Score:", all_cols, key='roc_score')
            method = st.radio("CI Method:", ["DeLong", "Binomial (Hanley)"])
            
            if st.button("📉 Plot ROC"):
                res, err, fig = diag_test.analyze_roc(df, truth, score, 'delong' if 'DeLong' in method else 'hanley')
                if err: st.error(err)
                else:
                    st.success(f"AUC = {res['AUC']:.4f}")
                    st.pyplot(fig)
                    st.json(res)

    # -----------------------------------------------
    # PAGE 3: TABLE 1 (BASELINE CHARACTERISTICS) - NEW!
    # -----------------------------------------------
    elif page == "3. Baseline Characteristics (Table 1)":
        st.header("📋 Baseline Characteristics (Table 1)")
        
        st.markdown("""
        สร้างตารางข้อมูลพื้นฐาน (Demographic Table) เปรียบเทียบระหว่างกลุ่มอัตโนมัติ
        * **Continuous:** แสดง Mean ± SD (ใช้ T-test/ANOVA)
        * **Categorical:** แสดง n (%) (ใช้ Chi-square)
        """)
        
        col_group = st.selectbox("Select Group Column (Optional):", ["None"] + all_cols, index=0)
        
        # เลือกตัวแปรที่จะใส่ในตาราง (Default คือทั้งหมด ยกเว้นตัว Group เอง)
        default_vars = [c for c in all_cols if c != col_group]
        selected_vars = st.multiselect("Select Variables to Include:", all_cols, default=default_vars)
        
        if st.button("📊 Generate Table 1"):
            with st.spinner("Generating table..."):
                try:
                    group_val = None if col_group == "None" else col_group
                    html_table = table_one.generate_table(df, selected_vars, group_val, st.session_state.var_meta)
                    
                    st.components.v1.html(html_table, height=800, scrolling=True)
                    st.download_button("📥 Download HTML Table", html_table, "table1.html", "text/html")
                except Exception as e:
                    st.error(f"Error generating table: {e}")

else:
    st.info("👈 Please upload data first.")
