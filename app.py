import streamlit as st
import pandas as pd
import numpy as np
# Import logic เดิม
from logic import process_data_and_generate_html
# 🟢 Import logic ใหม่
import diag_test 

st.set_page_config(page_title="Medical Stat Tool", layout="wide")

st.title("🏥 Medical Statistical Tool")

# --- GLOBAL DATA STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {} 

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Data & Logistic Regression", "2. Diagnostic Test (ROC / Chi2)"])

st.sidebar.markdown("---")
st.sidebar.header("Data Management")

# --- 1. DATA INPUT (Shared) ---
if st.sidebar.button("📄 Load Example Data"):
    # ... (ใช้ Data Example เดิมของคุณได้เลย) ...
    # เพื่อความกระชับ ผมขอละไว้ ให้ copy logic เดิมมาใส่
    # หรือถ้าขี้เกียจแก้ ใส่ code นี้ลงไปแทน data เดิมได้ครับ:
    data = {
        'age': [25, 28, 30, 35, 40, 42, 45, 22, 29, 33, 50, 55, 52, 58, 60, 62, 51, 59, 54, 57, 65, 70, 72, 75, 80, 82, 85, 78, 88, 90] * 2,
        'sex': [0, 1] * 30,
        'score_test': [1.2, 1.5, 1.1, 2.0, 2.5, 3.1, 4.0, 1.0, 1.8, 2.2, 5.0, 5.5, 4.8, 6.0, 6.5, 5.2, 4.9, 6.1, 5.3, 5.8, 8.0, 8.5, 9.0, 7.5, 9.2, 9.5, 9.8, 8.8, 9.9, 9.1] * 2,
        'outcome_disease': [0,0,0,0,0, 0,0,0,0,0, 1,1,0,1,0, 0,1,0,1,0, 1,1,1,1,1, 1,1,1,1,1] * 2
    }
    st.session_state.df = pd.DataFrame(data)
    st.sidebar.success("Loaded!")

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded_file)
        else: st.session_state.df = pd.read_excel(uploaded_file)
    except: st.sidebar.error("Load failed")

# 🟢 MAIN PAGE LOGIC
if st.session_state.df is not None:
    df = st.session_state.df
    
    # -----------------------------------------------
    # PAGE 1: LOGISTIC REGRESSION (ของเดิม)
    # -----------------------------------------------
    if page == "1. Data & Logistic Regression":
        st.subheader("Data Review")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=300)
        
        st.subheader("Logistic Regression Analysis")
        # ... (ใส่ Logic เลือก Outcome และปุ่ม Run Analysis เดิมของคุณตรงนี้) ...
        # เพื่อให้สั้นลง ผมเรียกฟังก์ชันเลย
        all_cols = edited_df.columns.tolist()
        target = st.selectbox("Select Outcome (Y)", all_cols)
        
        if st.button("🚀 Run Logistic Regression"):
             html = process_data_and_generate_html(edited_df, target, var_meta=st.session_state.var_meta)
             st.components.v1.html(html, height=600, scrolling=True)

    # -----------------------------------------------
    # PAGE 2: DIAGNOSTIC TEST (ของใหม่)
    # -----------------------------------------------
    elif page == "2. Diagnostic Test (ROC / Chi2)":
        st.header("🔬 Diagnostic Test & Statistics")
        
        # TAB แยกย่อย
        tab1, tab2, tab3 = st.tabs(["📊 Descriptive", "🎲 Chi-Square", "📈 ROC Curve & AUC"])
        
        # --- TAB 1: Descriptive ---
        with tab1:
            st.subheader("Descriptive Statistics")
            col_desc = st.selectbox("Select Variable:", df.columns)
            if col_desc:
                res_df = diag_test.calculate_descriptive(df, col_desc)
                st.table(res_df)
                
        # --- TAB 2: Chi-Square ---
        with tab2:
            st.subheader("Chi-Square Test (Categorical vs Categorical)")
            c1, c2 = st.columns(2)
            var1 = c1.selectbox("Variable 1:", df.columns, key='chi1')
            var2 = c2.selectbox("Variable 2:", df.columns, key='chi2')
            
            if st.button("Run Chi-Square"):
                tab_res, msg = diag_test.calculate_chi2(df, var1, var2)
                st.write(msg)
                if tab_res is not None:
                    st.write("Contingency Table:")
                    st.dataframe(tab_res)

        # --- TAB 3: ROC Curve ---
        with tab3:
            st.subheader("ROC Analysis (DeLong / Binomial CI)")
            
            rc1, rc2 = st.columns(2)
            truth_var = rc1.selectbox("Gold Standard (Binary Outcome):", df.columns, key='roc_truth')
            score_var = rc2.selectbox("Test Variable (Score/Continuous):", df.columns, key='roc_score')
            
            method = st.radio("CI Method for AUC:", 
                              ["DeLong et al.", "Binomial exact (Hanley & McNeil)"], 
                              horizontal=True)
            
            method_code = 'delong' if "DeLong" in method else 'hanley'
            
            if st.button("📉 Plot ROC & Calculate AUC"):
                stats_res, err, fig = diag_test.analyze_roc(df, truth_var, score_var, method_code)
                
                if err:
                    st.error(err)
                else:
                    # Show Stats
                    st.success(f"AUC = {stats_res['AUC']:.4f} ({stats_res['95% CI Lower']:.4f} - {stats_res['95% CI Upper']:.4f})")
                    
                    # แบ่งหน้าจอแสดงผล
                    resc1, resc2 = st.columns([1, 1.5])
                    
                    with resc1:
                        st.markdown("### Statistics")
                        st.json(stats_res)
                        
                    with resc2:
                        st.markdown("### ROC Graph")
                        st.pyplot(fig)

else:
    st.info("👈 Please upload data first.")
