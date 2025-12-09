import streamlit as st
import pandas as pd
import numpy as np
from logic import process_data_and_generate_html # Import จาก root

def check_perfect_separation(df, target_col):
    """Helper Function for Logistic"""
    risky_vars = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: return []
    except: return []

    for col in df.columns:
        if col == target_col: continue
        if df[col].nunique() < 10: 
            try:
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except: pass
    return risky_vars

def render(df, var_meta):
    st.subheader("4. Binary Logistic Regression Analysis")
    ##### Logistic Regression Analysis
    st.info("""
    **💡 Guide:** Models the relationship between predictors and the **probability** of a **binary outcome** (e.g., disease/no disease).

    * **Odds Ratio (OR/aOR):** The main result, reported with a 95% CI. Measures the change in the odds of the outcome for every one-unit increase in the predictor.
        * **Adjusted OR (aOR):** This is the output when **multiple features** are used, meaning the effect is **controlled/adjusted** for other variables in the model.
        * **OR/AOR > 1:** Increased odds (Risk factor).
        * **OR/AOR < 1:** Decreased odds (Protective factor).
    * **P-value:** Tests if the predictor's association with the outcome is statistically significant.
    
    **Variable Selection:**
    * **Target (Y):** Must be **Binary** (e.g.,Die/Survide, 0/1, Yes/No).
    * **Features (X):** Can be **Numeric** or **Categorical** (e.g., Age, Gender).
    * **Features (X) Inclusion:** All available features are **automatically included** by default; users can **manually exclude** any unwanted variables.
""")
    
    all_cols = df.columns.tolist()
    c1, c2 = st.columns([1, 2])
    
    with c1:
        def_idx = 0
        for i, c in enumerate(all_cols):
            if 'outcome' in c.lower() or 'died' in c.lower(): def_idx = i; break
        target = st.selectbox("Select Outcome (Y):", all_cols, index=def_idx, key='logit_target')
        
    with c2:
        risky_vars = check_perfect_separation(df, target)
        exclude_cols = []
        if risky_vars:
            st.warning(f"⚠️ Risk of Perfect Separation: {', '.join(risky_vars)}")
            exclude_cols = st.multiselect("Exclude Variables:", all_cols, default=risky_vars, key='logit_exclude')
        else:
            exclude_cols = st.multiselect("Exclude Variables (Optional):", all_cols, key='logit_exclude_opt')

    # 🟢 NEW: เพิ่มส่วนเลือก Method (User Selection)
    method_choice = st.radio(
        "Regression Method:",
        ["Auto (Recommended)", "Standard (MLE)", "Firth's (Penalized)"],
        index=0,
        horizontal=True,
        help="Standard: Usual Logistic Regression. Firth: Reduces bias and handles separation (Recommended for small sample size/rare events)."
    )
    
    # แปลงตัวเลือกเป็นรหัสที่ logic.py เข้าใจ
    if "Firth" in method_choice:
        algo = 'firth'
    elif "Standard" in method_choice:
        algo = 'bfgs'
    else:
        algo = 'auto'

    st.write("") # Spacer

    run_col, dl_col = st.columns([1, 1])
    if 'html_output_logit' not in st.session_state: st.session_state.html_output_logit = None

    if run_col.button("🚀 Run Logistic Regression", type="primary"):
        if df[target].nunique() < 2:
            st.error("Error: Outcome must have at least 2 values.")
        else:
            with st.spinner("Calculating..."):
                try:
                    final_df = df.drop(columns=exclude_cols, errors='ignore')
                    # 🟢 ส่งค่า algo (method) ไปให้ function
                    html = process_data_and_generate_html(final_df, target, var_meta=var_meta, method=algo)
                    st.session_state.html_output_logit = html 
                    st.components.v1.html(html, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Failed: {e}")
                    
    with dl_col:
        if st.session_state.html_output_logit:
            st.download_button("📥 Download Report", st.session_state.html_output_logit, "logit.html", "text/html", key='dl_logit')
        else: st.button("📥 Download Report", disabled=True, key='ph_logit')
