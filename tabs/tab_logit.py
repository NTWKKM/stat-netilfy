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
    st.subheader("4.Binary Logistic Regression Analysis")
    ##### Logistic Regression Analysis
    st.info("""
    **💡 Guide:** Used to model the relationship between one or more predictor variables (features) and a **binary outcome** (dependent variable). This model estimates the **probability** of an event occurring (e.g., probability of disease).

    * **Model Output:** The primary result is the **Odds Ratio (OR)**, typically presented with a 95% Confidence Interval (CI).
    * **Odds Ratio (OR):** Represents the change in the **odds** of the outcome for every one-unit increase in the predictor variable.
        * **OR > 1:** The predictor is associated with increased odds of the outcome.
        * **OR < 1:** The predictor is associated with decreased odds of the outcome.
    * **P-value:** Determines if the predictor variable has a statistically significant association with the outcome.

    **Variable Selection:**
    * **Target Variable (Y):** Must be **Binary** (e.g., 0/1, Yes/No, Die/Survive).
    * **Feature Variables (X):** Can be **Numeric** (e.g., Age) or **Categorical** (e.g., Gender).
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

    run_col, dl_col = st.columns([1, 1])
    if 'html_output_logit' not in st.session_state: st.session_state.html_output_logit = None

    if run_col.button("🚀 Run Logistic Regression", type="primary"):
        if df[target].nunique() < 2:
            st.error("Error: Outcome must have at least 2 values.")
        else:
            with st.spinner("Calculating..."):
                try:
                    final_df = df.drop(columns=exclude_cols, errors='ignore')
                    html = process_data_and_generate_html(final_df, target, var_meta=var_meta)
                    st.session_state.html_output_logit = html 
                    st.components.v1.html(html, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Failed: {e}")
                    
    with dl_col:
        if st.session_state.html_output_logit:
            st.download_button("📥 Download Report", st.session_state.html_output_logit, "logit.html", "text/html", key='dl_logit')
        else: st.button("📥 Download Report", disabled=True, key='ph_logit')
