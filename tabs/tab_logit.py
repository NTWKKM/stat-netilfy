import streamlit as st
import pandas as pd
import numpy as np
from logic import process_data_and_generate_html # Import จาก root

def check_perfect_separation(df, target_col):
    """
    Identify predictor columns that may cause perfect separation with the specified target.
    
    Checks predictors (excluding the target) that have fewer than 10 unique values and flags any whose contingency table with the target contains a zero cell, which may indicate perfect separation in a logistic model. If the target cannot be interpreted as numeric with at least two unique values, an empty list is returned; errors in per-predictor contingency tables are ignored.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing predictors and the target column.
        target_col (str): Name of the target column to evaluate against.
    
    Returns:
        list: Names of columns that may cause perfect separation; empty list if none are found or on error.
    """
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
    """
    Render the "4. Binary Logistic Regression Analysis" section in a Streamlit app.
    
    Renders UI controls to select a binary outcome, optionally exclude predictors, choose a regression method (Auto, Standard, Firth), and run a logistic regression. Validates the selected outcome has at least two unique values, launches the analysis, displays the resulting HTML report, and stores the generated report in `st.session_state['html_output_logit']`. If predictors with potential perfect separation are detected, they are offered as default exclusions.
    
    Parameters:
        df (pandas.DataFrame): Source dataset containing the outcome and predictor columns.
        var_meta (dict | Any): Variable metadata passed through to the report generation routine (used to annotate or format outputs).
    """
    st.subheader("4. Binary Logistic Regression Analysis")
    st.info("""
    **💡 Guide:** Models the relationship between predictors and the **probability** of a **binary outcome** (e.g., disease/no disease).

    * **Odds Ratio (OR/aOR):** The main result, reported with a 95% CI. Measures the change in the odds of the outcome for every one-unit increase in the predictor.
        * **Adjusted OR (aOR):** This is the output when **multiple features** are used, meaning the effect is **controlled/adjusted** for other variables in the model.
        * **OR/AOR > 1:** Increased odds (Risk factor).
        * **OR/AOR < 1:** Decreased odds (Protective factor).
    * **P-value:** Tests if the predictor's association with the outcome is statistically significant.
    
    **Variable Selection:**
    * **Target (Y):** Must be **Binary** (e.g., Die/Survive, 0/1, Yes/No).
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

    # 🟢 NEW: เพิ่มส่วนเลือ Method (User Selection)
    method_options = {
        "Auto (Recommended)": "auto",
        "Standard (MLE)": "bfgs",
        "Firth's (Penalized)": "firth",
    }
    method_choice = st.radio(
        "Regression Method:",
        list(method_options.keys()),
        index=0,
        horizontal=True,
        # 🟢 ใช้ """ (Triple Quotes) เพื่อเขียนหลายบรรทัช
        help="""
        - **Auto:** Automatically selects the most suitable method based on data characteristics and availability.
        - **Standard:** Usual Logistic Regression.
        - **Firth:** Reduces bias and handles separation (Recommended for small sample size/rare events).
        """
    )
    # แปลงตัวเลือกเป็นรหัสที่ logic.py เข้าใจ
    algo = method_options[method_choice]

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
        else:
            st.button("📥 Download Report", disabled=True, key='ph_logit')
    
    # --- NEW: Reference & Interpretation ---
    st.markdown("---")
    with st.expander("📚 Reference & Interpretation"):
        st.markdown("""
        ### Binary Logistic Regression Guide
        
        **When to Use:**
        - Predicting binary outcomes (Disease/No Disease)
        - Understanding risk/protective factors
        - Adjusted analysis (controlling for confounders)
        - Classification models
        
        **Key Metrics:**
        
        #### Odds Ratio (OR)
        - **OR = 1**: No effect
        - **OR > 1**: Increased odds (Risk Factor) 🔴
        - **OR < 1**: Decreased odds (Protective Factor) 🟢
        - **Example**: OR = 2.5 → 2.5× increased odds
        
        #### Adjusted OR (aOR)
        - Accounts for other variables in model
        - More reliable than unadjusted OR ✅
        - Preferred for reporting ✅
        
        #### Confidence Interval (95% CI)
        - **CI crosses 1.0**: Not statistically significant ⚠️
        - **CI doesn't cross 1.0**: Statistically significant ✅
        - Narrower CI = more precise estimate
        
        #### P-value
        - **p < 0.05**: Statistically significant ✅
        - **p ≥ 0.05**: Not significant ⚠️
        
        ---
        
        ### Regression Methods
        
        | Method | When to Use | Notes |
        |--------|------------|-------|
        | **Standard (MLE)** | Default, balanced data | Classic logistic regression |
        | **Firth's** | Small sample, rare events, separation | Reduces bias, more stable |
        | **Auto** | Recommended | Picks best method automatically |
        
        ---
        
        ### Common Mistakes ❌
        
        - **Reporting unadjusted OR** without adjustment → Use aOR ✅
        - **Perfect separation** (entire category = outcome) → Exclude problematic variables
        - **Not checking CI** (only looking at p-value) → CI provides range of plausible values
        - **Multicollinearity** (predictors highly correlated) → Check correlations first
        - **Overfitting** (too many variables) → Use variable selection
        - **Interpreting OR for log-transformed predictors** → Multiply by e^(log-unit) change
        
        ---
        
        ### Interpretation Example
        
        **Model Output:**
        - Variable: Smoking
        - aOR = 1.8 (95% CI: 1.2 - 2.4)
        - p = 0.003
        
        **Interpretation:** 
        Smoking is associated with 1.8× increased odds of outcome (compared to non-smoking), adjusting for other variables. This difference is statistically significant (p < 0.05), and we're 95% confident the true OR is between 1.2 and 2.4. ✅
        """)
