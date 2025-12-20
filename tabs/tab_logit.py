import streamlit as st
import pandas as pd
import numpy as np
from logic import process_data_and_generate_html # Import จาก root
from logger import get_logger
logger = get_logger(__name__)

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

# 🟢 NEW: Helper function to select dataset
def _get_dataset_for_analysis(df: pd.DataFrame):
    """
    Helper function to select between original and matched datasets.
    Returns tuple: (selected_df, data_source_label)
    """
    # Check if matched data is available
    has_matched = st.session_state.get('is_matched', False) and st.session_state.get('df_matched') is not None
    
    if has_matched:
        col1, _ = st.columns([2, 1])
        with col1:
            data_source = st.radio(
                "📄 Select Dataset:",
                ["📊 Original Data", "✅ Matched Data (from PSM)"],
                index=1,  # Default to matched data if available
                horizontal=True,
                key=f"data_source_{id(st.session_state)}"
            )
        
        if "✅" in data_source:
            selected_df = st.session_state.df_matched.copy()
            label = f"✅ Matched Data ({len(selected_df)} rows)"
        else:
            selected_df = df
            label = f"📊 Original Data ({len(df)} rows)"
    else:
        selected_df = df
        label = f"📊 Original Data ({len(df)} rows)"
    
    return selected_df, label

def render(df, var_meta):
    """
    Render the "4. Logistic Regression Analysis" section in a Streamlit app.
    
    Renders UI controls to select a binary outcome, optionally exclude predictors, choose a regression method (Auto, Standard, Firth), and run a logistic regression. Validates the selected outcome has at least two unique values, launches the analysis, displays the resulting HTML report, and stores the generated report in `st.session_state['html_output_logit']`. If predictors with potential perfect separation are detected, they are offered as default exclusions.
    
    Parameters:
        df (pandas.DataFrame): Source dataset containing the outcome and predictor columns.
        var_meta (dict | Any): Variable metadata passed through to the report generation routine (used to annotate or format outputs).
    """
    st.subheader("📏 Logistic Regression Analysis")
    
    # 🟢 NEW: Add matched data note if available
    if st.session_state.get('is_matched', False):
        st.info("✅ **Matched Dataset Available** - You can select it below for analysis")
    
    # Create subtabs (prepared for future: Binary, Multinomial, Ordinal, etc.)
    sub_tab1, sub_tab2 = st.tabs([
        "📈 Binary Logistic Regression",
        "ℹ️ Reference & Interpretation"
    ])
    
    # ==================================================
    # SUB-TAB 1: Binary Logistic Regression
    # ==================================================
    with sub_tab1:
        st.markdown("### Binary Logistic Regression")
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
        
        # 🟢 NEW: Dataset selection
        selected_df, data_label = _get_dataset_for_analysis()
        if selected_df is None:
            selected_df = df
        
        st.write(f"**Using:** {data_label}")
        st.write(f"**Rows:** {len(selected_df)} | **Columns:** {len(selected_df.columns)}")
        
        all_cols = selected_df.columns.tolist()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            def_idx = 0
            for i, c in enumerate(all_cols):
                if 'outcome' in c.lower() or 'died' in c.lower():
                    def_idx = i
                    break
            target = st.selectbox("Select Outcome (Y):", all_cols, index=def_idx, key='logit_target')
            
        with c2:
            risky_vars = check_perfect_separation(selected_df, target)
            exclude_cols = []
            if risky_vars:
                st.warning(f"⚠️ Risk of Perfect Separation: {', '.join(risky_vars)}")
                exclude_cols = st.multiselect("Exclude Variables:", all_cols, default=risky_vars, key='logit_exclude')
            else:
                exclude_cols = st.multiselect("Exclude Variables (Optional):", all_cols, key='logit_exclude_opt')

        # 🟢 Method Selection
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
            help="""
        - **Auto:** Automatically selects the most suitable method based on data characteristics and availability.
        - **Standard:** Usual Logistic Regression.
        - **Firth:** Reduces bias and handles separation (Recommended for small sample size/rare events).
        """
        )
        algo = method_options[method_choice]

        st.write("") # Spacer

        run_col, dl_col = st.columns([1, 1])
        if 'html_output_logit' not in st.session_state:
            st.session_state.html_output_logit = None

        if run_col.button("🚀 Run Logistic Regression", type="primary"):
            if selected_df[target].nunique() < 2:
                st.error("Error: Outcome must have at least 2 values.")
            else:
                with st.spinner("Calculating..."):
                    try:
                        final_df = selected_df.drop(columns=exclude_cols, errors='ignore')
                        
                        # 🟢 NEW: Re-check for perfect separation AFTER exclusion
                        risky_vars_final = check_perfect_separation(final_df, target)
                        
                        # 🟢 NEW: Warn if using Standard method on risky data
                        if risky_vars_final and algo == 'bfgs':
                            st.warning(
                                f"""\u26a0️ **WARNING: Perfect Separation Detected!**

**Variables with zero-cell contingency tables:** {', '.join(risky_vars_final)}

**Selected Method:** Standard (MLE)

**Problems this may cause:**
- \u274c Model may not converge
- \u274c Infinite coefficients (∞)
- \u274c Missing p-values and standard errors
- \u274c Invalid confidence intervals
- \u274c Unreliable results

**\u2705 Recommended Solution:** Use **Firth's (Penalized)** method instead!
- Handles perfect separation automatically
- Produces reliable confidence intervals
- Better for small samples and rare events

**Your Options:**
1. Cancel and select "Firth's (Penalized)" method
2. Cancel and exclude these variables manually
3. Proceed anyway (not recommended)
""",
                                icon="⚠️"
                            )
                            logger.warning("User selected Standard method with perfect separation: %s", risky_vars_final)
                        
                        html = process_data_and_generate_html(final_df, target, var_meta=var_meta, method=algo)
                        st.session_state.html_output_logit = html 
                        st.components.v1.html(html, height=600, scrolling=True)
                        
                        # 🟢 NEW: Log method used and data source
                        data_source_label = "✅ Matched" if selected_df is not None and st.session_state.get('is_matched') else "Original"
                        logger.info("\u2705 Logit analysis completed | method=%s | risky_vars=%d | n=%d | data_source=%s", algo, len(risky_vars_final), len(final_df), data_source_label)
                        
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        logger.exception("Logistic regression failed")
                        
        with dl_col:
            if st.session_state.html_output_logit:
                st.download_button("📥 Download Report", st.session_state.html_output_logit, "logit.html", "text/html", key='dl_logit')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_logit')

    # ==================================================
    # SUB-TAB 2: Reference & Interpretation
    # ==================================================
    with sub_tab2:
        st.markdown("##### 📚 Quick Reference: Logistic Regression")
        
        st.info("""
        **🎯 When to Use Logistic Regression:**
        
        | Type | Outcome | Predictors | Example |
        |------|---------|-----------|----------|
        | **Binary** | 2 categories (Yes/No) | Any | Disease/No Disease |
        | **Multinomial** | 3+ unordered categories | Any | Stage (I/II/III/IV) |
        | **Ordinal** | 3+ ordered categories | Any | Severity (Low/Med/High) |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Binary Logistic Regression")
            st.markdown("""
            **When to Use:**
            - Predicting binary outcomes (Disease/No Disease)
            - Understanding risk/protective factors
            - Adjusted analysis (controlling for confounders)
            - Classification models
            
            **Key Metrics:**
            
            **Odds Ratio (OR)**
            - **OR = 1**: No effect
            - **OR > 1**: Increased odds (Risk Factor) 🔴
            - **OR < 1**: Decreased odds (Protective Factor) 🟢
            - Example: OR = 2.5 → 2.5× increased odds
            
            **Adjusted OR (aOR)**
            - Accounts for other variables in model
            - More reliable than unadjusted ✅
            - Preferred for reporting ✅
            
            **CI & P-value**
            - CI crosses 1.0: Not significant ⚠️
            - CI doesn't cross 1.0: Significant ✅
            - p < 0.05: Significant ✅
            """)
        
        with col2:
            st.markdown("### Regression Methods")
            st.markdown("""
            | Method | When to Use | Notes |
            |--------|-------------|-------|
            | **Standard (MLE)** | Default, balanced data | Classic logistic regression |
            | **Firth's** | Small sample, rare events | Reduces bias, more stable |
            | **Auto** | Recommended | Picks best method |
            
            ---
            
            ### Common Mistakes \u274c
            
            - **Unadjusted OR** without adjustment → Use aOR ✅
            - **Perfect separation** (category = outcome) → Exclude or use Firth
            - **Ignoring CI** (only p-value) → CI shows range
            - **Multicollinearity** (correlated predictors) → Check correlations
            - **Overfitting** (too many variables) → Use variable selection
            - **Log-transformed interpreters** → Multiply by e^(unit change)
            """)
        
        st.markdown("---")
        
        # 🟢 NEW: Perfect Separation & Method Selection Guide
        st.markdown("""
        ### ⚠️ Perfect Separation & Method Selection
        
        **What is Perfect Separation?**
        
        A predictor perfectly predicts the outcome. Example:
        
        | High Risk | Survived | Died |
        |-----------|----------|------|
        | No        | 100      | 0    |
        | Yes       | 0        | 100  |
        
        → Perfect separation! (diagonal pattern)
        
        **Why is it a Problem?**
        
        Standard logistic regression (MLE):
        - \u274c Cannot estimate coefficients reliably
        - \u274c Returns infinite or missing values
        - \u274c Model doesn't converge
        - \u274c P-values are undefined
        - \u274c Results are invalid
        
        **How to Detect:**
        - 🔍 App shows warning: "⚠️ Risk of Perfect Separation: var_name"
        - 📊 Contingency table has a zero cell (entire row/column = 0)
        
        **4 Solutions (Ranked by Recommendation):**
        
        **Option 1: Auto Method** 🟢 (BEST - RECOMMENDED)
        - ✅ Automatically detects perfect separation
        - ✅ Automatically switches to Firth's method
        - ✅ No manual action required
        - ✅ Most reliable
        - ✅ **Just select "Auto (Recommended)" and run!**
        
        **Option 2: Firth's Method** 🟢 (GOOD)
        - ✅ Handles separation via penalized likelihood
        - ✅ Produces reliable coefficients & CI
        - ✅ Reduces coefficient bias
        - ⚠️ Requires manual method selection
        
        **Option 3: Exclude Variable** 🟢 (ACCEPTABLE)
        - ✅ Removes problematic variable
        - ✅ Simplifies model
        - ⚠️ Loses information from that variable
        - ⚠️ Requires manual exclusion
        
        **Option 4: Standard (MLE)** 🔴 (NOT RECOMMENDED)
        - \u274c May not converge
        - \u274c Infinite coefficients
        - \u274c Missing p-values
        - \u274c Invalid results
        - \u274c **DO NOT USE with perfect separation!**
        
        **Best Practice Summary:**
        1. Load your data
        2. Select "Auto (Recommended)" method
        3. Click "Run Logistic Regression"
        4. Done! App handles everything automatically
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Interpretation Example
        
        **Model Output:**
        - Variable: Smoking
        - aOR = 1.8 (95% CI: 1.2 - 2.4)
        - p = 0.003
        
        **Interpretation:** 
        Smoking is associated with 1.8× increased odds of outcome (compared to non-smoking), adjusting for other variables. This difference is statistically significant (p < 0.05), and we're 95% confident the true OR is between 1.2 and 2.4. ✅
        
        ---
        
        ### 💾 Future Expansions
        
        Planned additions to this tab:
        - **Multinomial Logistic Regression** (3+ unordered outcomes)
        - **Ordinal Logistic Regression** (3+ ordered outcomes)
        - **Mixed Effects Logistic** (clustered/repeated data)
        """)
