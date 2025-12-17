import streamlit as st
import pandas as pd
import numpy as np
import table_one  # Import from root
import psm_lib  # Import from root
from logger import get_logger

logger = get_logger(__name__)

def render(df, var_meta):
    st.subheader("📋 Table 1 & Matching")
    
    # Create three subtabs (added Reference & Interpretation)
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📊 Baseline Characteristics (Table 1)",
        "⚖️ Propensity Score Matching",
        "ℹ️ Reference & Interpretation"
    ])
    
    # ==========================================
    # SUBTAB 1: BASELINE CHARACTERISTICS (Table 1)
    # ==========================================
    with sub_tab1:
        st.markdown("### Baseline Characteristics (Table 1)")
        st.info("""
        **💡 Guide:** Summarizes key demographics and patient characteristics, stratified by a **Grouping Variable**, to assess **group comparability**.

        **Presentation:**
        * **Numeric:** Mean ± SD (Normally Distributed Data) or Median (IQR) (**Non**-Normally Distributed Data).
        * **Categorical:** Count (Percentage).
        * **Odds Ratio (OR):** Automatically calculated for categorical variables when there are **exactly 2 groups** (One-vs-Rest method).
        * **P-value & Test Used:** Tests for statistically significant differences in characteristics across groups.
        * **Automatically selects the appropriate test for P-value** (e.g., t-test, Chi-square, Kruskal-Wallis) based on the variable type and distribution.

        **Variable Selection:**
        * **Grouping Variable (Split):** The primary categorical variable used to stratify the **dataset** (e.g., 'Treatment' or 'Outcome').
        * **Characteristics:** All other variables (numeric/categorical) to be summarized and compared.
        """)
        
        all_cols = df.columns.tolist()
        grp_idx = 0
        for i, c in enumerate(all_cols):
            if 'group' in c.lower() or 'treat' in c.lower(): 
                grp_idx = i
                break
        
        c1, c2 = st.columns([1, 2])
        with c1:
            col_group = st.selectbox("Group By (Column):", ["None", *all_cols], index=grp_idx+1, key='t1_group')
        with c2:
            def_vars = [c for c in all_cols if c != col_group]
            selected_vars = st.multiselect("Include Variables:", all_cols, default=def_vars, key='t1_vars')
            
        run_col, dl_col = st.columns([1, 1])
        
        if 'html_output_t1' not in st.session_state:
            st.session_state.html_output_t1 = None

        if run_col.button("📊 Generate Table 1", type="primary", key='btn_t1'):
            with st.spinner("Generating..."):
                try:
                    grp = None if col_group == "None" else col_group
                    # Calling the generate_table function in table_one.py
                    html_t1 = table_one.generate_table(df, selected_vars, grp, var_meta)
                    st.session_state.html_output_t1 = html_t1 
                    st.components.v1.html(html_t1, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Table 1 generation failed")
                    st.session_state.html_output_t1 = None 
                    
        with dl_col:
            if st.session_state.html_output_t1:
                st.download_button("📥 Download HTML", st.session_state.html_output_t1, "table1.html", "text/html", key='dl_btn_t1')
            else:
                st.button("📥 Download HTML", disabled=True, key='ph_t1')
    
    # ==========================================
    # SUBTAB 2: PROPENSITY SCORE MATCHING
    # ==========================================
    with sub_tab2:
        st.markdown("### Propensity Score Matching (PSM)")
        st.info("""
        **💡 Concept:** Matches patients in the Treatment group with similar patients in the Control group using **Propensity Scores**.
        * **Goal:** To reduce selection bias and mimic a Randomized Controlled Trial (RCT).
        * **Algorithm:** Nearest Neighbor 1:1 Matching (Greedy, Without Replacement).
        
        **Step 1:** Check baseline imbalance in **Subtab 1 (Table 1)** - Look at P-values and group differences
        **Step 2:** If imbalanced (p<0.05 or differences exist) → Run PSM below
        **Step 3:** Check balance after matching using SMD table (SMD < 0.1 = good balance)
        """)

        all_cols = df.columns.tolist()
        if not all_cols:
            st.error("Dataset has no columns to analyze.")
            return
            
        # --- 1. Variable Selection ---
        c1, c2 = st.columns(2)
        
        # Auto-detect Treatment (Look for binary variables)
        treat_idx = 0
        for i, c in enumerate(all_cols):
            if df[c].nunique() == 2 and ('group' in c.lower() or 'treat' in c.lower()):
                treat_idx = i
                break
                
        treat_col = c1.selectbox("💊 Treatment Variable (Binary):", all_cols, index=treat_idx, key='psm_treat')
        outcome_col = c2.selectbox("🎯 Outcome Variable (Optional):", ["None", *all_cols], key='psm_outcome')
        
        # Covariates Selection
        cov_candidates = [c for c in all_cols if c not in [treat_col, outcome_col]]
        default_covs = [c for c in cov_candidates if any(x in c.lower() for x in ['age', 'sex', 'bmi', 'hyper', 'comorb'])]
        
        cov_cols = st.multiselect("📊 Covariates (Confounders):", cov_candidates, 
                                  default=default_covs,
                                  key='psm_cov')

        # --- 2. Data Preparation ---
        df_analysis = df.copy()
        unique_treat = df_analysis[treat_col].dropna().unique()
        
        # Check exactly 2 groups
        if len(unique_treat) != 2:
            st.warning(f"⚠️ Variable '{treat_col}' must have exactly 2 unique values (Found: {len(unique_treat)}). PSM cannot run.")
        else:
            # Check if 0/1 or needs encoding
            is_numeric_binary = set(unique_treat).issubset({0, 1}) or set(unique_treat).issubset({0.0, 1.0})
            
            target_val = None
            final_treat_col = treat_col

            if not is_numeric_binary:
                st.warning(f"⚠️ Variable '{treat_col}' is text/categorical. Please specify which value is the **Treatment Group**.")
                
                c_sel, c_msg = st.columns([2, 1]) 
                
                with c_sel:
                    target_val = st.selectbox("Select value for 'Treatment/Case' (will be mapped to 1):", unique_treat, key='psm_target_select')

                # Check for missing treatment values
                if df_analysis[treat_col].isna().any():
                    st.warning(f"⚠️ Treatment variable '{treat_col}' contains {df_analysis[treat_col].isna().sum()} missing values. These will be excluded from analysis.")
                    df_analysis = df_analysis.dropna(subset=[treat_col])
                    
                final_treat_col = f"{treat_col}_encoded"
                df_analysis[final_treat_col] = np.where(df_analysis[treat_col] == target_val, 1, 0)
                
                with c_msg:
                    st.write("")
                    st.success(f"✅ Mapped: '{target_val}' = 1")
            
            # Handle categorical covariates (One-hot Encoding)
            if cov_cols:
                # Detect string, object, and categorical dtypes
                cat_covs = [c for c in cov_cols if pd.api.types.is_string_dtype(df_analysis[c]) or 
                            pd.api.types.is_categorical_dtype(df_analysis[c]) or 
                            pd.api.types.is_object_dtype(df_analysis[c])]
                if cat_covs:
                    df_analysis = pd.get_dummies(df_analysis, columns=cat_covs, drop_first=True)
                # Exclude encoded treatment from new columns
                new_cols = [c for c in df_analysis.columns if c not in df.columns and c != final_treat_col]
                final_cov_cols = [c for c in cov_cols if c not in cat_covs] + new_cols
            else:
                final_cov_cols = []

            # PSM Settings
            with st.expander("⚙️ Advanced Settings"):
               caliper = st.slider(
                   "Caliper Width (SD of Logit):", 
                   0.05, 1.0, 0.5, 0.05,
                   help="Maximum distance for matching. Higher values allow more matches but may reduce balance. Common range: 0.1-0.5."
                   )
            
            # --- 3. Run Analysis ---
            if st.button("🚀 Run Matching", key='btn_psm'):
                if not final_cov_cols:
                    st.error("Please select at least one covariate.")
                else:
                    try:
                        # A. Calculate PS
                        with st.spinner("Calculating Propensity Scores..."):
                            df_ps, _model = psm_lib.calculate_ps(df_analysis, final_treat_col, final_cov_cols)
                        
                        # B. Perform Matching
                        with st.spinner("Matching Patients..."):
                            df_matched, msg = psm_lib.perform_matching(df_ps, final_treat_col, 'ps_logit', caliper)
                        
                        if df_matched is None:
                            st.error(msg)
                        else:
                            st.success(f"✅ Matching Complete! {msg}")
                            
                            # C. Check Balance (SMD)
                            smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
                            smd_post = psm_lib.calculate_smd(df_matched, final_treat_col, final_cov_cols)
                            
                            # Tabs for results
                            t_res1, t_res2, t_res3 = st.tabs(["📊 Balance Check (Love Plot)", "📋 Matched Data", "📉 Outcome Analysis"])
                            
                            with t_res1:
                                c_plot, c_tab = st.columns([2, 1])
                                fig_love = psm_lib.plot_love_plot(smd_pre, smd_post)
                                c_plot.plotly_chart(fig_love, use_container_width=True)
                                
                                c_tab.markdown("**SMD Table:**")
                                smd_merge = pd.merge(smd_pre, smd_post, on='Variable', suffixes=('_Pre', '_Post'))
                                
                                c_tab.dataframe(smd_merge.style.format({
                                    'SMD_Pre': '{:.4f}', 
                                    'SMD_Post': '{:.4f}'
                                }))
                                
                                c_tab.caption("*SMD < 0.1 indicates good balance.*")

                            with t_res2:
                                st.write(f"Matched Dataset ({len(df_matched)} rows):")
                                st.dataframe(df_matched.head(50))
                                
                                csv = df_matched.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download Matched CSV", csv, "matched_data.csv", "text/csv", key='dl_matched')

                            with t_res3:
                                if outcome_col != "None":
                                    # Validate outcome is numeric
                                    if not pd.api.types.is_numeric_dtype(df_matched[outcome_col]):
                                        st.warning(f"⚠️ Outcome variable '{outcome_col}' is not numeric. Comparison skipped.")
                                    else:
                                        st.markdown(f"##### Outcome Comparison: {outcome_col}")
                                        res_stats = df_matched.groupby(final_treat_col)[outcome_col].mean()
                                        st.write(res_stats)
                                    
                                    st.info("💡 Note: This is a simple comparison. For statistical significance, use the Hypothesis Testing tab with the 'Matched Dataset'.")
                                else:
                                    st.write("Select an Outcome variable to see comparison.")
                                    
                            # Generate Report
                            elements = [
                                {'type':'text', 'data': f"PSM Matching Report. Treatment: {treat_col} (Mapped 1={target_val if target_val else '1'})"},
                                {'type':'table', 'data': smd_merge},
                                {'type':'plot', 'data': fig_love}
                            ]
                            html_rep = psm_lib.generate_psm_report("PSM Analysis", elements)
                            st.download_button("📥 Download Report HTML", html_rep, "psm_report.html", "text/html", key='dl_psm_report')
                            
                    except Exception as e:
                        st.error(f"Error during PSM ({type(e).__name__}): {e}")
                        logger.exception("PSM analysis failed")

    # ==========================================
    # SUBTAB 3: REFERENCE & INTERPRETATION (NEW)
    # ==========================================
    with sub_tab3:
        st.markdown("##### 📚 Quick Reference: Table 1 & Matching")
        
        st.info("""
        **🎯 When to Use What:**
        
        | Analysis | Purpose | Output |
        |----------|---------|--------|
        | **Table 1** | Compare baseline characteristics | Mean/Median, % counts, p-values |
        | **PSM** | Balance groups (remove confounding) | SMD pre/post, Love plot, matched data |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Table 1 (Baseline)")
            st.markdown("""
            **When to Use:**
            - RCTs: Assess randomization balance
            - Observational: Check comparability
            - Publication standard
            
            **Interpretation:**
            - **p < 0.05**: Variables differ ⚠️
            - **p ≥ 0.05**: Balanced ✅
            
            **Presentation:**
            - Numeric: Mean ± SD (normal) or Median (IQR)
            - Categorical: Count (%) and OR
            - Include N for each group
            
            **Common Mistakes:**
            - Using t-test for non-normal data ❌
            - Multiple testing without adjustment
            - Assuming p>0.05 = no confounding
            """)
        
        with col2:
            st.markdown("### PSM (Propensity Matching)")
            st.markdown("""
            **When to Use:**
            - Observational studies (imbalance)
            - Can't randomize
            - Adjust for confounders
            - Mimic RCT
            
            **SMD (Balance Check):**
            - **SMD < 0.1**: Good ✅
            - **0.1-0.2**: Acceptable ⚠️
            - **> 0.2**: Poor ❌
            
            **Key Steps:**
            1. Check Table 1 (p-values)
            2. Run PSM if imbalanced
            3. Check SMD pre vs post
            4. Use matched data for analysis
            
            **Common Mistakes:**
            - Not checking Table 1 first ❌
            - Only checking p-values after PSM
            - PSM on balanced groups
            - Ignoring sample size loss
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Decision Guide
        
        **Question: Do my groups differ at baseline?**
        → Use **Table 1** (Tab 1) to check
        
        **Question: They're imbalanced. Can I fix it?**
        → Use **PSM** (Tab 2) to match
        
        **Question: After PSM, are they balanced?**
        → Check **SMD < 0.1** in Love plot ✅
        
        **Question: Now what? Use for analysis?**
        → YES: Use matched dataset for hypothesis testing
        """)
