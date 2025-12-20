import streamlit as st
import pandas as pd
import numpy as np
import table_one  # Import from root
import psm_lib  # Import from root
from logger import get_logger

logger = get_logger(__name__)

# 🟢 NEW: Helper function to select between original and matched datasets
def _get_dataset_for_table1(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Helper: เลือกระหว่าง original vs matched dataset สำหรับ Table 1
    คืนค่า: (selected_df, label_str)
    """
    has_matched = (
        st.session_state.get("is_matched", False)
        and st.session_state.get("df_matched") is not None
    )

    if has_matched:
        col1, _ = st.columns([2, 1])
        with col1:
            data_source = st.radio(
                "📄 Select Dataset:",
                ["📊 Original Data", "✅ Matched Data (from PSM)"],
                index=0,  # default Original สำหรับ Table 1
                horizontal=True,
                key="table1_data_source",
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
    Render the "Table 1 & Matching" Streamlit interface with four subtabs for baseline characteristics, propensity score matching, matched data view, and reference/interpretation.
    
    Displays interactive controls to select grouping variables, characteristics, treatment/outcome/covariates, and advanced matching settings; generates Table 1 HTML, performs propensity score calculation and 1:1 nearest-neighbor matching, shows balance diagnostics (SMD and Love plot), previews/downloads matched data, and provides explanatory guidance. Handles session state for persisted HTML output and displays user-facing errors and warnings when inputs are invalid.
    
    Parameters:
        df (pandas.DataFrame): The dataset to analyze and display in the UI.
        var_meta (Mapping): Metadata for variables (e.g., display names, types, formatting) used when generating Table 1 and reports.
    """
    st.subheader("📋 Table 1 & Matching")
    
    # 🟢 NEW: Create four subtabs (added Matched Data View)
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 Baseline Characteristics (Table 1)",
        "⚖️ Propensity Score Matching",
        "✅ Matched Data View",  # 🟢 NEW SUBTAB
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
        * **Odds Ratio (OR):** Automatically calculated for categorical variables when there are **exactly 2 groups**.
        * **P-value & Test Used:** Tests for statistically significant differences in characteristics across groups.
        * **Automatically selects the appropriate test for P-value** (e.g., t-test, Chi-square, Kruskal-Wallis) based on the variable type and distribution.

        **Variable Selection:**
        * **Grouping Variable (Split):** The primary categorical variable used to stratify the **dataset** (e.g., 'Treatment' or 'Outcome').
        * **Characteristics:** All other variables (numeric/categorical) to be summarized and compared.
        """)
        
        # 🟢 NEW: Display matched data status and selector
        if st.session_state.get("is_matched", False):
            st.info("✅ มี Matched dataset จาก PSM แล้ว สามารถเลือกใช้ด้านล่างได้")
        
        # 🟢 NEW: Select dataset (original or matched)
        t1_df, t1_label = _get_dataset_for_table1(df)
        st.write(f"**Using:** {t1_label}")
        st.write(f"**Rows:** {len(t1_df)} | **Columns:** {len(t1_df.columns)}")
        
        all_cols = t1_df.columns.tolist()
        grp_idx = 0
        for i, c in enumerate(all_cols):
            if 'group' in c.lower() or 'treat' in c.lower(): 
                grp_idx = i
                break
        
        c1, c2 = st.columns([1, 2])
        with c1:
            col_group = st.selectbox("Group By (Column):", ["None", *all_cols], index=grp_idx+1, key='t1_group')
            
            # 🟢 Added OR Display Option
            or_style_display = st.radio(
                "Choose OR Style:",
                ["All Levels (Every Level vs Ref)", "Simple (Single Line/Risk vs Ref)"],
                index=0,
                key="or_style_radio",
                help="All Levels: Shows OR for every sub-group (Detailed). Simple: Shows OR for only one comparison (Concise)."
            )
            # Map display string to internal code
            or_style_code = 'all_levels' if "All Levels" in or_style_display else 'simple'
            
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
                    # Calling the generate_table function in table_one.py with or_style
                    # 🟢 UPDATED: Use t1_df (selected dataset) instead of df
                    html_t1 = table_one.generate_table(t1_df, selected_vars, grp, var_meta, or_style=or_style_code)
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
        **Step 4:** View matched data in **Subtab 3 (Matched Data View)** and use for subsequent analyses
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

        # Check for all-NaN treatment column
        if len(unique_treat) == 0:
            st.error(f"⚠️ Variable '{treat_col}' contains only missing values. Please select a different treatment variable.")
        # Check exactly 2 groups
        elif len(unique_treat) != 2:
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
                            
                            # 🟢 NEW: Store matched data in session state
                            st.session_state.df_matched = df_matched
                            st.session_state.is_matched = True
                            st.session_state.matched_treatment_col = treat_col
                            st.session_state.matched_covariates = cov_cols
                            logger.info("💾 Matched data stored in session state. Rows: %d", len(df_matched))
                            
                            # C. Check Balance (SMD)
                            smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
                            smd_post = psm_lib.calculate_smd(df_matched, final_treat_col, final_cov_cols)
                            
                            # Tabs for results
                            t_res1, t_res2, t_res3 = st.tabs(["📊 Balance Check (Love Plot)", "📋 Matched Data Preview", "📉 Outcome Analysis"])
                            
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
                                st.write(f"✅ Matched Dataset Preview ({len(df_matched)} rows):")
                                st.dataframe(df_matched.head(50), use_container_width=True)
                                st.info("📌 Full matched data is available in the **Matched Data View** subtab.")
                                
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
                                {'type':'text', 'data': f"PSM Matching Report. Treatment: {treat_col}" + (f" (Mapped: '{target_val}' → 1)" if target_val else " (Binary: 1=Treated)")},
                                {'type':'table', 'data': smd_merge},
                                {'type':'plot', 'data': fig_love}
                            ]
                            html_rep = psm_lib.generate_psm_report("PSM Analysis", elements)
                            st.download_button("📥 Download Report HTML", html_rep, "psm_report.html", "text/html", key='dl_psm_report')
                            
                    except Exception as e:
                        st.error(f"Error during PSM ({type(e).__name__}): {e}")
                        logger.exception("PSM analysis failed")

    # ==========================================
    # SUBTAB 3: MATCHED DATA VIEW (🟢 NEW)
    # ==========================================
    with sub_tab3:
        st.markdown("### ✅ Matched Data View & Export")
        
        if st.session_state.is_matched and st.session_state.df_matched is not None:
            df_m = st.session_state.df_matched
            
            st.success(f"""
            ✅ **Matched Dataset Ready**
            - Total rows: **{len(df_m)}**
            - Original rows: **{len(df)}**
            - Excluded: **{len(df) - len(df_m)}** rows
            - Treatment variable: **{st.session_state.matched_treatment_col}**
            """)
            
            # Summary Statistics
            with st.expander("📊 Summary Statistics", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Group Sizes:**")
                    if st.session_state.matched_treatment_col in df_m.columns:
                        grp_counts = df_m[st.session_state.matched_treatment_col].value_counts().sort_index()
                        st.write(grp_counts)
                with col2:
                    st.markdown("**Data Types:**")
                    dtype_counts = df_m.dtypes.astype(str).value_counts()
                    st.write(dtype_counts)
            
            # Data Filter & Preview
            with st.expander("🔍 Filter & Preview", expanded=True):
                total_rows = len(df_m)

                # 🟢 FIXED: กรณีแถว ≤ 10: ไม่ต้องมี slider แสดงทั้งหมดไปเลย
                if total_rows <= 10:
                    n_display = total_rows
                    st.caption(f"Showing all {total_rows} rows (too few for slider).")
                else:
                    # 🟢 FIXED: ให้ min < max เสมอ
                    min_rows = 10
                    max_rows = total_rows
                    default_rows = min(50, max_rows)

                    # ถ้าแถวน้อยกว่า 20 ให้ step = 1 จะเลื่อนละเอียดได้
                    step = 10 if max_rows >= 20 else 1

                    n_display = st.slider(
                        "Rows to display:",
                        min_value=min_rows,
                        max_value=max_rows,
                        value=default_rows,
                        step=step,
                    )

                st.dataframe(df_m.head(n_display), use_container_width=True, height=400)

            # Download Options
            st.markdown("### 📥 Export Matched Data")
            col_csv, col_txt = st.columns(2)
            
            with col_csv:
                csv_data = df_m.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 CSV Format",
                    data=csv_data,
                    file_name="matched_data.csv",
                    mime="text/csv",
                    key="dl_matched_csv_view"
                )
            
            with col_txt:
                # Try to export as Excel if openpyxl available
                try:
                    import openpyxl
                    from io import BytesIO
                    
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_m.to_excel(writer, sheet_name='Matched Data', index=False)
                    
                    st.download_button(
                        label="📥 Excel Format",
                        data=buffer.getvalue(),
                        file_name="matched_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_matched_xlsx_view"
                    )
                except ImportError:
                    logger.debug("openpyxl not available for Excel export")
                    st.info("💡 Excel export requires openpyxl package")
            
            # Statistics by Treatment Group
            st.markdown("### 📈 Statistics by Group")
            if st.session_state.matched_treatment_col in df_m.columns:
                treat_col_m = st.session_state.matched_treatment_col
                
                # Show numeric columns summary
                numeric_cols = df_m.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != treat_col_m]
                
                if numeric_cols:
                    selected_col = st.selectbox("Select numeric variable to compare:", numeric_cols, key='matched_numeric_select')
                    
                    summary_tab1, summary_tab2 = st.tabs(["📊 Descriptive Stats", "📉 Visualization"])
                    
                    with summary_tab1:
                        summary_stats = df_m.groupby(treat_col_m)[selected_col].describe()
                        st.dataframe(summary_stats, use_container_width=True)
                    
                    with summary_tab2:
                        import plotly.express as px
                        fig = px.box(df_m, x=treat_col_m, y=selected_col, title=f"{selected_col} by {treat_col_m}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Reset Button
            if st.button("🔄 Clear Matched Data & Return to Analysis", type="secondary", key='btn_clear_matched'):
                st.session_state.df_matched = None
                st.session_state.is_matched = False
                st.session_state.matched_treatment_col = None
                st.session_state.matched_covariates = []
                logger.info("🔄 Matched data cleared")
                st.rerun()
        else:
            st.info("""
            ℹ️ **No matched data available yet.**
            
            1. Go to **Subtab 2 (Propensity Score Matching)**
            2. Configure variables and run PSM matching
            3. Return here to view and export matched data
            """)

    # ==========================================
    # SUBTAB 4: REFERENCE & INTERPRETATION (FORMERLY SUBTAB 3)
    # ==========================================
    with sub_tab4:
        st.markdown("##### 📚 Quick Reference: Table 1 & Matching")
        
        st.info("""
        **🎯 When to Use What:**
        
        | Analysis | Purpose | Output |
        |----------|---------|--------|
        | **Table 1** | Compare baseline characteristics | Mean/Median, % counts, p-values |
        | **PSM** | Balance groups (remove confounding) | SMD pre/post, Love plot, matched data |
        | **Matched Data View** | Export & summarize matched cohort | CSV/Excel, descriptive stats |
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
            
            **✨ NEW:** Can now compare both Original and Matched datasets!
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
        → Use **Table 1** (Tab 1) to check - compare Original vs Matched after PSM ✨
        
        **Question: They're imbalanced. Can I fix it?**
        → Use **PSM** (Tab 2) to match
        
        **Question: After PSM, are they balanced?**
        → Check **SMD < 0.1** in Love plot ✅
        
        **Question: Now what? Use for analysis?**
        → Export from **Matched Data View** (Tab 3) and select **"✅ Matched Data"** in other analysis tabs ✅
        """)
