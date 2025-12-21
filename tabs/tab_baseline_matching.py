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
    Helper: Select between original vs matched dataset for Table 1
    Returns: (selected_df, label_str)
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
                index=0,  # default Original for Table 1
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
    Render the "Table 1 & Matching" Streamlit interface with subtabs for baseline characteristics, 
    propensity score matching, matched data view, and reference/interpretation.
    
    Displays interactive controls to select grouping variables, characteristics, treatment/outcome/covariates, 
    and advanced matching settings; generates Table 1 HTML, performs propensity score calculation and 1:1 
    nearest-neighbor matching, shows balance diagnostics (SMD and Love plot), previews/downloads matched data, 
    and provides explanatory guidance. Handles session state for persisted HTML output and displays user-facing 
    errors and warnings when inputs are invalid.
    
    Parameters:
        df (pandas.DataFrame): The dataset to analyze and display in the UI.
        var_meta (Mapping): Metadata for variables (e.g., display names, types, formatting) used when 
                           generating Table 1 and reports.
    """
    st.subheader("📋 Table 1 & Matching")
    
    # 🟢 NEW: Create four subtabs
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 Baseline Characteristics (Table 1)",
        "⚖️ Propensity Score Matching",
        "✅ Matched Data View",
        "ℹ️ Reference & Interpretation"
    ])
    
    # ==========================================
    # SUBTAB 1: BASELINE CHARACTERISTICS (Table 1)
    # ==========================================
    with sub_tab1:
        st.markdown("### Baseline Characteristics (Table 1)")
        st.info("""
        **💡 Guide:** Summarizes key demographics and patient characteristics, stratified by a **Grouping Variable**, 
        to assess **group comparability**.

        **Presentation:**
        * **Numeric:** Mean ± SD (Normally Distributed Data) or Median (IQR) (**Non**-Normally Distributed Data).
        * **Categorical:** Count (Percentage).
        * **Odds Ratio (OR):** Automatically calculated for categorical variables when there are **exactly 2 groups**.
        * **P-value & Test Used:** Tests for statistically significant differences in characteristics across groups.
        * **Automatically selects the appropriate test for P-value** (e.g., t-test, Chi-square, Kruskal-Wallis) 
        based on the variable type and distribution.

        **Variable Selection:**
        * **Grouping Variable (Split):** The primary categorical variable used to stratify the **dataset** 
        (e.g., 'Treatment' or 'Outcome').
        * **Characteristics:** All other variables (numeric/categorical) to be summarized and compared.
        """)
        
        # 🟢 Display matched data status and selector
        if st.session_state.get("is_matched", False):
            st.info("✅ **Matched Dataset Available** - You can select it below for analysis")
        
        # 🟢 Select dataset (original or matched)
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
        st.markdown("### ⚖️ Propensity Score Matching (PSM)")
        st.info("""
        **💡 Workflow:**
        1. **Select variables** → Treatment, Outcome, Confounders
        2. **Configure matching** → Choose presets or custom settings
        3. **Run matching** → System calculates propensity scores
        4. **Review results** → Check balance metrics (SMD < 0.1 is good)
        5. **Export matched data** → Use for downstream analysis
        
        **Goal:** Reduce selection bias and create balanced treatment groups mimicking an RCT.
        """)

        all_cols = df.columns.tolist()
        if not all_cols:
            st.error("Dataset has no columns to analyze.")
            return
            
        # ==========================================
        # SECTION 1: VARIABLE CONFIGURATION WITH PRESETS
        # ==========================================
        st.subheader("Step 1️⃣: Configure Variables")
        
        col_preset, col_manual = st.columns([1, 2], gap="large")
        
        with col_preset:
            st.markdown("**Quick Presets:**")
            preset_choice = st.radio(
                "Start with template:",
                ["🔧 Custom (Manual)", "👥 Demographics", "🏥 Full Medical"],
                index=0,
                label_visibility="collapsed"
            )
            
            st.caption("""
            **Presets include:**
            - 👥 Demographics: Age, Sex, BMI
            - 🏥 Full Medical: Age, Sex, BMI, Comorbidities, Lab values
            - 🔧 Custom: You choose all variables
            """)
        
        with col_manual:
            st.markdown("**Manual Selection:**")
            
            # Auto-detect Treatment
            treat_idx = 0
            for i, c in enumerate(all_cols):
                if df[c].nunique() == 2 and ('group' in c.lower() or 'treat' in c.lower()):
                    treat_idx = i
                    break
                    
            treat_col = st.selectbox(
                "💊 Treatment Variable (Binary)",
                all_cols,
                index=treat_idx,
                key='psm_treat',
                help="Must have exactly 2 unique values"
            )
            
            outcome_col = st.selectbox(
                "🎯 Outcome Variable (Optional)",
                ["⊘ None / Skip", *all_cols],
                index=0,
                key='psm_outcome',
                help="For comparison after matching"
            )
            
            # Smart covariate defaults based on preset
            cov_candidates = [c for c in all_cols if c not in [treat_col, outcome_col if outcome_col != "⊘ None / Skip" else ""]]
            
            if preset_choice == "👥 Demographics":
                default_covs = [c for c in cov_candidates if any(x in c.lower() for x in ['age', 'sex', 'bmi'])]
            elif preset_choice == "🏥 Full Medical":
                default_covs = [c for c in cov_candidates if any(x in c.lower() for x in ['age', 'sex', 'bmi', 'comorb', 'hyper', 'diab', 'lab'])]
            else:
                default_covs = []
            
            cov_cols = st.multiselect(
                "📊 Confounding Variables",
                cov_candidates,
                default=default_covs,
                key='psm_cov',
                help="Select all baseline variables that might affect treatment assignment"
            )
        
        # ==========================================
        # CONFIGURATION SUMMARY
        # ==========================================
        config_valid = len(cov_cols) > 0
        summary_items = [
            f"💊 **Treatment:** `{treat_col}`",
            f"🎯 **Outcome:** `{outcome_col if outcome_col != '⊘ None / Skip' else 'Skip'}`",
            f"📊 **Confounders:** {len(cov_cols)} selected"
        ]
        
        if not config_valid:
            summary_items.append("❌ **Error:** Please select at least one covariate")
        
        st.info("**✅ Configuration Summary:**\n" + "\n".join(summary_items))
        
        # ==========================================
        # SECTION 2: ADVANCED SETTINGS (IMPROVED CALIPER)
        # ==========================================
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown("**Caliper Width (Matching Tolerance)**")
            
            col_cal, col_info = st.columns([2, 1])
            
            with col_cal:
                cal_presets = {
                    "🔓 Very Loose (1.0×SD) - Most matches, weaker balance": 1.0,
                    "📊 Loose (0.5×SD) - Balanced approach": 0.5,
                    "⚖️ Standard (0.25×SD) - RECOMMENDED ← START HERE": 0.25,
                    "🔒 Strict (0.1×SD) - Fewer matches, excellent balance": 0.1,
                }
                
                cal_label = st.radio(
                    "Select matching strictness:",
                    list(cal_presets.keys()),
                    index=2,
                    label_visibility="collapsed"
                )
                
                caliper = cal_presets[cal_label]
            
            with col_info:
                st.markdown("**📌 About Caliper:**")
                st.caption("Caliper = max distance to match treated with control. Wider = more matches, less balance.")
        
        # ==========================================
        # SECTION 3: RUN MATCHING
        # ==========================================
        st.subheader("Step 2️⃣: Run Matching")
        
        col_run, col_status = st.columns([2, 1])
        
        run_button = col_run.button(
            "🚀 Run Propensity Score Matching",
            type="primary",
            disabled=not config_valid,
            use_container_width=True,
            key='btn_psm'
        )
        
        with col_status:
            if config_valid:
                st.success("✅ Ready to run")
            else:
                st.error("⚠️ Select covariates")
        
        # ==========================================
        # SECTION 4: EXECUTION & RESULTS
        # ==========================================
        if run_button:
            if not cov_cols:
                st.error("Please select at least one covariate.")
            else:
                try:
                    # --- Data Preparation ---
                    df_analysis = df.copy()
                    unique_treat = df_analysis[treat_col].dropna().unique()
                    
                    if len(unique_treat) == 0:
                        st.error(f"⚠️ Variable '{treat_col}' contains only missing values.")
                    elif len(unique_treat) != 2:
                        st.warning(f"⚠️ Variable '{treat_col}' must have exactly 2 unique values (Found: {len(unique_treat)}).")
                    else:
                        # Handle categorical treatment
                        is_numeric_binary = set(unique_treat).issubset({0, 1}) or set(unique_treat).issubset({0.0, 1.0})
                        target_val = None
                        final_treat_col = treat_col
                        
                        if not is_numeric_binary:
                            val_counts = df_analysis[treat_col].value_counts()
                            minor_val = val_counts.index[-1]
                            major_val = val_counts.index[0]
                            
                            st.warning(f"""
                            ⚠️ **Treatment variable is categorical:**
                            - **{major_val}:** {val_counts[major_val]} patients ({val_counts[major_val]/len(df_analysis)*100:.1f}%)
                            - **{minor_val}:** {val_counts[minor_val]} patients ({val_counts[minor_val]/len(df_analysis)*100:.1f}%)
                            
                            Using **{minor_val}** as treatment (usually the minority group).
                            """)
                            
                            target_val = minor_val
                            final_treat_col = f"{treat_col}_encoded"
                            df_analysis[final_treat_col] = np.where(df_analysis[treat_col] == target_val, 1, 0)
                        
                        # Handle categorical covariates
                        if cov_cols:
                            cat_covs = [c for c in cov_cols if pd.api.types.is_string_dtype(df_analysis[c]) or 
                                       pd.api.types.is_categorical_dtype(df_analysis[c]) or 
                                       pd.api.types.is_object_dtype(df_analysis[c])]
                            if cat_covs:
                                df_analysis = pd.get_dummies(df_analysis, columns=cat_covs, drop_first=True)
                            new_cols = [c for c in df_analysis.columns if c not in df.columns and c != final_treat_col]
                            final_cov_cols = [c for c in cov_cols if c not in cat_covs] + new_cols
                        else:
                            final_cov_cols = []
                        
                        # --- Calculate Propensity Scores ---
                        with st.spinner("⏳ Calculating propensity scores..."):
                            df_ps, _model = psm_lib.calculate_ps(df_analysis, final_treat_col, final_cov_cols)
                        
                        # --- Perform Matching ---
                        with st.spinner("⏳ Matching patients..."):
                            df_matched, msg = psm_lib.perform_matching(df_ps, final_treat_col, 'ps_logit', caliper)
                        
                        if df_matched is None:
                            st.error(f"❌ Matching failed: {msg}")
                        else:
                            # Calculate SMD (including categorical)
                            smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
                            smd_post = psm_lib.calculate_smd(df_matched, final_treat_col, final_cov_cols)
                            
                            # 🟢 NEW: Add categorical SMD
                            smd_pre_cat = _calculate_categorical_smd(df_ps, final_treat_col, cat_covs if 'cat_covs' in locals() else [])
                            smd_post_cat = _calculate_categorical_smd(df_matched, final_treat_col, cat_covs if 'cat_covs' in locals() else [])
                            
                            smd_pre = pd.concat([smd_pre, smd_pre_cat], ignore_index=True)
                            smd_post = pd.concat([smd_post, smd_post_cat], ignore_index=True)
                            
                            st.success(f"✅ {msg}")
                            
                            # Store in session
                            st.session_state.df_matched = df_matched
                            st.session_state.is_matched = True
                            st.session_state.matched_treatment_col = treat_col
                            st.session_state.matched_covariates = cov_cols
                            logger.info("💾 Matched data stored. Rows: %d", len(df_matched))
                            
                            st.divider()
                            
                            # ==========================================
                            # 🟢 PRIORITY 1: QUALITY METRICS DASHBOARD
                            # ==========================================
                            st.subheader("Step 3️⃣: Match Quality Summary")
                            
                            match_rate = (df_matched[final_treat_col].sum() / df_ps[final_treat_col].sum()) * 100
                            good_balance_count = (smd_post['SMD'] < 0.1).sum()
                            
                            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                            
                            with m_col1:
                                st.metric(
                                    label="Pairs Matched",
                                    value=f"{df_matched[final_treat_col].sum():.0f}",
                                    delta=f"({match_rate:.1f}% of {df_ps[final_treat_col].sum():.0f})" 
                                )
                            
                            with m_col2:
                                st.metric(
                                    label="Sample Retained",
                                    value=f"{len(df_matched):,}",
                                    delta=f"({len(df_matched)/len(df_ps)*100:.1f}% of {len(df_ps):,})"
                                )
                            
                            with m_col3:
                                st.metric(
                                    label="Good Balance",
                                    value=f"{good_balance_count}/{len(smd_post)}",
                                    delta=f"(SMD < 0.1)" if good_balance_count == len(smd_post) else f"⚠️ {len(smd_post) - good_balance_count} vars"
                                )
                            
                            with m_col4:
                                smd_merge_qual = smd_pre.merge(smd_post, on='Variable', suffixes=('_pre', '_post'))
                                avg_smd_before = smd_merge_qual['SMD_pre'].mean()
                                avg_smd_after = smd_merge_qual['SMD_post'].mean()
                                improvement = ((avg_smd_before - avg_smd_after) / avg_smd_before * 100) if avg_smd_before > 0 else 0
                                
                                st.metric(
                                    label="SMD Improvement",
                                    value=f"{improvement:.1f}%",
                                    delta="↓ average reduction"
                                )
                            
                            # Balance warning
                            if (smd_post['SMD'] > 0.1).any():
                                bad_vars = smd_post[smd_post['SMD'] > 0.1]['Variable'].tolist()
                                st.warning(f"""
                                ⚠️ **Imbalance remains on {len(bad_vars)} variable(s):**
                                
                                {', '.join(bad_vars[:5])}{'...' if len(bad_vars) > 5 else ''}
                                
                                **Try:** Increase caliper width or check for outliers
                                """, icon="⚠️")
                            else:
                                st.success("✅ **Excellent balance achieved!** All variables have SMD < 0.1", icon="✅")
                            
                            st.divider()
                            
                            # Balance Check Tabs
                            st.subheader("Step 4️⃣: Balance Assessment")
                            
                            bal_tab1, bal_tab2, bal_tab3 = st.tabs([
                                "📉 Love Plot",
                                "📋 SMD Table",
                                "📊 Group Comparison"
                            ])
                            
                            with bal_tab1:
                                fig_love = psm_lib.plot_love_plot(smd_pre, smd_post)
                                st.plotly_chart(fig_love, use_container_width=True)
                                st.caption("Green (diamond) = matched, Red (circle) = unmatched. Target: All on left (SMD < 0.1)")
                            
                            with bal_tab2:
                                smd_merge_display = smd_pre.merge(smd_post, on='Variable', suffixes=(' Before', ' After'))
                                smd_merge_display['Improvement %'] = (
                                    ((smd_merge_display[' Before'] - smd_merge_display[' After']) / smd_merge_display[' Before'] * 100)
                                    .round(1)
                                )
                                
                                st.dataframe(
                                    smd_merge_display.style.format({
                                        ' Before': '{:.4f}',
                                        ' After': '{:.4f}',
                                        'Improvement %': '{:.1f}%'
                                    }).background_gradient(subset=[' After'], cmap='RdYlGn_r', vmin=0, vmax=0.2),
                                    use_container_width=True
                                )
                                st.caption("✅ Good balance: SMD < 0.1 after matching")
                            
                            with bal_tab3:
                                st.write("**Group sizes before and after matching:**")
                                comp_data = pd.DataFrame({
                                    'Stage': ['Before', 'After'],
                                    f'{target_val or "Treated (1)"}': [
                                        (df_ps[final_treat_col] == 1).sum(),
                                        (df_matched[final_treat_col] == 1).sum()
                                    ],
                                    f'Control (0)': [
                                        (df_ps[final_treat_col] == 0).sum(),
                                        (df_matched[final_treat_col] == 0).sum()
                                    ]
                                })
                                st.dataframe(comp_data, use_container_width=True)
                            
                            st.divider()
                            
                            # Export Options
                            st.subheader("Step 5️⃣: Export & Next Steps")
                            
                            col_exp1, col_exp2, col_exp3 = st.columns(3)
                            
                            with col_exp1:
                                csv_data = df_matched.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "📥 Download CSV",
                                    csv_data,
                                    "matched_data.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            
                            with col_exp2:
                                elements = [
                                    {'type': 'text', 'data': f"PSM Report - {treat_col}"},
                                    {'type': 'table', 'data': smd_merge_display},
                                    {'type': 'plot', 'data': fig_love}
                                ]
                                html_rep = psm_lib.generate_psm_report("Propensity Score Matching Report", elements)
                                st.download_button(
                                    "📥 Report HTML",
                                    html_rep,
                                    "psm_report.html",
                                    "text/html",
                                    use_container_width=True
                                )
                            
                            with col_exp3:
                                st.info("✅ Full matched data available in **Subtab 3 (Matched Data View)**")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.exception("PSM analysis failed")

    # ==========================================
    # SUBTAB 3: MATCHED DATA VIEW
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

                if total_rows <= 10:
                    n_display = total_rows
                    st.caption(f"Showing all {total_rows} rows")
                else:
                    min_rows = 10
                    max_rows = total_rows
                    default_rows = min(50, max_rows)
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
    # SUBTAB 4: REFERENCE & INTERPRETATION
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


def _calculate_categorical_smd(df, treatment_col, cat_cols):
    """
    🟢 NEW: Calculate SMD for categorical variables
    Formula: SMD = sqrt(sum((p_treated[i] - p_control[i])^2))
    """
    if not cat_cols:
        return pd.DataFrame(columns=['Variable', 'SMD'])
    
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    for col in cat_cols:
        try:
            categories = df[col].dropna().unique()
            smd_cat = 0
            
            for cat in categories:
                p_treated = (treated[col] == cat).sum() / len(treated) if len(treated) > 0 else 0
                p_control = (control[col] == cat).sum() / len(control) if len(control) > 0 else 0
                smd_cat += (p_treated - p_control) ** 2
            
            smd = np.sqrt(smd_cat)
            smd_data.append({'Variable': col, 'SMD': smd})
        except Exception as e:
            logger.warning(f"Error calculating categorical SMD for {col}: {e}")
            continue
    
    return pd.DataFrame(smd_data)
