import streamlit as st
import pandas as pd
import numpy as np
from logic import process_data_and_generate_html # Import from root
from logger import get_logger
from forest_plot_lib import create_forest_plot  # 🟢 IMPORT NEW LIBRARY
import json

# ✅ FIX IMPORT: Use the updated module from root
from subgroup_analysis_module import SubgroupAnalysisLogit 

logger = get_logger(__name__)

def check_perfect_separation(df, target_col):
    """
    Identify predictor columns that may cause perfect separation with a binary target.
    
    Parameters:
        df (pd.DataFrame): Dataset containing the target and predictor columns.
        target_col (str): Name of the binary target column to evaluate.
    
    Returns:
        list[str]: Predictor column names that have at least one zero cell in their predictor×target contingency table. Returns an empty list if the target is not binary or if an error occurs.
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
def _get_dataset_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Selects which DataFrame to use for analysis, preferring a propensity-score matched dataset when available.
    
    Returns:
        tuple[pd.DataFrame, str]: The selected DataFrame and a label describing the data source and row count (e.g., "✅ Matched Data (123 rows)" or "📊 Original Data (456 rows)").
    """
    has_matched = st.session_state.get('is_matched', False) and st.session_state.get('df_matched') is not None
    
    if has_matched:
        col1, _ = st.columns([2, 1])
        with col1:
            data_source = st.radio(
                "📄 Select Dataset:",
                ["📊 Original Data", "✅ Matched Data (from PSM)"],
                index=1,  # Default to matched data if available
                horizontal=True,
                key="data_source_logit"
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


def _render_logit_subgroup_analysis(df: pd.DataFrame) -> None:
    """
    Render the "Subgroup Analysis" subtab UI for performing and exporting logistic regression subgroup analyses.
    
    Renders Streamlit controls to select a binary outcome, a treatment/exposure, a categorical subgroup (2–10 levels), optional adjustment covariates, and advanced settings; runs SubgroupAnalysisLogit to compute subgroup-specific odds ratios and an interaction test; displays forest plot, summary metrics, a detailed results table, interpretation text, reporting guidance, and export buttons for HTML, CSV, and JSON. Results and the analyzer instance are saved to st.session_state['subgroup_results_logit'] and st.session_state['subgroup_analyzer_logit']; an optional UI flag st.session_state['edit_forest_title_logit'] may be created for editing the plot title. User-facing errors are shown via Streamlit and exceptions are logged.
    
    Parameters:
        df (pd.DataFrame): Dataset used for analysis. Must contain at least one binary column for the outcome and at least one categorical column with 2–10 unique values for subgrouping.
    """
    st.header("🗒️ Subgroup Analysis")
    
    # Info box
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.info("🚀")
        with col2:
            st.markdown("""
            **Test for Treatment-by-Subgroup Interaction**
            
            Determine if treatment effect varies by patient characteristics.
            🙋: Does the drug work differently in men vs women? Young vs old?
            """)
    
    st.markdown("---")
    
    # ========== INPUT SECTION ==========
    st.subheader("📝 Step 1: Select Variables")
    
    col1, col2, col3 = st.columns(3)
    
    # Outcome variable
    with col1:
        binary_cols = [col for col in df.columns if df[col].nunique() == 2]
        if not binary_cols:
            st.error("No binary outcome columns found.")
            return
        outcome_col = st.selectbox(
            "Outcome (Binary)",
            options=binary_cols,
            index=0,
            help="Select binary outcome variable (0/1 or No/Yes)",
            key="logit_sg_outcome"
        )
    
    # Treatment variable
    with col2:
        treatment_col = st.selectbox(
            "Treatment/Exposure",
            options=[col for col in df.columns if col != outcome_col],
            index=0,
            help="Main variable of interest",
            key="logit_sg_treatment"
        )
    
    # Subgroup variable
    with col3:
        subgroup_options = [
            col for col in df.columns
            if col not in [outcome_col, treatment_col]
            and 2 <= df[col].nunique() <= 10
        ]
        if not subgroup_options:
            st.error("No suitable subgroup variable found (need 2–10 categories).")
            return
        subgroup_col = st.selectbox(
            "Stratify By",
            options=subgroup_options,
            help="Categorical variable with 2-10 categories",
            key="logit_sg_subgroup"
        )
    
    st.markdown("---")
    
    # ========== ADJUSTMENT VARIABLES ==========
    st.subheader("📌 Step 2: Adjustment Variables (Optional)")
    
    adjustment_cols = st.multiselect(
        "Select covariates to adjust for:",
        options=[col for col in df.columns 
                if col not in [outcome_col, treatment_col, subgroup_col]],
        help="Same covariates as your main analysis",
        default=[],
        key="logit_sg_adjust"
    )
    
    st.markdown("---")
    
    # ========== SETTINGS ==========
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_subgroup_n = st.number_input(
                "Minimum N per subgroup:",
                min_value=2, max_value=50, value=5, step=1,
                help="Subgroups with N < this value will be excluded",
                key="logit_sg_min_n"
            )
        
        with col2:
            analysis_title = st.text_input(
                "Custom title (optional):",
                value=f"Subgroup Analysis by {subgroup_col}",
                help="Leave blank for auto title",
                key="logit_sg_title"
            )
    
    st.markdown("---")
    
    # ========== RUN ANALYSIS ==========
    if st.button("🚀 Run Subgroup Analysis", key="logit_subgroup_run", use_container_width=True, type="primary"):
        try:
            # Initialize analyzer
            analyzer = SubgroupAnalysisLogit(df)
            
            # Run analysis with progress tracking
            with st.spinner("🚧 Running analysis..."):
                results = analyzer.analyze(
                    outcome_col=outcome_col,
                    treatment_col=treatment_col,
                    subgroup_col=subgroup_col,
                    adjustment_cols=adjustment_cols if adjustment_cols else None,
                    min_subgroup_n=min_subgroup_n
                )
            
            # Store in session state for persistence
            st.session_state['subgroup_results_logit'] = results
            st.session_state['subgroup_analyzer_logit'] = analyzer
            
            st.success("✅ Analysis complete!")
            
            # ========== RESULTS DISPLAY ==========
            st.markdown("---")
            st.header("📈 Results")
            
            # Forest Plot
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("Forest Plot")
                with col2:
                    if st.button("🗐️ Edit Title", key="edit_forest_title_logit"):
                        st.session_state['edit_forest_title_logit'] = True
                
                if st.session_state.get('edit_forest_title_logit', False):
                    forest_title = st.text_input(
                        "Plot title:",
                        value=analysis_title,
                        key="forest_title_input_logit"
                    )
                else:
                    forest_title = analysis_title
                
                fig = analyzer.create_forest_plot(title=forest_title)
                st.plotly_chart(fig, use_container_width=True, key="logit_forest_plot")
            
            # Summary Statistics
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            summary = results['summary']
            overall = results['overall']
            
            with col1:
                st.metric(
                    label="Overall N",
                    value=f"{summary['n_overall']:,}",
                    delta=f"{overall['events']} events"
                )
            
            with col2:
                st.metric(
                    label="Overall OR",
                    value=f"{overall['or']:.3f}",
                    delta=f"[{overall['ci'][0]:.3f}-{overall['ci'][1]:.3f}]"
                )
            
            with col3:
                st.metric(
                    label="Overall P-value",
                    value=f"{overall['p_value']:.4f}",
                    delta="Significant" if overall['p_value'] < 0.05 else "Not Sig"
                )
            
            with col4:
                p_int = results['interaction']['p_value']
                het_status = "⚠️ Het" if results['interaction']['significant'] else "✅ Hom"
                st.metric(
                    label="P for Interaction",
                    value=f"{p_int:.4f}" if p_int is not None else "N/A",
                    delta=het_status
                )
            
            st.markdown("---")
            
            # Detailed Results Table
            st.subheader("📄 Detailed Results")
            
            results_df = results['results_df'].copy()
            display_cols = ['group', 'n', 'events', 'or', 'ci_low', 'ci_high', 'p_value']
            
            # Format for display
            display_table = results_df[display_cols].copy()
            display_table.columns = ['Group', 'N', 'Events', 'OR', 'CI Lower', 'CI Upper', 'P-value']
            display_table['OR'] = display_table['OR'].apply(lambda x: f"{x:.3f}")
            display_table['CI Lower'] = display_table['CI Lower'].apply(lambda x: f"{x:.3f}")
            display_table['CI Upper'] = display_table['CI Upper'].apply(lambda x: f"{x:.3f}")
            display_table['P-value'] = display_table['P-value'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_table, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Interpretation
            st.subheader("💡 Interpretation")
            
            interpretation = analyzer.get_interpretation()
            if results['interaction']['significant']:
                st.warning(interpretation, icon="⚠️")
            else:
                st.success(interpretation, icon="✅")
            
            # Clinical Guidelines
            with st.expander("📚 Clinical Reporting Guidelines", expanded=False):
                st.markdown(f"""
                ### Subgroup Analysis Reporting (CONSORT, ICMJE)
                
                **Findings:**
                - Overall sample: {summary['n_overall']:,} participants ({overall['events']} events)
                - Number of subgroups: {summary['n_subgroups']}
                - Subgroup variable: {subgroup_col}
                - Effect range: {summary['or_range'][0]:.3f} to {summary['or_range'][1]:.3f}
                
                **Interaction Test:**
                - Test: Wald test of {treatment_col} × {subgroup_col} interaction
                - P-value: {results['interaction']['p_value']:.4f}
                - Conclusion: {"Evidence of significant heterogeneity" if results['interaction']['significant'] else "No significant heterogeneity detected"}
                
                **Recommendations:**
                {"- Report results separately for each subgroup\n- Discuss possible mechanisms for differential effect" if results['interaction']['significant'] else "- Overall estimate is appropriate for all subgroups"}
                """)
            
            st.markdown("---")
            
            # Export Options
            st.subheader("📛 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            # HTML Export
            with col1:
                if analyzer.figure is None:
                    st.warning("Forest plot not available for export")
                else:
                    html_plot = analyzer.figure.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="💿 HTML Plot",
                        data=html_plot,
                        file_name=f"subgroup_{treatment_col}_{subgroup_col}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            
            # CSV Export
            with col2:
                csv_data = display_table.to_csv(index=False)
                st.download_button(
                    label="📋 CSV Results",
                    data=csv_data,
                    file_name=f"subgroup_{treatment_col}_{subgroup_col}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # JSON Export
            with col3:
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📝 JSON Data",
                    data=json_data,
                    file_name=f"subgroup_{treatment_col}_{subgroup_col}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error: {e!s}", icon="💥")
            st.info("**Troubleshooting:**\n- Ensure outcome is binary (2 categories)\n- Check subgroup has 2-10 categories\n- Verify minimum N per subgroup", icon="🗣")
            logger.exception("Logit subgroup analysis error")
    
    # Display previous results if available
    elif 'subgroup_results_logit' in st.session_state and st.session_state.get('show_previous_results', True):
        st.info("📋 Showing previous results. Click 'Run Subgroup Analysis' to refresh.")


def render(df, var_meta):
    """
    Render the "4. Logistic Regression Analysis" section of the Streamlit app.
    
    Creates three subtabs for Binary Logistic Regression, Subgroup Analysis, and Reference & Interpretation,
    and wires UI controls, analysis execution, plotting, and export features for logistic regression workflows.
    
    Parameters:
        df (pandas.DataFrame): The dataset available for analysis (original or matched dataset is selectable via session state).
        var_meta (dict): Variable metadata that guides auto-detection and reporting modes (e.g., categorical vs. linear handling for variables).
    """
    st.subheader("📰 Logistic Regression Analysis")
    
    if st.session_state.get('is_matched', False):
        st.info("✅ **Matched Dataset Available** - You can select it below for analysis")
    
    # Create subtabs
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📈 Binary Logistic Regression",
        "🗒️ Subgroup Analysis",
        "ℹ️ Reference & Interpretation"
    ])
    
    # ==================================================
    # SUB-TAB 1: Binary Logistic Regression
    # ==================================================
    with sub_tab1:
        st.markdown("### Binary Logistic Regression")
        st.info("""
    **💡 Guide:** Models the relationship between predictors and the **probability** of a **binary outcome**.

    * **Odds Ratio (OR/aOR):**
        * **OR > 1:** Increased odds (Risk factor).
        * **OR < 1:** Decreased odds (Protective factor).
    * **Forest Plots:** Both Crude OR and Adjusted OR forest plots are automatically generated.
""")
        
        # Dataset selection
        selected_df, data_label = _get_dataset_for_analysis(df)
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

        # Method Selection
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
            help="Auto selects best method based on data quality."
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
                        
                        # Re-check separation
                        risky_vars_final = check_perfect_separation(final_df, target)
                        if risky_vars_final and algo == 'bfgs':
                            st.warning(f"⚠️ Warning: Perfect separation detected in {risky_vars_final}. Consider using Firth's method.")
                            logger.warning("User selected Standard method with perfect separation")
                        
                        # Run Analysis
                        html, or_results, aor_results = process_data_and_generate_html(final_df, target, var_meta=var_meta, method=algo)
                        st.session_state.html_output_logit = html
                        
                        # 🟢 SHOW NATIVE FOREST PLOT (Interactive)
                        if aor_results or or_results:
                            st.divider()
                            st.subheader("🌳 Forest Plots (Interactive)")
                            
                            fp_tabs = []
                            fp_titles = []
                            
                            if aor_results:
                                fp_titles.append("Adjusted OR (Multivariable)")
                            if or_results: 
                                fp_titles.append("Crude OR (Univariable)")
                            
                            if fp_titles:
                                fp_tabs = st.tabs(fp_titles)
                                
                                # Tab 1: Adjusted OR
                                if aor_results:
                                    with fp_tabs[0]:
                                        # Convert dict to df for library
                                        data_adj = [{'variable': k, **v} for k, v in aor_results.items()]
                                        df_adj = pd.DataFrame(data_adj)
                                        fig_adj = create_forest_plot(
                                            df_adj,
                                            estimate_col='aor', ci_low_col='ci_low', ci_high_col='ci_high', 
                                            pval_col='p_value', label_col='variable',
                                            title="<b>Multivariable Analysis: Adjusted Odds Ratios</b>",
                                            x_label="Adjusted OR",
                                            ref_line=1.0
                                        )
                                        st.plotly_chart(fig_adj, use_container_width=True)
                                
                                # Tab 2: Crude OR (if Adjusted exists, it's index 1, else 0)
                                if or_results:
                                    idx = 1 if aor_results else 0
                                    with fp_tabs[idx]:
                                        data_crude = [{'variable': k, **v} for k, v in or_results.items()]
                                        df_crude = pd.DataFrame(data_crude)
                                        fig_crude = create_forest_plot(
                                            df_crude,
                                            estimate_col='or', ci_low_col='ci_low', ci_high_col='ci_high', 
                                            pval_col='p_value', label_col='variable',
                                            title="<b>Univariable Analysis: Crude Odds Ratios</b>",
                                            x_label="Crude OR",
                                            ref_line=1.0
                                        )
                                        st.plotly_chart(fig_crude, use_container_width=True)
                        
                        st.divider()
                        st.subheader("📋 Detailed Report")
                        st.components.v1.html(html, height=600, scrolling=True)
                        st.success("✅ Analysis complete!")
                        
                        logger.info("✅ Logit analysis completed")
                        
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        logger.exception("Logistic regression failed")
                        
        with dl_col:
            if st.session_state.html_output_logit:
                st.download_button("📥 Download Report", st.session_state.html_output_logit, "logit_report.html", "text/html", key='dl_logit')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_logit')

    # ==================================================
    # SUB-TAB 2: Logistic Regression Subgroup Analysis
    # ==================================================
    with sub_tab2:
        selected_df, _ = _get_dataset_for_analysis(df)
        _render_logit_subgroup_analysis(selected_df)

    # ==================================================
    # SUB-TAB 3: Reference & Interpretation
    # ==================================================
    with sub_tab3:
        st.markdown("##### 📚 Quick Reference: Logistic Regression")
        
        st.info("""
        **🌲 When to Use Logistic Regression:**
        
        | Type | Outcome | Predictors | Example |
        |------|---------|-----------|----------|
        | **Binary** | 2 categories (Yes/No) | Any | Disease/No Disease |
        | **Multinomial** | 3+ unordered categories | Any | Stage (I/II/III/IV) |
        | **Ordinal** | 3+ ordered categories | Any | Severity (Low/Med/High) |
        | **Subgroup Analysis** | Binary + treatment x subgroup | Treatment variable | Drug effectiveness varies by age/sex? |
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
            
            **🌳 Forest Plots**
            - Visual representation of OR/aOR
            - Included in downloadable HTML report
            - Interactive charts with CI error bars
            - Log scale for easy interpretation
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
            
            ### Common Mistakes ❌
            
            - **Unadjusted OR** without adjustment → Use aOR ✅
            - **Perfect separation** (category = outcome) → Exclude or use Firth
            - **Ignoring CI** (only p-value) → CI shows range
            - **Multicollinearity** (correlated predictors) → Check correlations
            - **Overfitting** (too many variables) → Use variable selection
            - **Log-transformed interpreters** → Multiply by e^(unit change)
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 💡 Two OR Modes: When to Use Each
        
        The system supports **2 different modes** for handling categorical/continuous variables:
        """)
        
        # 🟢 IMPROVED: Mode guide with examples (Removed Simple Mode)
        tab_cat, tab_linear = st.tabs([
            "📊 Categorical (All Levels)",
            "📉 Linear (Trend)"
        ])
        
        with tab_cat:
            st.markdown("""
            #### 📊 Categorical Mode: All Levels vs Reference
            
            **When to Use:**
            - Variable has multiple discrete categories
            - All level comparisons are meaningful
            - Example: Stage (I, II, III, IV)
            
            **What You Get:**
            - Separate OR for each level compared to Reference
            - Ref vs Level 1, Ref vs Level 2, Ref vs Level 3...
            - **Output: Multiple lines** (one per level)
            
            **Example:**
            ```
            Stage (Reference = I):
            - Ref.
            - Level II vs I: OR = 1.8 (95% CI: 1.2-2.4)
            - Level III vs I: OR = 2.5 (95% CI: 1.6-3.2)
            - Level IV vs I: OR = 3.2 (95% CI: 2.0-4.1)
            ```
            
            **When NOT to use:**
            - Continuous variable (consider Linear mode)
            
            **How to Specify:**
            ```python
            var_meta = {
                'stage': {'type': 'Categorical'}  # All levels
            }
            ```
            """)
        
        with tab_linear:
            st.markdown("""
            ### 📉 Linear Mode: Per-Unit Trend
            
            **When to Use:**
            - Continuous or quasi-continuous variables
            - Interested in per-unit increase effect
            - Example: Age (years), BMI (kg/m²), Blood Pressure (mmHg)
            
            **What You Get:**
            - Single OR per 1-unit increase
            - **Output: Single line**
            - Assumes linear dose-response relationship
            
            **Example:**
            ```
            Age (years):
            - Per 1-year increase: OR = 1.02 (95% CI: 1.01-1.03)
            
            Interpretation: Each additional year of age increases 
            odds of outcome by 2% (assuming linear relationship)
            ```
            
            **Interpretation Tips:**
            - For per-SD increase:
              ```
              If age SD = 10 years:
              aOR per 10-year = 1.02^10 = 1.22
              → Per SD increase = 22% higher odds
              ```
            
            **When NOT to use:**
            - Non-linear relationship (e.g., U-shaped)
            - Sparse data in outer ranges
            - Categorical with few distinct levels (use Categorical)
            
            **How to Specify:**
            ```python
            var_meta = {
                'age': {'type': 'Linear'},       # Continuous
                'bmi': {'type': 'Linear'}       # Continuous
            }
            ```
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🛠️ Auto-Detection Logic
        
        If you **don't specify** a mode in `var_meta`, the system auto-detects:
        
        1. **Binary (0/1)?** → Categorical
        2. **Few levels (<10) with mostly integers?** → Categorical
        3. **Otherwise** → Linear
        
        **Example Auto-Detection:**
        - `stage` = [1, 2, 3, 4] → Categorical 📊
        - `age` = [23.4, 45.2, 67.8...] → Linear 📉
        - `location` = [0, 1] → Categorical 📊
        - `treatment` = [0, 1] → Categorical 📊
        """)
        
        st.markdown("---")
        
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
        - ❌ Cannot estimate coefficients reliably
        - ❌ Returns infinite or missing values
        - ❌ Model doesn't converge
        - ❌ P-values are undefined
        - ❌ Results are invalid
        
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
        - ❌ May not converge
        - ❌ Infinite coefficients
        - ❌ Missing p-values
        - ❌ Invalid results
        - ❌ **DO NOT USE with perfect separation!**
        
        **Best Practice Summary:**
        1. Load your data
        2. Select "Auto (Recommended)" method
        3. Click "Run Logistic Regression"
        4. Download HTML report to view forest plots
        5. Done! App handles everything automatically
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 📄 Subgroup Analysis
        
        **When to Use:**
        - Testing for treatment x subgroup interactions
        - Examining differential treatment effects
        - Identifying patient populations with greater benefit
        
        **Key Concepts:**
        - **Homogeneous effect** → One OR applies to all (no interaction)
        - **Heterogeneous effect** → Different OR by subgroup (interaction exists)
        - **Interaction p-value** → p < 0.05 = significant heterogeneity
        
        **Interpretation:**
        - If p_interaction < 0.05: Report results separately by subgroup
        - If p_interaction ≥ 0.05: Overall estimate applies to all
        
        **Forest Plot in Subgroup Tab:**
        - Shows OR with 95% CI by subgroup
        - Horizontal line at OR=1 (null effect)
        - Interactive plot for data exploration
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Interpretation Example
        
        **Model Output:**
        - Variable: Smoking
        - aOR = 1.8 (95% CI: 1.2 - 2.4)
        - p = 0.003
        
        **Interpretation:** Smoking is associated with 1.8× increased odds of outcome (compared to non-smoking), adjusting for other variables. This difference is statistically significant (p < 0.05), and we're 95% confident the true OR is between 1.2 and 2.4. ✅
        
        ---
        
        ### 💾 Future Expansions
        
        Planned additions to this tab:
        - **Multinomial Logistic Regression** (3+ unordered outcomes)
        - **Ordinal Logistic Regression** (3+ ordered outcomes)
        - **Mixed Effects Logistic** (clustered/repeated data)
        """)