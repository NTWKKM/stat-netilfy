"""
Subgroup Analysis SubTab for Logistic Regression

Provides interactive interface for publication-grade subgroup analysis.
Integrated into logistic regression workflow.

Author: NTWKKM
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from subgroup_analysis_module import SubgroupAnalysisLogit
from logger import get_logger

logger = get_logger(__name__)


def render(df: pd.DataFrame, outcome_var: str | None = None, treatment_var: str | None = None):
    """
    Render the Streamlit Subgroup Analysis subtab for logistic regression.
    
    Displays a user interface to select a binary outcome, a treatment/exposure, a categorical subgroup (2–10 levels), optional adjustment covariates, and advanced settings; runs a subgroup logistic regression via SubgroupAnalysisLogit when requested and renders results (forest plot, summary statistics, detailed table, interpretation, reporting guidance, and export options). Successful analysis objects are saved to st.session_state under 'subgroup_results_logit' and 'subgroup_analyzer_logit'.
    
    Parameters:
        df (pd.DataFrame): Dataset used to populate selection widgets and run the analysis.
        outcome_var (str | None): Optional pre-selected binary outcome column name; applied if present in df and binary.
        treatment_var (str | None): Optional pre-selected treatment/exposure column name; applied if present in df.
    """
    st.markdown("---")
    st.header("🗒️ Subgroup Analysis")
    
    # Info box
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.info("🚀", help="Click for more information")
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

    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    if not binary_cols:
       st.error("No binary columns found for outcome.")
       return
    default_idx = 0
    if outcome_var and outcome_var in binary_cols:
        default_idx = binary_cols.index(outcome_var)
        
    # Outcome variable
    with col1:
        outcome_col = st.selectbox(
            "Outcome (Binary)",
            options=binary_cols,
            index=default_idx,
            help="Select binary outcome variable (0/1 or No/Yes)"
        )
    
    # Treatment variable
    with col2:
        treatment_col = st.selectbox(
            "Treatment/Exposure",
            options=[col for col in df.columns if col != outcome_col],
            index=0 if treatment_var is None else [col for col in df.columns if col != outcome_col].index(treatment_var) if treatment_var in [col for col in df.columns if col != outcome_col] else 0,
            help="Main variable of interest"
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
            help="Categorical variable with 2-10 categories"
        )
    
    st.markdown("---")
    
    # ========== ADJUSTMENT VARIABLES ==========
    st.subheader("📌 Step 2: Adjustment Variables (Optional)")
    
    adjustment_cols = st.multiselect(
        "Select covariates to adjust for:",
        options=[col for col in df.columns 
                if col not in [outcome_col, treatment_col, subgroup_col]],
        help="Same covariates as your main analysis",
        default=[]
    )
    
    st.markdown("---")
    
    # ========== SETTINGS ==========
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_subgroup_n = st.number_input(
                "Minimum N per subgroup:",
                min_value=2, max_value=50, value=5, step=1,
                help="Subgroups with N < this value will be excluded"
            )
        
        with col2:
            analysis_title = st.text_input(
                "Custom title (optional):",
                value=f"Subgroup Analysis by {subgroup_col}",
                help="Leave blank for auto title"
            )
    
    st.markdown("---")
    
    # ========== RUN ANALYSIS ==========
    if st.button("🚀 Run Subgroup Analysis", key="logit_subgroup_run", use_container_width=True, type="primary"):
        try:
            # Initialize analyzer
            analyzer = SubgroupAnalysisLogit(df)
            
            # Run analysis with progress tracking
            with st.spinner("🧰 Running analysis..."):
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
                - Test: Wald test of {treatment_col} x {subgroup_col} interaction
                - P-value: {results['interaction']['p_value']:.4f}
                - Conclusion: {"Evidence of significant heterogeneity" if results['interaction']['significant'] else "No significant heterogeneity detected"}
                
                **Recommendations:**
                {"- Report results separately for each subgroup\n- Discuss possible mechanisms for differential effect" if results['interaction']['significant'] else "- Overall estimate is appropriate for all subgroups"}
                """)
            
            st.markdown("---")
            
            # Export Options
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            # HTML Export
            with col1:
                if analyzer.figure is None:
                    st.warning("Forest plot not available for export")
                else:
                    html_plot = analyzer.figure.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="📿 HTML Plot",
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
            st.info("**Troubleshooting:**\n- Ensure outcome is binary (2 categories)\n- Check subgroup has 2-10 categories\n- Verify minimum N per subgroup", icon="💭")
            logger.exception("Logit subgroup analysis error")
    
    # Display previous results if available
    elif 'subgroup_results_logit' in st.session_state and st.session_state.get('show_previous_results', True):
        st.info("💻 Showing previous results. Click 'Run Subgroup Analysis' to refresh.")
        # Display logic would go here