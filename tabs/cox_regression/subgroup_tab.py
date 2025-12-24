"""
Subgroup Analysis SubTab for Cox Regression (Survival Analysis)

Provides interactive interface for publication-grade subgroup analysis.
Integrated into Cox regression workflow.

Author: NTWKKM
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from subgroup_analysis_module import SubgroupAnalysisCox
from logger import get_logger

logger = get_logger(__name__)


def render(df: pd.DataFrame, time_col: str = None, event_col: str = None, treatment_var: str = None):
    """
    Render Subgroup Analysis SubTab for Cox Regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data frame
    time_col : str, optional
        Pre-selected time/duration variable
    event_col : str, optional
        Pre-selected event indicator variable
    treatment_var : str, optional
        Pre-selected treatment variable
    """
    st.markdown("---")
    st.header("🗒️ Subgroup Analysis (Survival)")
    
    # Info box
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.info("🚀", help="Click for more information")
        with col2:
            st.markdown("""
            **Test for Treatment-by-Subgroup Interaction in Survival Analysis**
            
            Determine if treatment effect on survival varies by patient characteristics.
            🙋: Does the drug prolong survival differently in men vs women? Young vs old?
            """)
    
    st.markdown("---")
    
    # ========== INPUT SECTION ==========
    st.subheader("📝 Step 1: Select Variables")
    
    col1, col2, col3 = st.columns(3)
    
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    
    # Time variable
    with col1:
        time_col_selected = st.selectbox(
            "Follow-up Time",
            options=numeric_cols,
            index=0 if time_col is None else numeric_cols.index(time_col) if time_col in numeric_cols else 0,
            help="Duration to event or censoring (days, months, years)"
        )
    
    # Event variable
    with col2:
        event_col_selected = st.selectbox(
            "Event Indicator (Binary)",
            options=binary_cols,
            index=0 if event_col is None else binary_cols.index(event_col) if event_col in binary_cols else 0,
            help="1/Yes = event occurred, 0/No = censored"
        )
    
    # Treatment variable
    with col3:
        treatment_col_selected = st.selectbox(
            "Treatment/Exposure",
            options=[col for col in df.columns if col not in [time_col_selected, event_col_selected]],
            index=0 if treatment_var is None else [col for col in df.columns if col not in [time_col_selected, event_col_selected]].index(treatment_var) if treatment_var in [col for col in df.columns if col not in [time_col_selected, event_col_selected]] else 0,
            help="Main variable of interest"
        )
    
    st.markdown("---")
    
    # ========== SUBGROUP & ADJUSTMENT ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Step 2: Subgroup Variable")
        subgroup_col_selected = st.selectbox(
            "Stratify By:",
            options=[col for col in df.columns 
                    if col not in [time_col_selected, event_col_selected, treatment_col_selected]
                    and df[col].nunique() >= 2 
                    and df[col].nunique() <= 10],
            help="Categorical variable with 2-10 categories"
        )
    
    with col2:
        st.subheader("📌 Step 3: Adjustment Variables")
        adjustment_cols_selected = st.multiselect(
            "Select covariates:",
            options=[col for col in df.columns 
                    if col not in [time_col_selected, event_col_selected, treatment_col_selected, subgroup_col_selected]],
            default=[]
        )
    
    st.markdown("---")
    
    # ========== SETTINGS ==========
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_subgroup_n = st.number_input(
                "Min N per subgroup:",
                min_value=2, max_value=50, value=5,
                help="Subgroups with N < this will be excluded"
            )
        
        with col2:
            min_events = st.number_input(
                "Min events per subgroup:",
                min_value=1, max_value=50, value=2,
                help="Subgroups with < this many events excluded"
            )
        
        with col3:
            analysis_title = st.text_input(
                "Custom title:",
                value=f"Survival Subgroup Analysis by {subgroup_col_selected}",
                help="Leave blank for auto title"
            )
    
    st.markdown("---")
    
    # ========== RUN ANALYSIS ==========
    if st.button("🚀 Run Subgroup Analysis", key="cox_subgroup_run", use_container_width=True, type="primary"):
        try:
            # Check lifelines availability
            try:
                from lifelines import CoxPHFitter
            except ImportError:
                st.error("❌ Lifelines library required for Cox analysis")
                st.info("Install: `pip install lifelines`")
                return
            
            # Initialize analyzer
            analyzer = SubgroupAnalysisCox(df)
            
            # Run analysis with progress tracking
            with st.spinner("🧰 Running Cox subgroup analysis..."):
                results = analyzer.analyze(
                    time_col=time_col_selected,
                    event_col=event_col_selected,
                    treatment_col=treatment_col_selected,
                    subgroup_col=subgroup_col_selected,
                    adjustment_cols=adjustment_cols_selected if adjustment_cols_selected else None,
                    min_subgroup_n=min_subgroup_n,
                    min_events=min_events
                )
            
            # Store in session state
            st.session_state['subgroup_results_cox'] = results
            st.session_state['subgroup_analyzer_cox'] = analyzer
            
            st.success("✅ Cox subgroup analysis complete!")
            
            # ========== RESULTS DISPLAY ==========
            st.markdown("---")
            st.header("📈 Results")
            
            # Forest Plot
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("Forest Plot - Hazard Ratios")
                with col2:
                    if st.button("🗐️ Edit", key="edit_forest_title_cox"):
                        st.session_state['edit_forest_title_cox'] = True
                
                if st.session_state.get('edit_forest_title_cox', False):
                    forest_title = st.text_input(
                        "Plot title:",
                        value=analysis_title,
                        key="forest_title_input_cox"
                    )
                else:
                    forest_title = analysis_title
                
                fig = analyzer.create_forest_plot(title=forest_title)
                st.plotly_chart(fig, use_container_width=True, key="cox_forest_plot")
            
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
                    label="Overall HR",
                    value=f"{overall['hr']:.3f}",
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
            display_cols = ['group', 'n', 'events', 'hr', 'ci_low', 'ci_high', 'p_value']
            
            # Format for display
            display_table = results_df[display_cols].copy()
            display_table.columns = ['Group', 'N', 'Events', 'HR', 'CI Lower', 'CI Upper', 'P-value']
            display_table['HR'] = display_table['HR'].apply(lambda x: f"{x:.3f}")
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
            with st.expander("📚 Clinical Reporting Guidelines (CONSORT Extension)", expanded=False):
                st.markdown(f"""
                ### Subgroup Analysis in Survival Studies
                
                **Study Population:**
                - Total sample: {summary['n_overall']:,} participants
                - Total events: {overall['events']}
                - Follow-up variable: {time_col_selected}
                - Event variable: {event_col_selected}
                
                **Subgroup Analysis:**
                - Stratification variable: {subgroup_col_selected}
                - Number of subgroups: {summary['n_subgroups']}
                - HR range: {summary['hr_range'][0]:.3f} to {summary['hr_range'][1]:.3f}
                
                **Interaction Test:**
                - Method: Wald test of {treatment_col_selected} × {subgroup_col_selected} interaction
                - P-value: {results['interaction']['p_value']:.4f}
                - Result: {"Evidence of significant heterogeneity" if results['interaction']['significant'] else "No significant heterogeneity"}
                
                **Reporting Recommendations:**
                {"- Report Kaplan-Meier curves by subgroup\n- Discuss differential survival benefits\n- Consider stratified analyses in future trials" if results['interaction']['significant'] else "- Overall HR applies to all subgroups\n- No need for separate reporting by subgroup"}
                """)
            
            st.markdown("---")
            
            # Export Options
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            # HTML Export
            with col1:
                html_plot = analyzer.figure.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="📿 HTML Plot",
                    data=html_plot,
                    file_name=f"subgroup_cox_{treatment_col_selected}_{subgroup_col_selected}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            # CSV Export
            with col2:
                csv_data = display_table.to_csv(index=False)
                st.download_button(
                    label="📋 CSV Results",
                    data=csv_data,
                    file_name=f"subgroup_cox_{treatment_col_selected}_{subgroup_col_selected}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # JSON Export
            with col3:
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📝 JSON Data",
                    data=json_data,
                    file_name=f"subgroup_cox_{treatment_col_selected}_{subgroup_col_selected}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}", icon="💥")
            st.info("""
            **Troubleshooting:**
            - Time variable must be numeric and > 0
            - Event must be binary (0/1)
            - Subgroup must have 2-10 categories
            - Minimum 5 observations per subgroup
            - Minimum 2 events per subgroup
            """, icon="💭")
            logger.error(f"Cox subgroup analysis error: {e}")
    
    # Display previous results if available
    elif 'subgroup_results_cox' in st.session_state and st.session_state.get('show_previous_results_cox', True):
        st.info("💻 Showing previous results. Click 'Run Subgroup Analysis' to refresh.")
