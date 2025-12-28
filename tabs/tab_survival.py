import streamlit as st
import pandas as pd
import numpy as np
import survival_lib
import time
from pandas.api.types import is_numeric_dtype
import logging
import json
from logger import get_logger

# ✅ FIX IMPORT: Use the unified module from root
from subgroup_analysis_module import SubgroupAnalysisCox 

logger = get_logger(__name__)

# 🟢 NEW: Helper function to select between original and matched datasets
def _get_dataset_for_survival(df: pd.DataFrame):
    """
    Choose between the original DataFrame and an available propensity-score matched DataFrame for survival analysis.
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
                index=1,  # default Matched สำหรับ survival analysis
                horizontal=True,
                key="survival_data_source",
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


def _render_cox_subgroup_analysis(df: pd.DataFrame) -> None:
    """
    Render the Cox proportional hazards subgroup analysis UI and execute the analysis workflow.
    """
    st.header("🗒️ Subgroup Analysis (Survival)")
    
    # Info box
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.info("🚀")
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

    if not numeric_cols:
       st.error("No numeric columns found for follow-up time.")
       return
    if not binary_cols:
       st.error("No binary columns found for event indicator.")
       return
       
    # Time variable
    with col1:
        time_col_selected = st.selectbox(
            "Follow-up Time",
            options=numeric_cols,
            index=0,
            help="Duration to event or censoring (days, months, years)",
            key="cox_sg_time"
        )
    
    # Event variable
    with col2:
        event_col_selected = st.selectbox(
            "Event Indicator (Binary)",
            options=binary_cols,
            index=0,
            help="1/Yes = event occurred, 0/No = censored",
            key="cox_sg_event"
        )
    
    # Treatment variable
    with col3:
        treatment_col_selected = st.selectbox(
            "Treatment/Exposure",
            options=[col for col in df.columns if col not in [time_col_selected, event_col_selected]],
            index=0,
            help="Main variable of interest",
            key="cox_sg_treatment"
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
            help="Categorical variable with 2-10 categories",
            key="cox_sg_subgroup"
        )
    
    with col2:
        st.subheader("📌 Step 3: Adjustment Variables")
        adjustment_cols_selected = st.multiselect(
            "Select covariates:",
            options=[col for col in df.columns 
                    if col not in [time_col_selected, event_col_selected, treatment_col_selected, subgroup_col_selected]],
            default=[],
            key="cox_sg_adjust"
        )
    
    st.markdown("---")
    
    # ========== SETTINGS ==========
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_subgroup_n = st.number_input(
                "Min N per subgroup:",
                min_value=2, max_value=50, value=5,
                help="Subgroups with N < this will be excluded",
                key="cox_sg_min_n"
            )
        
        with col2:
            min_events = st.number_input(
                "Min events per subgroup:",
                min_value=1, max_value=50, value=2,
                help="Subgroups with < this many events excluded",
                key="cox_sg_min_events"
            )
        
        with col3:
            analysis_title = st.text_input(
                "Custom title:",
                value=f"Survival Subgroup Analysis by {subgroup_col_selected}",
                help="Leave blank for auto title",
                key="cox_sg_title"
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
                if results['interaction']['significant']:
                    result_text = "Evidence of significant heterogeneity"
                    rec_text = "- Report Kaplan-Meier curves by subgroup\n- Discuss differential survival benefits\n- Consider stratified analyses in future trials"
                else:
                    result_text = "No significant heterogeneity"
                    rec_text = "- Overall HR applies to all subgroups\n- No need for separate reporting by subgroup"

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
                - Result: {result_text}
                
                **Reporting Recommendations:**
                {rec_text}
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
            st.error(f"❌ Error: {e!s}", icon="💥")
            st.info("""
            **Troubleshooting:**
            - Time variable must be numeric and > 0
            - Event must be binary (0/1)
            - Subgroup must have 2-10 categories
            - Minimum 5 observations per subgroup
            - Minimum 2 events per subgroup
            """, icon="💭")
            logger.exception("Cox subgroup analysis error")
    
    # Display previous results if available (separate check)
    else:
        if 'subgroup_results_cox' in st.session_state and st.session_state.get('show_previous_results_cox', True):
            st.info("💻 Showing previous results. Click 'Run Subgroup Analysis' to refresh.")


def render(df, _var_meta):
    """
    Render the Streamlit UI and orchestrate interactive survival analysis workflows (Kaplan–Meier / Nelson–Aalen curves, landmark analysis, Cox regression with assumption checks and forest plots, and Cox subgroup analysis) for a chosen dataset.
    """
    st.subheader("⏳ Survival Analysis")
    
    # 🟢 NEW: Display matched data status if available
    if st.session_state.get("is_matched", False):
        st.info("✅ **Matched Dataset Available** - You can select it below for analysis")
    
    surv_df, surv_label = _get_dataset_for_survival(df)
    st.write(f"**Using:** {surv_label}")
    st.write(f"**Rows:** {len(surv_df)} | **Columns:** {len(surv_df.columns)}")
    
    all_cols = surv_df.columns.tolist()
    
    if len(all_cols) < 2:
        st.error("Dataset must contain at least 2 columns (time and event).")
        return
        
    # Global Selectors
    c1, c2 = st.columns(2)
    
    # Auto-detect logic
    time_idx = 0
    for k in ['stop', 'time', 'dur']:
        found = next((i for i, c in enumerate(all_cols) if k in c.lower()), None)
        if found is not None:
            time_idx = found
            break
            
    event_idx = next((i for i, c in enumerate(all_cols) if 'event' in c.lower() or 'status' in c.lower() or 'dead' in c.lower()), min(1, len(all_cols)-1))
    
    col_time = c1.selectbox("⏳ Time Variable:", all_cols, index=time_idx, key='surv_time')
    col_event = c2.selectbox("🗣️ Event Variable (1=Event):", all_cols, index=event_idx, key='surv_event')
    
    # Tabs
    tab_curves, tab_landmark, tab_cox, tab_cox_sg, tab_ref = st.tabs(["📈 Survival Curves (KM/NA)", "📋 Landmark Analysis", "📊 Cox Regression", "🗒️ Subgroup Analysis", "ℹ️ Reference & Interpretation"])
    
    # ==========================
    # TAB 1: Curves (KM & Nelson-Aalen)
    # ==========================
    with tab_curves:
        c1, c2 = st.columns([1, 2])
        col_group = c1.selectbox("Compare Groups (Optional):", ["None", *all_cols], key='surv_group')
        plot_type = c2.radio("Select Plot Type:", ["Kaplan-Meier (Survival Function)", "Nelson-Aalen (Cumulative Hazard)"], horizontal=True)
        
        # 🟢 Layout: Buttons side-by-side
        col_btn1, col_btn2 = st.columns([2, 4])
        with col_btn1:
            run_curves = st.button("🚀 Generate Survival Curve", type="primary", key='btn_run_curves', use_container_width=True)
        
        # Initialize session state for this tab
        if 'surv_curve_res' not in st.session_state:
            st.session_state['surv_curve_res'] = None

        if run_curves:
            grp = None if col_group == "None" else col_group
            try:
                if "Kaplan-Meier" in plot_type:
                    # Run KM
                    fig, stats_df = survival_lib.fit_km_logrank(surv_df, col_time, col_event, grp)
                    
                    elements = [{'type':'header','data':'Kaplan-Meier'}, {'type':'plot','data':fig}, {'type':'table','data':stats_df}]
                    report_html = survival_lib.generate_report_survival(f"KM: {col_time}", elements)
                    
                    st.session_state['surv_curve_res'] = {
                        'fig': fig,
                        'stats': stats_df,
                        'html': report_html,
                        'type': 'KM'
                    }
                    
                else:
                    # Run Nelson-Aalen
                    fig, stats_df = survival_lib.fit_nelson_aalen(surv_df, col_time, col_event, grp)
                    
                    elements = [
                        {'type':'header','data':'Nelson-Aalen Cumulative Hazard'}, 
                        {'type':'plot','data':fig},
                        {'type':'header','data':'Summary Statistics'},
                        {'type':'table','data':stats_df}
                    ]
                    report_html = survival_lib.generate_report_survival(f"NA: {col_time}", elements)

                    st.session_state['surv_curve_res'] = {
                        'fig': fig,
                        'stats': stats_df,
                        'html': report_html,
                        'type': 'NA'
                    }
            
            except Exception as e:
                logger.exception("Unexpected error in survival curves analysis")
                st.error(f"Error: {e}")
                st.session_state['surv_curve_res'] = None

        # Render Results & Download Button
        if st.session_state['surv_curve_res']:
            res = st.session_state['surv_curve_res']
            
            # Show download button if results exist (in the same row)
            with col_btn2:
                file_name = "km_report.html" if res['type'] == 'KM' else "na_report.html"
                st.download_button("📥 Download Report", res['html'], file_name, "text/html")
            
            st.plotly_chart(res['fig'], use_container_width=True)
            
            st.markdown("##### Statistics")
            st.dataframe(res['stats'])
            if res['type'] == 'NA':
                 st.caption("Note: Nelson-Aalen estimates the cumulative hazard rate function (H(t)).")


    # ==========================
    # TAB 2: Landmark Analysis
    # ==========================
    with tab_landmark:    
        st.caption("Principle: Exclude patients who had an event or were censored before the Landmark Time.")
        
        max_t = surv_df[col_time].dropna().max() if not surv_df.empty and is_numeric_dtype(surv_df[col_time]) and surv_df[col_time].notna().any() else 1.0
        if max_t <= 0:
            max_t = 1.0 
        
        st.write(f"**Select Landmark Time (Max: {max_t:.2f})**")
        
        if 'landmark_val' not in st.session_state:
            st.session_state.landmark_val = float(round(float(max_t) * 0.1))

        def update_from_slider() -> None:
            st.session_state.landmark_val = st.session_state.lm_slider_widget
        
        def update_from_number() -> None:
            st.session_state.landmark_val = st.session_state.lm_number_widget

        c_slide, c_num = st.columns([3, 1])
        with c_slide:
            st.slider("Use Slider:", min_value=0.0, max_value=float(max_t) * 0.99, key='lm_slider_widget', value=min(st.session_state.landmark_val, float(max_t) * 0.99), on_change=update_from_slider, label_visibility="collapsed")
        with c_num:
            st.number_input("Enter Value:", min_value=0.0, max_value=float(max_t) * 0.99, key='lm_number_widget', value=min(st.session_state.landmark_val, float(max_t) * 0.99), on_change=update_from_number, step=1.0, label_visibility="collapsed")
            
        landmark_t = st.session_state.landmark_val
        
        group_idx = 0
        available_cols = [c for c in all_cols if c not in [col_time, col_event]]

        if not available_cols:
            st.warning("Landmark analysis requires at least one group/covariate column beyond time and event.")
        else:
            for priority_key in ['group', 'treatment', 'comorbid']:
                found_idx = next((i for i, c in enumerate(available_cols) if priority_key in c.lower()), None)
                if found_idx is not None:
                    group_idx = found_idx
                    break
            
            col_group = st.selectbox("Compare Group:", available_cols, index=group_idx, key='lm_group_sur')

            # 🟢 Layout: Buttons side-by-side
            col_lm_btn1, col_lm_btn2 = st.columns([1, 4])
            with col_lm_btn1:
                run_landmark = st.button("🚀 Run Landmark Analysis", type="primary", key='btn_lm_sur', use_container_width=True)

            # Initialize session state for this tab
            if 'surv_lm_res' not in st.session_state:
                st.session_state['surv_lm_res'] = None

            if run_landmark:
                try:
                    with st.spinner(f"Running Landmark Analysis at t={landmark_t:.2f}..."):
                        fig, stats, n_pre, n_post, err = survival_lib.fit_km_landmark(
                            surv_df, col_time, col_event, col_group, landmark_t
                        )
                        
                        if err:
                            st.error(err)
                            st.session_state['surv_lm_res'] = None
                        elif fig:
                            elements = [
                                {'type':'header','data':f'Landmark Analysis (Survival from t={landmark_t:.2f})'},
                                {'type':'text', 'data': f"N Included: {n_post}, N Excluded: {n_pre - n_post}"},
                                {'type':'plot','data':fig},
                                {'type':'table','data':stats}
                            ]
                            report_html = survival_lib.generate_report_survival(f"Landmark Analysis: {col_time} (t >= {landmark_t})", elements)
                            
                            st.session_state['surv_lm_res'] = {
                                'fig': fig,
                                'stats': stats,
                                'html': report_html,
                                'n_pre': n_pre,
                                'n_post': n_post,
                                't': landmark_t
                            }
                except (ValueError, KeyError) as e:
                    st.error(f"Analysis error: {e}")
                    st.session_state['surv_lm_res'] = None
                except Exception as e:
                    logger.exception("Unexpected error in landmark analysis")
                    st.error(f"Unexpected error: {e}")
                    st.session_state['surv_lm_res'] = None

            # Render Results & Download Button
            if st.session_state['surv_lm_res']:
                res = st.session_state['surv_lm_res']
                
                with col_lm_btn2:
                    st.download_button("📥 Download Report", res['html'], "lm_report.html", "text/html")
                
                st.markdown(f"""
                <p style='font-size:1em;'>
                Total N before filter: <b>{res['n_pre']}</b> | 
                N Included (Survived $\ge$ {res['t']:.2f}): <b>{res['n_post']}</b> | 
                N Excluded: <b>{res['n_pre'] - res['n_post']}</b>
                </p>
                """, unsafe_allow_html=True)
                    
                st.plotly_chart(res['fig'], use_container_width=True)
                st.markdown("##### Log-Rank Test Results (Post-Landmark)")
                st.dataframe(res['stats'])


    # ==========================
    # TAB 3: Cox Regression
    # ==========================
    with tab_cox:
        covariates = st.multiselect("Select Covariates (Predictors):", [c for c in all_cols if c not in [col_time, col_event]], key='surv_cox_vars')
        
        if 'cox_res' not in st.session_state: 
            st.session_state.cox_res = None
        if 'cox_html' not in st.session_state: 
            st.session_state.cox_html = None
        if 'cox_fig_objects' not in st.session_state:
            st.session_state.cox_fig_objects = None
        if 'cox_txt_report' not in st.session_state:
            st.session_state.cox_txt_report = None
        
        # 🟢 Layout: Buttons side-by-side
        col_cox_btn1, col_cox_btn2 = st.columns([1, 4])
        with col_cox_btn1:
            run_cox = st.button("🚀 Run Cox Model", type="primary", key='btn_run_cox', use_container_width=True)

        if run_cox:
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                try:
                    with st.spinner("Fitting Cox Model and Checking Assumptions..."):
                        cph, res, model_data, err = survival_lib.fit_cox_ph(surv_df, col_time, col_event, covariates)
                        
                        if err:
                            st.error(f"Error: {err}")
                            st.session_state.cox_res = None
                            st.session_state.cox_html = None
                        else:
                            # Receive list of figures
                            txt_report, fig_objects = survival_lib.check_cph_assumptions(cph, model_data)
                            
                            st.session_state.cox_res = res
                            st.session_state.cox_txt_report = txt_report
                            st.session_state.cox_fig_objects = fig_objects
                            
                            # Generate HTML report immediately
                            elements = [
                                {'type':'header','data':'Cox Proportional Hazards'},
                                {'type':'table','data':res},
                                {'type':'header','data':'Assumption Check (Schoenfeld Residuals)'},
                                {'type':'preformatted','data':txt_report} 
                            ]
                            if fig_objects:
                                for fig in fig_objects:
                                    elements.append({'type':'plot','data':fig})
                            
                            forest_plot_html = survival_lib.generate_forest_plot_cox_html(res)
                            elements.append({'type':'html','data':forest_plot_html})
                            
                            report_html = survival_lib.generate_report_survival(f"Cox: {col_time}", elements)
                            st.session_state.cox_html = report_html
                            
                            st.success("✅ Analysis Complete!")

                except (ValueError, KeyError, RuntimeError) as e:
                    st.error(f"Analysis error: {e}")
                    st.session_state.cox_res = None
                except Exception as e:
                    logger.exception("Unexpected error in Cox regression analysis")
                    st.error(f"Unexpected error: {e}")
                    st.session_state.cox_res = None

        # Render Results & Download Button
        if st.session_state.cox_html:
             with col_cox_btn2:
                st.download_button("📥 Download Report", st.session_state.cox_html, "cox_report.html", "text/html")

        if st.session_state.cox_res is not None:
            res = st.session_state.cox_res
            # Display results
            format_dict = {
                'HR': '{:.4f}',
                '95% CI Lower': '{:.4f}',
                '95% CI Upper': '{:.4f}',
                'P-value': '{:.4f}'
            }
            st.dataframe(res.style.format(format_dict))
            
            if 'Method' in res.columns and len(res) > 0:
                st.caption(f"Method Used: {res['Method'].iloc[0]}")
            
            st.markdown("##### 🔍 Proportional Hazards Assumption Check")
            
            if st.session_state.cox_txt_report:
                with st.expander("View Assumption Advice (Text)", expanded=False):
                    st.text(st.session_state.cox_txt_report)
            
            if st.session_state.cox_fig_objects:
                st.write("**Schoenfeld Residuals Plots:**")
                for fig in st.session_state.cox_fig_objects:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No assumption plots generated.")
            
            st.markdown("---")
            st.subheader("🌳 Forest Plot: Hazard Ratios")
            try:
                fig_forest = survival_lib.create_forest_plot_cox(res)
                st.plotly_chart(fig_forest, use_container_width=True)
                
                with st.expander("📄 View Raw Data Table"):
                    st.dataframe(res[['HR', '95% CI Lower', '95% CI Upper', 'P-value']].reset_index())
                
            except Exception as e:
                logger.warning("Forest plot generation failed: %s", e)
                st.warning(f"Could not generate HR forest plot: {e}")


    # ==========================
    # TAB 4: Cox Subgroup Analysis
    # ==========================
    with tab_cox_sg:
        _render_cox_subgroup_analysis(surv_df)

    # ==========================
    # TAB 5: Reference & Interpretation
    # ==========================
    with tab_ref:
        st.markdown("##### 📚 Quick Reference: Survival Analysis")
        
        st.info("""
        **🎲 When to Use What:**
        
        | Method | Purpose | Output |
        |--------|---------|--------|
        | **KM Curves** | Visualize time-to-event by group | Survival %, median, p-value |
        | **Nelson-Aalen** | Cumulative hazard over time | H(t) curve, risk accumulation |
        | **Landmark** | Late/surrogate endpoints | Filtered KM, immortal time removed |
        | **Cox** | Multiple predictors of survival | HR, CI, p-value per variable + **forest plot** ✨ |
        | **Subgroup Analysis** | Treatment effect heterogeneity | HR by subgroup, interaction test |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Kaplan-Meier (KM) Curves")
            st.markdown("""
            **When to Use:**
            - Time-to-event analysis (survival, recurrence)
            - Comparing survival between groups
            - Estimating survival at fixed times
            
            **Interpretation:**
            - Y-axis = % surviving
            - X-axis = time
            - Step down = event occurred
            - Median survival = 50% point
            
            **Log-Rank Test:**
            - p < 0.05: Curves differ ✅
            - p ≥ 0.05: No difference ⚠️
            
            **Common Mistakes:**
            - Not handling censoring ❌
            - Comparing unequal follow-up times
            - Assuming flat after last event ❌
            
            **✨ NEW:** Can analyze both Original and Matched datasets!
            """)
            
            st.markdown("### Landmark Analysis")
            st.markdown("""
            **When to Use:**
            - Evaluating late/surrogate endpoints
            - Excluding early events
            - Controlling immortal time bias
            
            **Key Steps:**
            1. Select landmark time (e.g., t=1 year)
            2. Exclude pre-landmark events
            3. Reset time to 0 at landmark (✅ done auto)
            4. Compare post-landmark survival
            
            **Common Mistakes:**
            - Not resetting time ❌
            - Including pre-landmark events ❌
            - Too many landmarks (overfitting)
            """)
        
        with col2:
            st.markdown("### Cox Regression")
            st.markdown("""
            **When to Use:**
            - Multiple survival predictors
            - Adjusted hazard ratios
            - Semi-parametric modeling
            
            **Interpretation:**
            
            **HR (Hazard Ratio):**
            - HR > 1: Increased risk 🔴
            - HR < 1: Decreased risk (protective) 🟢
            - HR = 2 → 2× increased hazard
            
            **PH Assumption:**
            - Plot should be flat ✅
            - Non-flat → time-dependent effect ⚠️
            
            **🌳 Forest Plot**
            - Visual representation of HR with 95% CI
            - Shown in Web UI + HTML report
            - Interactive chart with error bars
            - Log scale for easy interpretation
            
            **Common Mistakes:**
            - Not checking PH assumption ❌
            - Time-varying covariates (use time-dep Cox)
            - Too many variables (overfitting)
            - Ignoring interactions
            """)
        
        st.markdown("### Subgroup Analysis")
        st.markdown("""
        **When to Use:**
        - Testing for treatment × subgroup interactions
        - Examining differential treatment effects
        - Identifying patient populations with greater benefit
        
        **Key Concepts:**
        - **Homogeneous effect** → One HR applies to all (no interaction)
        - **Heterogeneous effect** → Different HR by subgroup (interaction exists)
        - **Interaction p-value** → p < 0.05 = significant heterogeneity
        
        **Interpretation:**
        - If p_interaction < 0.05: Report results separately by subgroup
        - If p_interaction ≥ 0.05: Overall estimate applies to all
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Quick Decision Tree
        
        **Question: I have time-to-event data, single group?**
        → **KM/NA Curves** (Tab 1) - Visualize survival trajectory
        
        **Question: Compare survival between 2+ groups?**
        → **KM + Log-Rank** (Tab 1) - Test group differences
        
        **Question: Should I use landmark analysis?**
        → **Landmark** (Tab 2) - Exclude immortal time bias
        
        **Question: Multiple predictors affecting survival?**
        → **Cox Regression** (Tab 3) - Adjusted HR for each variable with **forest plot visualization** ✨ (shown in UI + HTML report)
        
        **Question: Does treatment effect vary by subgroup?**
        → **Subgroup Analysis** (Tab 4) - Test interaction, visualize HR by group
        
        **Question: Covariates change over time?**
        → **Time-Dependent Cox** (Advanced Survival tab)
        
        **💡 TIP:** After running PSM, switch to **"✅ Matched Data"** to compare survival outcomes in balanced cohort!
        """)
