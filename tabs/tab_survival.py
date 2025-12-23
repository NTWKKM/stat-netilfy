import streamlit as st
import pandas as pd
import numpy as np
import survival_lib
import time
from pandas.api.types import is_numeric_dtype
import logging
from forest_plot_lib import create_forest_plot_from_cox  # 🟢 Import HR forest plot

logger = logging.getLogger(__name__)

# 🟢 NEW: Helper function to select between original and matched datasets
def _get_dataset_for_survival(df: pd.DataFrame):
    """
    Selects which dataset to use for survival analysis and returns the chosen DataFrame with a descriptive label.
    
    Parameters:
        df (pd.DataFrame): The original dataset to use if no matched dataset is chosen or available.
    
    Returns:
        tuple: (selected_df, label) where `selected_df` is either the original `df` or the matched dataset from session state, and `label` is a short descriptive string indicating the source and row count (e.g., "✅ Matched Data (123 rows)" or "📊 Original Data (123 rows)").
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


def render(df, _var_meta):
    """
    Render the Streamlit UI for conducting survival analyses (Kaplan–Meier, Nelson–Aalen, landmark analysis, and Cox regression) on a selected dataset.
    
    Parameters:
        df (pd.DataFrame): Input dataset from which time-to-event, event indicator, and covariates are selected. May be swapped with a matched dataset if present in Streamlit session state.
        _var_meta (Any): Optional variable metadata (unused by the UI), provided for compatibility with the surrounding app.
    """
    st.subheader("⏳ Survival Analysis")
    st.info("""
**💡 Guide:**
* **Survival Analysis** models the relationship between predictors and the **Time-to-Event**.
* **Hazard Ratio (HR):** >1 Increased Hazard (Risk), <1 Decreased Hazard (Protective).
""")
    
    # 🟢 NEW: Display matched data status and selector
    if st.session_state.get("is_matched", False):
        st.info("✅ **Matched Dataset Available** - You can select it below for analysis")
    
    # 🟢 NEW: Select dataset (original or matched)
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
    col_event = c2.selectbox("💫 Event Variable (1=Event):", all_cols, index=event_idx, key='surv_event')
    
    # Tabs
    tab_curves, tab_landmark, tab_cox, tab_ref = st.tabs(["📈 Survival Curves (KM/NA)", "📑 Landmark Analysis", "📊 Cox Regression", "ℹ️ Reference & Interpretation"])
    
    # ==========================
    # TAB 1: Curves (KM & Nelson-Aalen)
    # ==========================
    with tab_curves:
        c1, c2 = st.columns([1, 2])
        col_group = c1.selectbox("Compare Groups (Optional):", ["None", *all_cols], key='surv_group')
        plot_type = c2.radio("Select Plot Type:", ["Kaplan-Meier (Survival Function)", "Nelson-Aalen (Cumulative Hazard)"], horizontal=True)
        
        if st.button("Run Analysis", key='btn_run_curves'):
            grp = None if col_group == "None" else col_group
            try:
                if "Kaplan-Meier" in plot_type:
                    # Run KM (using surv_df instead of df)
                    fig, stats_df = survival_lib.fit_km_logrank(surv_df, col_time, col_event, grp)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("##### Log-Rank / Statistics")
                    st.dataframe(stats_df)
                    
                    elements = [{'type':'header','data':'Kaplan-Meier'}, {'type':'plot','data':fig}, {'type':'table','data':stats_df}]
                    report_html = survival_lib.generate_report_survival(f"KM: {col_time}", elements)
                    st.download_button("📥 Download Report (KM)", report_html, "km_report.html", "text/html")
                    
                else:
                    # Run Nelson-Aalen (using surv_df instead of df)
                    fig, stats_df = survival_lib.fit_nelson_aalen(surv_df, col_time, col_event, grp)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("##### Summary Statistics (N / Events)")
                    st.dataframe(stats_df)
                    st.caption("Note: Nelson-Aalen estimates the cumulative hazard rate function (H(t)).")
                    
                    elements = [
                        {'type':'header','data':'Nelson-Aalen Cumulative Hazard'}, 
                        {'type':'plot','data':fig},
                        {'type':'header','data':'Summary Statistics'},
                        {'type':'table','data':stats_df}
                    ]
                    report_html = survival_lib.generate_report_survival(f"NA: {col_time}", elements)
                    st.download_button("📥 Download Report (NA)", report_html, "na_report.html", "text/html")
                    
            except Exception as e:
                logger.exception("Unexpected error in survival curves analysis")
                st.error(f"Error: {e}")

    # ==========================
    # TAB 2: Landmark Analysis
    # ==========================
    with tab_landmark:    
        st.caption("Principle: Exclude patients who had an event or were censored before the Landmark Time.")
        
        # Calculate Max Time (Robust check) - using surv_df instead of df
        max_t = surv_df[col_time].dropna().max() if not surv_df.empty and is_numeric_dtype(surv_df[col_time]) and surv_df[col_time].notna().any() else 1.0
        if max_t <= 0:
            max_t = 1.0 
        
        st.write(f"**Select Landmark Time (Max: {max_t:.2f})**")
        
        # State Management for Landmark Time
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
        
        # Auto-detect group column for landmark analysis
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

            if st.button("Run Landmark Analysis", key='btn_lm_sur'):
                try:
                    with st.spinner(f"Running Landmark Analysis at t={landmark_t:.2f}..."):
                        # Use surv_df instead of df
                        fig, stats, n_pre, n_post, err = survival_lib.fit_km_landmark(
                            surv_df, col_time, col_event, col_group, landmark_t
                        )
                        
                    if err:
                        st.error(err)
                    elif fig:
                        st.markdown(f"""
                        <p style='font-size:1em;'>
                        Total N before filter: <b>{n_pre}</b> | 
                        N Included (Survived $\ge$ {landmark_t:.2f}): <b>{n_post}</b> | 
                        N Excluded: <b>{n_pre - n_post}</b>
                        </p>
                        """, unsafe_allow_html=True)
                            
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("##### Log-Rank Test Results (Post-Landmark)")
                        st.dataframe(stats)
                            
                        elements = [
                            {'type':'header','data':f'Landmark Analysis (Survival from t={landmark_t:.2f})'},
                            {'type':'text', 'data': f"N Included: {n_post}, N Excluded: {n_pre - n_post}"},
                            {'type':'plot','data':fig},
                            {'type':'table','data':stats}
                        ]
                            
                        report_html = survival_lib.generate_report_survival(f"Landmark Analysis: {col_time} (t >= {landmark_t})", elements)
                        st.download_button("📥 Download Report (Landmark)", report_html, "lm_report.html", "text/html")
                        
                except (ValueError, KeyError) as e:
                    st.error(f"Analysis error: {e}")
                except Exception as e:
                    logger.exception("Unexpected error in landmark analysis")
                    st.error(f"Unexpected error: {e}")

    # ==========================
    # TAB 3: Cox Regression
    # ==========================
    with tab_cox:
        covariates = st.multiselect("Select Covariates (Predictors):", [c for c in all_cols if c not in [col_time, col_event]], key='surv_cox_vars')
        
        if 'cox_res' not in st.session_state: 
            st.session_state.cox_res = None
        if 'cox_html' not in st.session_state: 
            st.session_state.cox_html = None
        if 'cox_hr_dict' not in st.session_state:  # 🟢 NEW: Store HR dict for forest plot
            st.session_state.cox_hr_dict = {}

        if st.button("🚀 Run Cox Model & Check Assumptions", key='btn_run_cox'):
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                try:
                    with st.spinner("Fitting Cox Model and Checking Assumptions..."):
                        # Use surv_df instead of df
                        cph, res, model_data, err = survival_lib.fit_cox_ph(surv_df, col_time, col_event, covariates)
                        
                        if err:
                            st.error(f"Error: {err}")
                            st.session_state.cox_res = None
                            st.session_state.cox_html = None
                        else:
                            txt_report, fig_images = survival_lib.check_cph_assumptions(cph, model_data)
                            
                            st.session_state.cox_res = res
                            st.success("Analysis Complete!")
                            
                            # 🟢 FIXED: Apply formatting ONLY to numeric columns to avoid string error
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
                            
                            if txt_report:
                                with st.expander("View Assumption Advice (Text)", expanded=False):
                                    st.text(txt_report)
                            
                            if fig_images:
                                st.write("**Schoenfeld Residuals Plots:**")
                                for img_bytes in fig_images:
                                    st.image(img_bytes, caption="Assumption Check Plot", use_container_width=True)
                            else:
                                st.info("No assumption plots generated.")

                            elements = [
                                {'type':'header','data':'Cox Proportional Hazards'},
                                {'type':'table','data':res},
                                {'type':'header','data':'Assumption Check (Schoenfeld Residuals)'},
                                {'type':'preformatted','data':txt_report} 
                            ]
                            
                            if fig_images:
                                for img_bytes in fig_images:
                                    elements.append({'type':'image','data':img_bytes})
                            
                            report_html = survival_lib.generate_report_survival(f"Cox: {col_time}", elements)
                            st.session_state.cox_html = report_html
                            
                            # 🟢 NEW: Extract HR dict from results for forest plot
                            hr_dict = {}
                            if res is not None and not res.empty:
                                for idx, row in res.iterrows():
                                    var_name = str(idx)
                                    hr_dict[var_name] = {
                                        'hr': float(row.get('HR', row.get('exp(coef)', np.nan))),
                                        'ci_low': float(row.get('95% CI Lower', np.nan)),
                                        'ci_high': float(row.get('95% CI Upper', np.nan))
                                    }
                            st.session_state.cox_hr_dict = hr_dict

                except (ValueError, KeyError, RuntimeError) as e:
                    st.error(f"Analysis error: {e}")
                    st.session_state.cox_res = None
                except Exception as e:
                    logger.exception("Unexpected error in Cox regression analysis")
                    st.error(f"Unexpected error: {e}")
                    st.session_state.cox_res = None

        if st.session_state.cox_html:
            st.download_button("📥 Download Full Report (Cox)", st.session_state.cox_html, "cox_report.html", "text/html")
        
        # 🟢 NEW: Display Forest Plot for HR
        if st.session_state.cox_hr_dict:
            st.markdown("---")
            st.subheader("🌳 Forest Plot: Hazard Ratios")
            
            try:
                fig_forest = create_forest_plot_from_cox(
                    st.session_state.cox_hr_dict,
                    title="Forest Plot: Hazard Ratios (95% CI)"
                )
                st.plotly_chart(fig_forest, use_container_width=True)
                
                # Summary table
                st.markdown("**Summary Table:**")
                hr_table_data = []
                for var, res_data in st.session_state.cox_hr_dict.items():
                    hr_table_data.append({
                        'Variable': var,
                        'HR': f"{res_data['hr']:.4f}",
                        'CI Lower': f"{res_data['ci_low']:.4f}",
                        'CI Upper': f"{res_data['ci_high']:.4f}",
                    })
                hr_df = pd.DataFrame(hr_table_data)
                st.dataframe(hr_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.warning(f"Could not generate HR forest plot: {e}")

    # ==========================
    # TAB 4: Reference & Interpretation
    # ==========================
    with tab_ref:
        st.markdown("##### 📚 Quick Reference: Survival Analysis")
        
        st.info("""
        **🎰 When to Use What:**
        
        | Method | Purpose | Output |
        |--------|---------|--------|
        | **KM Curves** | Visualize time-to-event by group | Survival %, median, p-value |
        | **Nelson-Aalen** | Cumulative hazard over time | H(t) curve, risk accumulation |
        | **Landmark** | Late/surrogate endpoints | Filtered KM, immortal time removed |
        | **Cox** | Multiple predictors of survival | HR, CI, p-value per variable |
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
            
            **Common Mistakes:**
            - Not checking PH assumption ❌
            - Time-varying covariates (use time-dep Cox)
            - Too many variables (overfitting)
            - Ignoring interactions
            
            **✨ NEW:** Forest plot visualization of HR with 95% CI!
            """)
            
            st.markdown("### Nelson-Aalen")
            st.markdown("""
            **When to Use:**
            - Cumulative hazard visualization
            - Risk accumulation over time
            - Non-parametric alternative to KM
            
            **Interpretation:**
            - Steeper curve = higher hazard
            - Flat at end = no new events
            - Useful for diagnosis checking
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
        → **Cox Regression** (Tab 3) - Adjusted HR for each variable with **forest plot visualization** ✨
        
        **Question: Covariates change over time?**
        → **Time-Dependent Cox** (Advanced Survival tab)
        
        **💡 TIP:** After running PSM, switch to **"✅ Matched Data"** to compare survival outcomes in balanced cohort!
        """)
