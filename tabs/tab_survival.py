import streamlit as st
import pandas as pd
import survival_lib

def render(df, var_meta):
    st.subheader("5. Survival Analysis")
    st.info("""
    **💡 Analysis Modules:**
    1.  **Survival Curves:** Kaplan-Meier (Survival Prob.) & Nelson-Aalen (Cumulative Hazard).
    2.  **Cox Regression:** Identify risk factors (Hazard Ratios) & Check Assumptions.
    """)
    
    all_cols = df.columns.tolist()
    
    # Global Selectors
    c1, c2 = st.columns(2)
    time_idx = next((i for i, c in enumerate(all_cols) if 'time' in c.lower()), 0)
    event_idx = next((i for i, c in enumerate(all_cols) if 'event' in c.lower() or 'status' in c.lower()), 0)
    
    col_time = c1.selectbox("⏳ Time Variable:", all_cols, index=time_idx, key='surv_time')
    col_event = c2.selectbox("💀 Event Variable (1=Event):", all_cols, index=event_idx, key='surv_event')
    
    # Tabs
    tab_curves, tab_cox = st.tabs(["📉 Survival Curves (KM/NA)", "📊 Cox Regression"])
    
    # ==========================
    # TAB 1: Curves (KM & Nelson-Aalen)
    # ==========================
    with tab_curves:
        c1, c2 = st.columns([1, 2])
        col_group = c1.selectbox("Compare Groups (Optional):", ["None"] + all_cols, key='surv_group')
        
        # 🟢 ตัวเลือกกราฟ
        plot_type = c2.radio("Select Plot Type:", 
                             ["Kaplan-Meier (Survival Function)", "Nelson-Aalen (Cumulative Hazard)"], 
                             horizontal=True)
        
        if st.button("Run Analysis", key='btn_run_curves'):
            grp = None if col_group == "None" else col_group
            try:
                if "Kaplan-Meier" in plot_type:
                    # Run KM
                    fig, stats_df = survival_lib.fit_km_logrank(df, col_time, col_event, grp)
                    st.pyplot(fig)
                    st.markdown("##### Log-Rank / Statistics")
                    st.dataframe(stats_df)
                    
                    # Report HTML (KM)
                    elements = [{'type':'header','data':'Kaplan-Meier'}, {'type':'plot','data':fig}, {'type':'table','data':stats_df}]
                    report_html = survival_lib.generate_report_survival(f"KM: {col_time}", elements)
                    st.download_button("📥 Download Report (KM)", report_html, "km_report.html", "text/html")
                    
                else:
                    # 🟢 Run Nelson-Aalen
                    fig = survival_lib.fit_nelson_aalen(df, col_time, col_event, grp)
                    st.pyplot(fig)
                    st.caption("Note: Nelson-Aalen estimates the cumulative hazard rate function (H(t)).")
                    
                    # Report HTML (NA)
                    elements = [{'type':'header','data':'Nelson-Aalen Cumulative Hazard'}, {'type':'plot','data':fig}]
                    report_html = survival_lib.generate_report_survival(f"NA: {col_time}", elements)
                    st.download_button("📥 Download Report (NA)", report_html, "na_report.html", "text/html")
                    
            except Exception as e:
                st.error(f"Error: {e}")

    # ==========================
    # TAB 2: Cox Regression
    # ==========================
    with tab_cox:
        covariates = st.multiselect("Select Covariates (Predictors):", 
                                    [c for c in all_cols if c not in [col_time, col_event]], 
                                    key='surv_cox_vars')
        
        if st.button("Run Cox Model", key='btn_run_cox'):
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                cph, res, err = survival_lib.fit_cox_ph(df, col_time, col_event, covariates)
                
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Model Fitted Successfully!")
                    st.dataframe(res.style.format("{:.4f}"))
                    
                    st.markdown("##### 🔍 Assumption Check (Schoenfeld Residuals)")
                    try:
                        import matplotlib.pyplot as plt
                        
                        # --- FIX for Report: Capture generated figures ---
                        
                        # 1. Get a list of currently active figure numbers before running check_assumptions
                        initial_fignums = plt.get_fignums()
                        
                        # 2. Run check_assumptions to generate plots
                        # FIX: Remove the faulty 'ax' parameter
                        cph.check_assumptions(survival_lib.clean_survival_data(df, col_time, col_event, covariates), show_plots=True)
                        
                        # 3. Capture newly generated figures
                        final_fignums = plt.get_fignums()
                        new_fignums = [num for num in final_fignums if num not in initial_fignums]
                        
                        # Get the actual Figure objects, display them, and prepare for report
                        assumption_figs = []
                        for num in new_fignums:
                            fig = plt.figure(num)
                            st.pyplot(fig) # Display figure in Streamlit
                            assumption_figs.append(fig)
                        
                        # Report HTML (Cox) - NOW INCLUDE ALL CAPTURED FIGURES
                        elements = [
                            {'type':'header','data':'Cox Proportional Hazards'},
                            {'type':'table','data':res},
                            {'type':'header','data':'Assumption Check Plots (Schoenfeld Residuals)'},
                            # Include all captured figures in the report elements
                            *[{'type':'plot','data':fig} for fig in assumption_figs] 
                        ]
                        report_html = survival_lib.generate_report_survival(f"Cox: {col_time}", elements)
                        st.download_button("📥 Download Report (Cox)", report_html, "cox_report.html", "text/html")
                        
                    except Exception as e:
                        st.warning(f"Could not plot or report assumptions: {e}")
