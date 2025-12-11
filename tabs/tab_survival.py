import streamlit as st
import pandas as pd
import survival_lib
import matplotlib.pyplot as plt

def render(df, var_meta):
    st.subheader("5. Survival Analysis")
    st.info("""
**💡 Guide:**
* **Survival Analysis** models the relationship between predictors and the **Time-to-Event**.
* **Hazard Ratio (HR):** >1 Increased Hazard (Risk), <1 Decreased Hazard (Protective).
""")
    
    all_cols = df.columns.tolist()
    
    # Global Selectors
    c1, c2 = st.columns(2)
    # Auto-detect logic
    time_idx = next((i for i, c in enumerate(all_cols) if 'time' in c.lower() or 'dur' in c.lower()), 0)
    event_idx = next((i for i, c in enumerate(all_cols) if 'event' in c.lower() or 'status' in c.lower() or 'dead' in c.lower()), min(1, len(all_cols)-1))
    
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
        plot_type = c2.radio("Select Plot Type:", ["Kaplan-Meier (Survival Function)", "Nelson-Aalen (Cumulative Hazard)"], horizontal=True)
        
        if st.button("Run Analysis", key='btn_run_curves'):
            grp = None if col_group == "None" else col_group
            try:
                if "Kaplan-Meier" in plot_type:
                    # Run KM
                    fig, stats_df = survival_lib.fit_km_logrank(df, col_time, col_event, grp)
                    st.pyplot(fig)
                    plt.close(fig) 
                    
                    st.markdown("##### Log-Rank / Statistics")
                    st.dataframe(stats_df)
                    
                    elements = [{'type':'header','data':'Kaplan-Meier'}, {'type':'plot','data':fig}, {'type':'table','data':stats_df}]
                    report_html = survival_lib.generate_report_survival(f"KM: {col_time}", elements)
                    st.download_button("📥 Download Report (KM)", report_html, "km_report.html", "text/html")
                    
                else:
                    # Run Nelson-Aalen
                    fig, stats_df = survival_lib.fit_nelson_aalen(df, col_time, col_event, grp)
                    st.pyplot(fig)
                    plt.close(fig) 
                    
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
                st.error(f"Error: {e}")

    # ==========================
    # TAB 2: Cox Regression
    # ==========================
    with tab_cox:
        covariates = st.multiselect("Select Covariates (Predictors):", [c for c in all_cols if c not in [col_time, col_event]], key='surv_cox_vars')
        
        # State Management (ยังคงไว้เผื่อ Download Button)
        if 'cox_res' not in st.session_state: st.session_state.cox_res = None
        if 'cox_html' not in st.session_state: st.session_state.cox_html = None

        if st.button("🚀 Run Cox Model & Check Assumptions", key='btn_run_cox'):
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                try:
                    with st.spinner("Fitting Cox Model and Checking Assumptions..."):
                        # 1. Run Cox Model
                        cph, res, model_data, err = survival_lib.fit_cox_ph(df, col_time, col_event, covariates)
                        
                        if err:
                            st.error(f"Error: {err}")
                            st.session_state.cox_res = None
                            st.session_state.cox_html = None
                        else:
                            # 2. Check Assumptions (Auto Run)
                            txt_report, figs_assump = survival_lib.check_cph_assumptions(cph, model_data)
                            
                            st.session_state.cox_res = res
                            st.success("Analysis Complete!")
                            
                            # --- Display Results ---
                            st.dataframe(res.style.format("{:.4f}"))
                            
                            st.markdown("##### 🔍 Proportional Hazards Assumption Check")
                            
                            # Show Text Report
                            if txt_report:
                                with st.expander("View Assumption Advice (Text)", expanded=False):
                                    st.text(txt_report)
                            
                            # Show Plots (วนลูปโชว์ทุกรูป)
                            if figs_assump:
                                st.write("**Schoenfeld Residuals Plots:**")
                                for fig in figs_assump:
                                    st.pyplot(fig)
                                    # ไม่ต้อง plt.close() ที่นี่
                            else:
                                st.info("No assumption plots generated (maybe model is valid or too simple).")

                            # --- Generate Report for Download ---
                            elements = [
                                {'type':'header','data':'Cox Proportional Hazards'},
                                {'type':'table','data':res},
                                {'type':'header','data':'Assumption Check (Schoenfeld Residuals)'},
                                {'type':'text','data':f"<pre>{txt_report}</pre>"}
                            ]
                            if figs_assump:
                                 for fig in figs_assump:
                                     elements.append({'type':'plot','data':fig})
                            
                            report_html = survival_lib.generate_report_survival(f"Cox: {col_time}", elements)
                            st.session_state.cox_html = report_html

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Show Download Button (if result exists)
        if st.session_state.cox_html:
             st.download_button("📥 Download Full Report (Cox)", st.session_state.cox_html, "cox_report.html", "text/html")
