import streamlit as st
import pandas as pd
import survival_lib # Import ไฟล์ logic ใหม่

def render(df, var_meta):
    st.subheader("5. Survival Analysis")
    st.info("""
    **💡 Guide:** วิเคราะห์ระยะเวลาจนถึงการเกิดเหตุการณ์ (Time-to-Event Analysis)
    
    * **Kaplan-Meier (KM) Curve:** กราฟแสดงโอกาสรอดชีพสะสมตามเวลา และ Log-Rank Test เพื่อเปรียบเทียบระหว่างกลุ่ม
    * **Cox Proportional Hazards:** วิเคราะห์หาปัจจัยเสี่ยง (Hazard Ratio) ที่มีผลต่อเวลาการรอดชีพ (Multivariate)
    * **Assumptions Check:** ตรวจสอบข้อตกลงเบื้องต้นของ Cox Model (Proportional Hazards Assumption)
    """)
    
    all_cols = df.columns.tolist()
    
    # --- Select Variables (Global for this tab) ---
    c1, c2, c3 = st.columns(3)
    
    # 1. Time Column
    time_idx = 0
    for i, c in enumerate(all_cols):
        if 'time' in c.lower() or 'dur' in c.lower() or 'os' in c.lower(): time_idx = i; break
    col_time = c1.selectbox("⏳ Time Variable (Numeric):", all_cols, index=time_idx, key='surv_time')

    # 2. Event Column
    event_idx = 0
    for i, c in enumerate(all_cols):
        if 'event' in c.lower() or 'status' in c.lower() or 'dead' in c.lower() or 'died' in c.lower(): event_idx = i; break
    col_event = c2.selectbox("💀 Event Variable (0=Censored, 1=Event):", all_cols, index=event_idx, key='surv_event')
    
    # 3. Group/Covariates (แยกตามการใช้งาน)
    st.markdown("---")
    
    tab_km, tab_cox = st.tabs(["📉 Kaplan-Meier & Log-Rank", "📊 Cox Regression & Assumptions"])
    
    # ==========================
    # TAB 1: Kaplan-Meier
    # ==========================
    with tab_km:
        st.markdown("##### Kaplan-Meier Survival Curve")
        col_group = st.selectbox("Compare Groups (Optional):", ["None"] + all_cols, key='surv_group_km')
        
        run_km = st.button("Run Kaplan-Meier", key='btn_run_km')
        
        if 'html_surv_km' not in st.session_state: st.session_state.html_surv_km = None
        
        if run_km:
            grp = None if col_group == "None" else col_group
            try:
                fig, stats_df = survival_lib.fit_km_logrank(df, col_time, col_event, grp)
                
                # แสดงผลใน Streamlit ทันที
                c_plot, c_stat = st.columns([2, 1])
                c_plot.pyplot(fig)
                c_stat.dataframe(stats_df)
                
                # สร้าง Report HTML
                elements = [
                    {'type': 'header', 'data': 'Kaplan-Meier Analysis'},
                    {'type': 'text', 'data': f'<b>Time:</b> {col_time}, <b>Event:</b> {col_event}, <b>Group:</b> {grp}'},
                    {'type': 'plot', 'data': fig},
                    {'type': 'header', 'data': 'Statistics'},
                    {'type': 'table', 'data': stats_df}
                ]
                st.session_state.html_surv_km = survival_lib.generate_report_survival(f"KM: {col_time}", elements)
                
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.html_surv_km:
             st.download_button("📥 Download Report (KM)", st.session_state.html_surv_km, "km_report.html", "text/html")

    # ==========================
    # TAB 2: Cox Regression
    # ==========================
    with tab_cox:
        st.markdown("##### Cox Proportional Hazards Model")
        covariates = st.multiselect("Select Covariates (Predictors):", [c for c in all_cols if c not in [col_time, col_event]], key='surv_cox_vars')
        
        run_cox = st.button("Run Cox Model", key='btn_run_cox')
        
        if 'html_surv_cox' not in st.session_state: st.session_state.html_surv_cox = None
        
        if run_cox:
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                cph_model, res_df, err = survival_lib.fit_cox_ph(df, col_time, col_event, covariates)
                
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Model Fitted Successfully!")
                    st.dataframe(res_df.style.format("{:.4f}"))
                    
                    # --- Assumption Check ---
                    st.markdown("##### 🔍 Proportional Hazards Assumption Check")
                    st.write("Checking if the Hazard Ratio remains constant over time...")
                    
                    try:
                        import matplotlib.pyplot as plt
                        fig_assump, ax_assump = plt.subplots(figsize=(10, 6))
                        cph_model.check_assumptions(survival_lib.clean_survival_data(df, col_time, col_event, covariates), show_plots=True, ax=ax_assump)
                        st.pyplot(fig_assump)
                        
                        # Report HTML
                        elements = [
                            {'type': 'header', 'data': 'Cox Proportional Hazards Model'},
                            {'type': 'text', 'data': f'<b>Time:</b> {col_time}, <b>Event:</b> {col_event}'},
                            {'type': 'header', 'data': 'Model Summary'},
                            {'type': 'table', 'data': res_df},
                            {'type': 'header', 'data': 'Assumption Check (Schoenfeld Residuals)'},
                            {'type': 'plot', 'data': fig_assump}
                        ]
                        st.session_state.html_surv_cox = survival_lib.generate_report_survival(f"Cox: {col_time}", elements)
                        
                    except Exception as e:
                        st.warning(f"Could not plot assumptions: {e}")

        if st.session_state.html_surv_cox:
             st.download_button("📥 Download Report (Cox)", st.session_state.html_surv_cox, "cox_report.html", "text/html")
