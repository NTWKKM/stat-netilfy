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
        
        # State Management (เพื่อไม่ให้ผลหายเวลากด Checkbox)
        if 'cox_res' not in st.session_state: st.session_state.cox_res = None
        if 'cox_model_data' not in st.session_state: st.session_state.cox_model_data = None
        
        if st.button("Run Cox Model", key='btn_run_cox'):
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                # 🟢 UPDATE: รับค่า 4 ตัว (cph, res_df, data, err) ตาม survival_lib.py ตัวใหม่
                cph, res, model_data, err = survival_lib.fit_cox_ph(df, col_time, col_event, covariates)
                
                if err:
                    st.error(f"Error: {err}")
                    st.session_state.cox_res = None
                    st.session_state.cox_model_data = None
                else:
                    st.session_state.cox_res = res
                    st.session_state.cox_model_data = (cph, model_data) # เก็บ cph และ data ไว้เช็ค assumption
                    st.success("Model Fitted Successfully!")

        # แสดงผลลัพธ์ (ดึงจาก Session State)
        if st.session_state.cox_res is not None:
            res = st.session_state.cox_res
            st.dataframe(res.style.format("{:.4f}"))
            
            st.markdown("---")
            st.markdown("##### 🔍 Assumption Check")
            
            # Checkbox เพื่อความสะอาดของหน้าจอ
            check_assump = st.checkbox("Show Proportional Hazards Assumption Check (Schoenfeld Residuals)")
            
            if check_assump and st.session_state.cox_model_data:
                cph, data = st.session_state.cox_model_data
                
                try:
                    with st.spinner("Checking assumptions..."):
                        # 🟢 เรียกใช้ฟังก์ชันจาก lib (ง่ายกว่าและได้ Text Advice ด้วย)
                        txt_report, fig_assump = survival_lib.check_cph_assumptions(cph, data)
                        
                        st.text_area("Assumption Report & Advice:", value=txt_report, height=150)
                        
                        if fig_assump:
                            st.pyplot(fig_assump)
                            # ไม่ต้อง plt.close ที่นี่ เพราะเดี๋ยวส่งไปทำ report
                        
                        # Prepare Report Elements
                        elements = [
                            {'type':'header','data':'Cox Proportional Hazards'},
                            {'type':'table','data':res},
                            {'type':'header','data':'Assumption Check (Schoenfeld Residuals)'},
                            {'type':'text','data':f"<pre>{txt_report}</pre>"}
                        ]
                        if fig_assump:
                             elements.append({'type':'plot','data':fig_assump})
                        
                        report_html = survival_lib.generate_report_survival(f"Cox: {col_time}", elements)
                        st.download_button("📥 Download Report (Cox)", report_html, "cox_report.html", "text/html")
                        
                except Exception as e:
                    st.warning(f"Could not plot assumptions: {e}")
