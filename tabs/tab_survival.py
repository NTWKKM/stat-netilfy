import streamlit as st
import pandas as pd
import numpy as np
import survival_lib
import time
import pandas.api.types as ptypes # 🟢 Import เพื่อความชัดเจนในการใช้ types

def render(df, _var_meta):
    """
    Render an interactive Streamlit UI for survival analysis including Kaplan-Meier, Nelson-Aalen, landmark analysis, and Cox regression workflows.
    """
    st.subheader("5. Survival Analysis")
    st.info("""
**💡 Guide:**
* **Survival Analysis** models the relationship between predictors and the **Time-to-Event**.
* **Hazard Ratio (HR):** >1 Increased Hazard (Risk), <1 Decreased Hazard (Protective).
""")
    
    all_cols = df.columns.tolist()
    
    if len(all_cols) < 2:
        st.error("Dataset must contain at least 2 columns (time and event).")
        return
        
    # Global Selectors
    c1, c2 = st.columns(2)
    
    # Auto-detect logic (ใช้โค้ดเดิม)
    time_idx = 0
    for k in ['stop', 'time', 'dur']:
        found = next((i for i, c in enumerate(all_cols) if k in c.lower()), None)
        if found is not None:
            time_idx = found
            break
            
    event_idx = next((i for i, c in enumerate(all_cols) if 'event' in c.lower() or 'status' in c.lower() or 'dead' in c.lower()), min(1, len(all_cols)-1))
    
    col_time = c1.selectbox("⏳ Time Variable:", all_cols, index=time_idx, key='surv_time')
    col_event = c2.selectbox("💀 Event Variable (1=Event):", all_cols, index=event_idx, key='surv_event')
    
    # Tabs
    tab_curves, tab_landmark, tab_cox = st.tabs(["📉 Survival Curves (KM/NA)", "📍 Landmark Analysis", "📊 Cox Regression"])
    
    # ==========================
    # TAB 1: Curves (KM & Nelson-Aalen) - ใช้โค้ดเดิม
    # ==========================
    with tab_curves:
        c1, c2 = st.columns([1, 2])
        col_group = c1.selectbox("Compare Groups (Optional):", ["None", *all_cols], key='surv_group')
        plot_type = c2.radio("Select Plot Type:", ["Kaplan-Meier (Survival Function)", "Nelson-Aalen (Cumulative Hazard)"], horizontal=True)
        
        if st.button("Run Analysis", key='btn_run_curves'):
            grp = None if col_group == "None" else col_group
            try:
                if "Kaplan-Meier" in plot_type:
                    # Run KM
                    fig, stats_df = survival_lib.fit_km_logrank(df, col_time, col_event, grp)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("##### Log-Rank / Statistics")
                    st.dataframe(stats_df)
                    
                    elements = [{'type':'header','data':'Kaplan-Meier'}, {'type':'plot','data':fig}, {'type':'table','data':stats_df}]
                    report_html = survival_lib.generate_report_survival(f"KM: {col_time}", elements)
                    st.download_button("📥 Download Report (KM)", report_html, "km_report.html", "text/html")
                    
                else:
                    # Run Nelson-Aalen
                    fig, stats_df = survival_lib.fit_nelson_aalen(df, col_time, col_event, grp)
                    
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
                st.error(f"Error: {e}")

    # ==========================
    # TAB 2: Landmark Analysis 🟢 UPDATED
    # ==========================
    with tab_landmark:    
        st.caption("Principle: Exclude patients who had an event or were censored before the Landmark Time.")
        
        # Calculate Max Time (Robust check)
        max_t = df[col_time].dropna().max() if not df.empty and ptypes.is_numeric_dtype(df[col_time]) and df[col_time].notna().any() else 1.0
        if max_t <= 0:
            max_t = 1.0 
        
        st.write(f"**Select Landmark Time (Max: {max_t:.2f})**")
        
        # State Management for Landmark Time
        if 'landmark_val' not in st.session_state:
            st.session_state.landmark_val = float(max_t) * 0.1

        def update_from_slider() -> None:
            st.session_state.landmark_val = st.session_state.lm_slider_widget
        
        def update_from_number() -> None:
            st.session_state.landmark_val = st.session_state.lm_number_widget

        c_slide, c_num = st.columns([3, 1])
        with c_slide:
            # ใช้ st.session_state.landmark_val เป็น default value
            st.slider("Use Slider:", min_value=0.0, max_value=float(max_t) * 0.99, key='lm_slider_widget', value=min(st.session_state.landmark_val, float(max_t) * 0.99), on_change=update_from_slider, label_visibility="collapsed")
        with c_num:
            # ใช้ st.session_state.landmark_val เป็น default value
            st.number_input("Enter Value:", min_value=0.0, max_value=float(max_t) * 0.99, key='lm_number_widget', value=min(st.session_state.landmark_val, float(max_t) * 0.99), on_change=update_from_number, step=1.0, label_visibility="collapsed")
            
        landmark_t = st.session_state.landmark_val
        st.info(f"📍 Current Landmark Time: **{landmark_t:.2f}** ({col_time})")
        
        col_group = st.selectbox("Compare Group:", [c for c in all_cols if c not in [col_time, col_event]], key='lm_group_sur')

        if st.button("Run Landmark Analysis", key='btn_lm_sur'):
            if col_group is None:
                st.error("Please select a Group Variable for comparison.")
                return

            try:
                with st.spinner(f"Running Landmark Analysis at t={landmark_t:.2f}..."):
                    # 🟢 CALL NEW LANDMARK FUNCTION: ให้ฟังก์ชันจัดการ Filtering และ Time Reset เอง
                    fig, stats, n_pre, n_post, err = survival_lib.fit_km_landmark(
                        df, col_time, col_event, col_group, landmark_t
                    )
                
                if err:
                    st.error(err)
                elif fig:
                    # Show Filtering Results
                    st.markdown(f"""
                    <p style='font-size:1em;'>
                    Total N before filter: <b>{n_pre}</b> | 
                    N Included (Survived $\\ge$ {landmark_t:.2f}): <b>{n_post}</b> | 
                    N Excluded: <b>{n_pre - n_post}</b>
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # 🟢 Graph is now correctly zero-based and filtered
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
                st.error(f"Unexpected error: {e}")
                # Consider: import traceback; st.exception(e) for debugging

    # ==========================
    # TAB 3: Cox Regression - ใช้โค้ดเดิม
    # ==========================
    with tab_cox:
        covariates = st.multiselect("Select Covariates (Predictors):", [c for c in all_cols if c not in [col_time, col_event]], key='surv_cox_vars')
        
        if 'cox_res' not in st.session_state: 
            st.session_state.cox_res = None
        if 'cox_html' not in st.session_state: 
            st.session_state.cox_html = None

        if st.button("🚀 Run Cox Model & Check Assumptions", key='btn_run_cox'):
            if not covariates:
                st.error("Please select at least one covariate.")
            else:
                try:
                    with st.spinner("Fitting Cox Model and Checking Assumptions..."):
                        cph, res, model_data, err = survival_lib.fit_cox_ph(df, col_time, col_event, covariates)
                        
                        if err:
                            st.error(f"Error: {err}")
                            st.session_state.cox_res = None
                            st.session_state.cox_html = None
                        else:
                            txt_report, fig_images = survival_lib.check_cph_assumptions(cph, model_data)
                            
                            st.session_state.cox_res = res
                            st.success("Analysis Complete!")
                            
                            st.dataframe(res.style.format("{:.4f}"))
                            st.markdown("##### 🔍 Proportional Hazards Assumption Check")
                            
                            if txt_report:
                                with st.expander("View Assumption Advice (Text)", expanded=False):
                                    st.text(txt_report)
                            
                            if fig_images:
                                st.write("**Schoenfeld Residuals Plots:**")
                                for img_bytes in fig_images:
                                    st.image(img_bytes, caption="Assumption Check Plot", use_column_width=True)
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

                except (ValueError, KeyError, RuntimeError) as e:
                    st.error(f"Analysis error: {e}")
                    st.session_state.cox_res = None
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.session_state.cox_res = None
                    # Consider: import traceback; st.exception(e) for debugging

        if st.session_state.cox_html:
            st.download_button("📥 Download Full Report (Cox)", st.session_state.cox_html, "cox_report.html", "text/html")

# ฟังก์ชัน render ต้องถูกเรียกใช้จาก app.py หรือไฟล์หลัก
# if __name__ == '__main__':
#     render(pd.DataFrame(), {})
