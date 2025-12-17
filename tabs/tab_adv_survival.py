import streamlit as st
import pandas as pd
import survival_lib

def render(df, _var_meta):
    """
    Render Streamlit controls to configure, fit, and report a time-dependent (start–stop) Cox proportional hazards model.
    
    Validates that at least one covariate is selected and that the selected start, stop, event, and covariate columns are numeric; calls survival_lib.fit_cox_time_varying to fit the model; displays results or errors in the UI; and stores a generated HTML report in st.session_state["html_output_adv_survival"] for download.
    
    Parameters:
        df (pandas.DataFrame): Input dataset in long (start–stop) format containing ID, start, stop, event, and covariate columns.
        _var_meta (Any): Optional variable metadata provided for compatibility with the tabs API (not used by the UI).
    """
    st.subheader("⏳ Advanced Survival Analysis")
    st.info("""
    **Modules:**
    * **Time-Dependent Cox:** For variables that change over time (Requires Long-Format Data: Start-Stop).
    """)

    all_cols = df.columns.tolist()
    if not all_cols:
        st.error("Dataset has no columns to analyze.")
        return

    # ==========================
    # 2. Time-Dependent Cox
    # ==========================
    st.warning("⚠️ **Requirement:** Data must be in **Long Format** (Start-Stop rows).")
        
    c1, c2, c3, c4 = st.columns(4)
    id_col = c1.selectbox("🆔 ID Column:", all_cols, key='td_id')
    
    start_col = c2.selectbox("▶️ Start Time:", [c for c in all_cols if c != id_col], key='td_start')
    stop_col = c3.selectbox("⏹️ Stop Time:", [c for c in all_cols if c not in [id_col, start_col]], key='td_stop')
    event_col = c4.selectbox("💫 Event (at Stop):", [c for c in all_cols if c not in [id_col, start_col, stop_col]], key='td_event')
        
    covs = st.multiselect("Select Time-Dependent Covariates:", 
                             [c for c in all_cols if c not in [id_col, start_col, stop_col, event_col]], 
                             key='td_covs')
        
    if st.button("Run Time-Dependent Model", key='btn_td'):
        if not covs:
            st.error("Select covariates first.")
        else:
            # Validate that required columns are numeric
            numeric_cols = [start_col, stop_col, event_col, *covs]
            non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                st.error(f"The following columns must be numeric: {', '.join(non_numeric)}")
                return

            with st.spinner("Fitting Model..."):
                _ctv, res, _data, err = survival_lib.fit_cox_time_varying(df, id_col, event_col, start_col, stop_col, covs)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Model Converged!")
                    st.dataframe(res.style.format("{:.4f}"))
                    st.caption("Interpretation: HR > 1 indicates increased risk.")
                    
                    # [แก้ไข] เปลี่ยนจาก Tuple เป็น Dictionary และแก้ header2 -> header
                    report_elements = [
                        {"type": "header", "data": "Time-Dependent Cox Regression Results"},
                        {"type": "header", "data": "Model Configuration"},
                        {"type": "text", "data": f"ID Column: {id_col}"},
                        {"type": "text", "data": f"Start Time: {start_col}"},
                        {"type": "text", "data": f"Stop Time: {stop_col}"},
                        {"type": "text", "data": f"Event: {event_col}"},
                        {"type": "text", "data": f"Covariates: {', '.join(covs)}"},
                        {"type": "header", "data": "Model Results"},  # แก้ header2 เป็น header
                        {"type": "table", "data": res},
                        {"type": "header", "data": "Interpretation"},
                        {"type": "text", "data": "Hazard Ratio (HR) > 1 indicates increased risk; HR < 1 indicates decreased risk. P-values < 0.05 suggest statistical significance."},
                    ]
                    
                    html_report = survival_lib.generate_report_survival(
                        "Time-Dependent Cox Regression Analysis",
                        report_elements
                    )
                    
                    # Store in session state
                    st.session_state["html_output_adv_survival"] = html_report
    
    # [แก้ไข] ปรับการเชค session state ให้สั้นลง (ตามคำแนณ Nitpick)
    html_output = st.session_state.get("html_output_adv_survival")
    if html_output:
        st.download_button(
            label="📥 Download Full Report (HTML)",
            data=html_output,  # ใช้ตัวแปรที่ดึงมาแลว
            file_name="adv_survival_report.html",
            mime="text/html",
            key="download_adv_survival"
        )
    
    # --- NEW: Reference & Interpretation ---
    st.markdown("---")
    with st.expander("📚 Reference & Interpretation"):
        st.markdown("""
        ### Time-Dependent Cox Regression Guide
        
        **When to Use:**
        - Variables that change over time
        - Long-format data (start-stop rows)
        - Time-varying confounders
        - Complex follow-up scenarios
        
        **vs. Regular Cox:**
        | Aspect | Regular Cox | Time-Dependent Cox |
        |--------|-------------|-------------------|
        | **Covariates** | Constant over time | Change at transitions |
        | **Data Format** | Wide (one row/person) | Long (start-stop rows) |
        | **Use Case** | Baseline predictors | Dynamic variables |
        | **Example** | Age, gender (fixed) | Drug dosage, treatment stage (changing) |
        
        ---
        
        ### Data Format Requirements
        
        **Must be LONG Format (Start-Stop):**
        ```
        ID | Start | Stop | Event | Covariate_Value
        1  |   0   |  30  |   0   |      5
        1  |  30   |  60  |   1   |      7          <- Same ID, value changes
        1  |  60   |  90  |   0   |      8
        2  |   0   |  45  |   0   |      6
        2  |  45   |  100 |   1   |      9
        ```
        
        **Each row represents:**
        - Time interval [Start, Stop)
        - Covariate value during that interval
        - Event status at Stop time
        
        ---
        
        ### Key Concepts
        
        **Time-Dependent:**
        - Covariate value updates at transitions
        - Captures dynamic effect over follow-up
        - Robust to non-constant confounding ✅
        
        **Robust to:**
        - Changing treatment doses
        - Protocol changes mid-study
        - Varying medication compliance
        - Evolving risk factors
        
        **Interpretation:**
        - Same as regular Cox (HR, p-value)
        - HR reflects effect per unit change in covariate
        - Time-averaged effect across person-time
        
        ---
        
        ### Common Mistakes ❌
        
        - **Using wide-format instead of long-format** → Use long-format (start-stop)
        - **Not updating covariate values** → Update at each transition
        - **Forgetting to sort by ID, start time** → Sort for correct intervals
        - **Ignoring gap/overlap in intervals** → Check [Start, Stop] continuity
        - **Assuming constant when should be time-varying** → Use time-dep Cox if changes
        
        ---
        
        ### Example Interpretation
        
        **Model Output:**
        - Variable: Drug_Dose (time-varying)
        - HR = 1.05 (95% CI: 1.02 - 1.08)
        - p = 0.001
        
        **Interpretation:**
        For every 1-unit increase in drug dose during a follow-up interval, the hazard increases by 5%, controlling for time effects. This association is statistically significant (p < 0.05). ✅
        
        **Note:** Effects are averaged across the entire follow-up, accounting for dose changes over time.
        """)