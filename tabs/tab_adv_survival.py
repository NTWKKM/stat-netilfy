import streamlit as st
import pandas as pd
import survival_lib

def render(df, _var_meta):
    """
    Render a Streamlit UI for fitting a time-dependent (start-stop) Cox proportional hazards model.
    
    Displays controls to select ID, start time, stop time, event indicator, and time-dependent covariates; validates that at least one covariate is selected and that the selected time/event/covariate columns are numeric, shows a spinner while fitting, calls survival_lib.fit_cox_time_varying, and presents model results or errors.
    
    Parameters:
        df (pandas.DataFrame): Input dataset in long (start-stop) format containing ID, start, stop, event, and covariate columns.
        var_meta (Any): Optional variable metadata (not required by the UI; provided for compatibility with the tabs API).
    """
    st.subheader("⏳ Advanced Survival Analysis")
    st.info("""
    **Modules:**
    * **Time-Dependent Cox:** For variables that change over time (Requires Long-Format Data: Start-Stop).
    """)

    all_cols = df.columns.tolist() 

    # ==========================
    # 2. Time-Dependent Cox
    # ==========================
    st.warning("⚠️ **Requirement:** Data must be in **Long Format** (Start-Stop rows).")
        
    c1, c2, c3, c4 = st.columns(4)
    id_col = c1.selectbox("🆔 ID Column:", all_cols, key='td_id')
    
    start_col = c2.selectbox("▶️ Start Time:", [c for c in all_cols if c != id_col], key='td_start')
    stop_col = c3.selectbox("⏹️ Stop Time:", [c for c in all_cols if c not in [id_col, start_col]], key='td_stop')
    event_col = c4.selectbox("💀 Event (at Stop):", [c for c in all_cols if c not in [id_col, start_col, stop_col]], key='td_event')
        
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
                _ctv, res, err = survival_lib.fit_cox_time_varying(df, id_col, event_col, start_col, stop_col, covs)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Model Converged!")
                    st.dataframe(res.style.format("{:.4f}"))
                    st.caption("Interpretation: HR > 1 indicates increased risk.")
                    
                    # Generate HTML report
                    report_elements = [
                        ("header", "Time-Dependent Cox Regression Results"),
                        ("text", f"**Model Configuration:**"),
                        ("text", f"- ID Column: {id_col}"),
                        ("text", f"- Start Time: {start_col}"),
                        ("text", f"- Stop Time: {stop_col}"),
                        ("text", f"- Event: {event_col}"),
                        ("text", f"- Covariates: {', '.join(covs)}"),
                        ("header2", "Model Results"),
                        ("table", res),
                        ("text", "**Interpretation:** Hazard Ratio (HR) > 1 indicates increased risk; HR < 1 indicates decreased risk. P-values < 0.05 suggest statistical significance.")
                    ]
                    
                    html_report = survival_lib.generate_report_survival(
                        "Time-Dependent Cox Regression Analysis",
                        report_elements
                    )
                    
                    # Store in session state
                    st.session_state["html_output_adv_survival"] = html_report
    
    # Download button (appears when report exists)
    if "html_output_adv_survival" in st.session_state and st.session_state["html_output_adv_survival"]:
        st.download_button(
            label="📥 Download Full Report (HTML)",
            data=st.session_state["html_output_adv_survival"],
            file_name="adv_survival_report.html",
            mime="text/html",
            key="download_adv_survival"
        )
