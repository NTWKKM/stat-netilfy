import streamlit as st
import pandas as pd
import diag_test # Import from root

def render(df, var_meta):
    st.subheader("2. Diagnostic Test & Statistics")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📈 ROC Curve & AUC", "🎲 Chi-Square & Risk-RR,OR,NNT (Categorical)", "📊 Descriptive"])
    all_cols = df.columns.tolist()

    # --- ROC --- (ส่วนนี้เหมือนเดิม)
    with sub_tab1:
        st.markdown("##### ROC Curve Analysis")
        st.info("""
    **💡 Guide:** Evaluates the performance of a **continuous diagnostic test** (e.g., 'lab value' or 'risk index') against a **binary Gold Standard** (e.g., 'cancer' or 'not cancer').

    * **AUC (Area Under Curve):** Measures overall test discrimination ability (0.5 = random guess, 1.0 = perfect test).
    * **Youden Index (J):** Identifies the **optimal cut-off point** by maximizing the difference between Sensitivity and (1 - Specificity).
    * **P-value:** Tests the null hypothesis that the AUC is equal to 0.5 (i.e., the test performs better than chance).
        """)
        
        rc1, rc2, rc3, rc4 = st.columns(4)
        
        def_idx = 0
        for i, c in enumerate(all_cols):
            if 'outcome' in c.lower() or 'died' in c.lower(): def_idx = i; break
        
        truth = rc1.selectbox("Gold Standard (Binary):", all_cols, index=def_idx, key='roc_truth_diag')
        
        score_idx = 0
        for i, c in enumerate(all_cols):
            if 'score' in c.lower(): score_idx = i; break
        score = rc2.selectbox("Test Score (Continuous):", all_cols, index=score_idx, key='roc_score_diag')
        
        method = rc3.radio("CI Method:", ["DeLong et al.", "Binomial (Hanley)"], key='roc_method_diag')

        # Positive Label
        pos_label = None
        unique_vals = df[truth].dropna().unique()
        if len(unique_vals) == 2:
            sorted_vals = sorted([str(x) for x in unique_vals])
            
            # 🟢 NEW LOGIC: Set default index to '1' if it exists
            default_pos_idx = 0
            if '1' in sorted_vals:
                default_pos_idx = sorted_vals.index('1')
                
            pos_label = rc4.selectbox("Positive Label (1):", sorted_vals, index=default_pos_idx, key='roc_pos_diag')
        elif len(unique_vals) != 2:
            rc4.warning("Requires 2 unique values.")

        run_col, dl_col = st.columns([1, 1])
        if 'html_output_roc' not in st.session_state: st.session_state.html_output_roc = None
        
        if run_col.button("📉 Analyze ROC", key='btn_roc_diag'):
            if pos_label and len(unique_vals) == 2:
                # Call analyze_roc from diag_test
                res, err, fig, coords_df = diag_test.analyze_roc(df, truth, score, 'delong' if 'DeLong' in method else 'hanley', pos_label_user=pos_label)
                if err: st.error(err)
                else:
                    rep = [
                        {'type':'text', 'data':f"Analysis: <b>{score}</b> vs <b>{truth}</b>"},
                        {'type':'plot', 'data':fig},
                        {'type':'table', 'header':'Key Statistics', 'data':pd.DataFrame([res]).T},
                        {'type':'table', 'header':'Diagnostic Performance', 'data':coords_df}
                    ]
                    html = diag_test.generate_report(f"ROC: {score}", rep)
                    st.session_state.html_output_roc = html
                    st.components.v1.html(html, height=800, scrolling=True)
            else:
                st.error("Invalid Target configuration.")

        with dl_col:
            if st.session_state.html_output_roc:
                st.download_button("📥 Download Report", st.session_state.html_output_roc, "roc_report.html", "text/html", key='dl_roc_diag')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_roc_diag')

    # --- Chi-Square ---
    with sub_tab2:
        st.markdown("##### Chi-Square Test & Risk Analysis")
        st.info("""
            **💡 Guide:** Used to analyze the association between **two categorical variables**.
            * **Chi-Square Test:** Determines if there is a significant association between the variables (P-value).
            * **Risk/Odds Ratio:** For **2x2 tables**, the tool provides **automatically calculated** metrics: **Risk Ratio (RR)**, **Odds Ratio (OR)**, and **Number Needed to Treat (NNT)**.
            
            **Variable Selection:**
            * **Variable 1 (Row):** Typically the **Exposure**, **Risk Factor**, or **Intervention**.
            * **Variable 2 (Column):** Typically the **Outcome** or **Event** of interest.
        """)

        c1, c2, c3 = st.columns(3)
        
        # 🟢 UPDATE 1: Auto-select V1 and V2
        v1_default_name = 'Hypertension'
        v2_default_name = 'Outcome_Disease'
        
        # Locate index. Use 0 and min(1, len-1) as fallbacks.
        v1_idx = next((i for i, c in enumerate(all_cols) if c == v1_default_name), 0)
        v2_idx = next((i for i, c in enumerate(all_cols) if c == v2_default_name), min(1, len(all_cols)-1))
        
        v1 = c1.selectbox("Variable 1 (Exposure/Row):", all_cols, index=v1_idx, key='chi_v1_diag')
        v2 = c2.selectbox("Variable 2 (Outcome/Col):", all_cols, index=v2_idx, key='chi_v2_diag')
        
        correction_flag = c3.radio("Correction (2x2):", ['Pearson', "Yates'"], index=0, key='chi_corr_diag') == "Yates'"
        
        # 🟢 NEW: Positive Label Selectors
        st.markdown("---")
        st.caption("Select Positive Label for Risk/Odds Ratio calculation (default is '1'):")
        
        # Helper function to get unique values and set default index
        def get_pos_label_settings(df, col_name):
            unique_vals = [str(x) for x in df[col_name].dropna().unique()]
            unique_vals.sort()
            default_idx = 0
            if '1' in unique_vals:
                default_idx = unique_vals.index('1')
            return unique_vals, default_idx

        # Selector for V1 Positive Label
        c4, c5, c6 = st.columns(3)
        v1_uv, v1_default_idx = get_pos_label_settings(df, v1)
        v1_pos_label = c4.selectbox(f"Positive Label (Row: {v1}):", v1_uv, index=v1_default_idx, key='chi_v1_pos_diag')

        # Selector for V2 Positive Label (Outcome)
        v2_uv, v2_default_idx = get_pos_label_settings(df, v2)
        v2_pos_label = c5.selectbox(f"Positive Label (Col: {v2}):", v2_uv, index=v2_default_idx, key='chi_v2_pos_diag')

        # Add a placeholder column to maintain alignment
        c6.empty()
        st.markdown("---")
        
        run_col, dl_col = st.columns([1, 1])
        if 'html_output_chi' not in st.session_state: st.session_state.html_output_chi = None

        if run_col.button("Run Chi-Square", key='btn_chi_diag'):
           # 🟢 UPDATE 4: Pass new parameters to calculate_chi2
            tab, stats, msg, risk_df = diag_test.calculate_chi2(
                df, v1, v2, 
                correction=correction_flag,
                v1_pos=v1_pos_label, # <--- NEW PARAMETER
                v2_pos=v2_pos_label  # <--- NEW PARAMETER
            )

            tab, stats, msg, risk_df = diag_test.calculate_chi2(df, v1, v2, correction=correction_flag)
            
            if tab is not None:
                # 🟢 UPDATE 2: สร้าง Report โดยใช้ตัวแปรใหม่
                rep_elements = [
                    {'type': 'text', 'data': f"<b>Result:</b> {msg}"},
                    {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab, 'outcome_col': v2},
                    {'type': 'table', 'header': 'Statistics', 'data': pd.DataFrame([stats]).T}
                ]
                
                # 🟢 UPDATE 3: เพิ่ม Risk Table ถ้ามี
                if risk_df is not None:
                    rep_elements.append({'type': 'table', 'header': 'Risk & Effect Measures (2x2 Table)', 'data': risk_df})
                
                html = diag_test.generate_report(f"Chi2: {v1} vs {v2}", rep_elements)
                st.session_state.html_output_chi = html
                st.components.v1.html(html, height=600, scrolling=True)
            else: 
                # ถ้ามี error msg จะถูกส่งกลับมาใน msg ตัวแปรที่ 3
                st.error(msg)
        
        with dl_col:
            if st.session_state.html_output_chi:
                st.download_button("📥 Download Report", st.session_state.html_output_chi, "chi2_diag.html", "text/html", key='dl_chi_diag')
            else: st.button("📥 Download Report", disabled=True, key='ph_chi_diag')

    # --- Descriptive --- (ส่วนนี้เหมือนเดิม)
    with sub_tab3:
        st.markdown("##### Descriptive Statistics")
        st.info("""
            **💡 Guide:** Summarizes the distribution of a single variable.
            * **Numeric:** Mean, SD, Median, Min, Max, Quartiles.
            * **Categorical:** Frequency Counts and Percentages.
        """)
        dv = st.selectbox("Select Variable:", all_cols, key='desc_v_diag')
        run_col, dl_col = st.columns([1, 1])
        if 'html_output_desc' not in st.session_state: st.session_state.html_output_desc = None
        
        if run_col.button("Show Stats", key='btn_desc_diag'):
            res = diag_test.calculate_descriptive(df, dv)
            if res is not None:
                html = diag_test.generate_report(f"Descriptive: {dv}", [{'type':'table', 'data':res}])
                st.session_state.html_output_desc = html
                st.components.v1.html(html, height=500, scrolling=True)
        
        with dl_col:
            if st.session_state.html_output_desc:
                st.download_button("📥 Download Report", st.session_state.html_output_desc, "desc.html", "text/html", key='dl_desc_diag')
            else: st.button("📥 Download Report", disabled=True, key='ph_desc_diag')
