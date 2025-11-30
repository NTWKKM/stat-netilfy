import streamlit as st
import pandas as pd
import diag_test # Import from root

def render(df, var_meta):
    st.subheader("2. Diagnostic Test & Statistics")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📈 ROC Curve & AUC", "🎲 Chi-Square", "📊 Descriptive"])
    all_cols = df.columns.tolist()

    # --- ROC ---
    with sub_tab1:
        st.markdown("##### ROC Curve Analysis")
        st.info("""
            **💡 Guide:** Evaluates the performance of a **continuous diagnostic test** against a binary **Gold Standard**.
            * **AUC (Area Under Curve):** Measures overall performance (0.5 = random, 1.0 = perfect).
            * **Youden Index:** Identifies the optimal cut-off point maximizing Sensitivity + Specificity.
            * **P-value:** Tests if AUC is significantly different from 0.5.
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
            pos_label = rc4.selectbox("Positive Label (1):", sorted_vals, key='roc_pos_diag')
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
            **💡 Guide:** Analyzes the association between **two categorical variables**.
            * **Chi-Square Test:** P-value for association.
            * **Risk Estimates (2x2):** Calculates **RR**, **OR**, **NNT** automatically for 2x2 tables.
            
            **Selection:**
            * **Variable 1:** Exposure / Treatment / Group.
            * **Variable 2:** Outcome / Event.
        """)

        c1, c2, c3 = st.columns(3)
        v1 = c1.selectbox("Variable 1 (Exposure/Row):", all_cols, key='chi_v1_diag')
        v2 = c2.selectbox("Variable 2 (Outcome/Col):", all_cols, index=min(1,len(all_cols)-1), key='chi_v2_diag')
        
        correction_flag = c3.radio("Correction (2x2):", ['Pearson', "Yates'"], index=0, key='chi_corr_diag') == "Yates'"

        run_col, dl_col = st.columns([1, 1])
        if 'html_output_chi' not in st.session_state: st.session_state.html_output_chi = None

        if run_col.button("Run Chi-Square", key='btn_chi_diag'):
            # diag_test.calculate_chi2 returns (display_tab, results)
            tab, results = diag_test.calculate_chi2(df, v1, v2, correction=correction_flag)
            
            if tab is not None and 'error' not in results:
                # Prepare elements for generate_report
                # Note: 'results' dict contains 'chi2_msg', 'RR', etc.
                
                # Check formatting of display table
                # The returned tab is already formatted with counts/% from diag_test.py
                
                # Create stats dataframe for report
                # We need to extract relevant scalar stats from 'results' dict
                stats_data = {k: v for k, v in results.items() if k not in ['chi2_msg', 'R_exp_label', 'R_unexp_label', 'Event_label', 'Is_2x2', 'R_exp', 'R_unexp']}
                
                rep_elements = [
                    {'type': 'text', 'data': f"<b>Result:</b> {results.get('chi2_msg', '')}"},
                    # Use the new contingency_table type provided in diag_test.generate_report
                    {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab, 'outcome_col': v2},
                    {'type': 'table', 'header': 'Statistics', 'data': pd.DataFrame([stats_data]).T}
                ]
                
                html = diag_test.generate_report(f"Chi2: {v1} vs {v2}", rep_elements)
                st.session_state.html_output_chi = html
                st.components.v1.html(html, height=600, scrolling=True)
            else: 
                st.error(results.get('error', 'An error occurred'))
        
        with dl_col:
            if st.session_state.html_output_chi:
                st.download_button("📥 Download Report", st.session_state.html_output_chi, "chi2_diag.html", "text/html", key='dl_chi_diag')
            else: st.button("📥 Download Report", disabled=True, key='ph_chi_diag')

    # --- Descriptive ---
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
