import streamlit as st
import pandas as pd
import diag_test # Import จาก root

def render(df, var_meta):
    st.subheader("2. Diagnostic Test & Statistics")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📈 ROC Curve & AUC", "🎲 Chi-Square", "📊 Descriptive"])
    all_cols = df.columns.tolist()

    # --- ROC ---
    with sub_tab1:
        st.markdown("##### ROC Curve Analysis")
        rc1, rc2, rc3, rc4 = st.columns(4)
        
        def_idx = 0
        for i, c in enumerate(all_cols):
            if 'outcome' in c.lower() or 'died' in c.lower(): def_idx = i; break
        
        truth = rc1.selectbox("Gold Standard (Binary):", all_cols, index=def_idx, key='roc_truth')
        
        score_idx = 0
        for i, c in enumerate(all_cols):
            if 'score' in c.lower(): score_idx = i; break
        score = rc2.selectbox("Test Score (Continuous):", all_cols, index=score_idx, key='roc_score')
        
        method = rc3.radio("CI Method:", ["DeLong et al.", "Binomial (Hanley)"])

        # Positive Label
        pos_label = None
        unique_vals = df[truth].dropna().unique()
        if len(unique_vals) == 2:
            sorted_vals = sorted([str(x) for x in unique_vals])
            pos_label = rc4.selectbox("Positive Label (1):", sorted_vals, key='roc_pos')
        elif len(unique_vals) != 2:
            rc4.warning("Requires 2 unique values.")

        run_col, dl_col = st.columns([1, 1])
        if 'html_output_roc' not in st.session_state: st.session_state.html_output_roc = None
        
        if run_col.button("📉 Analyze ROC", key='btn_roc'):
            if pos_label and len(unique_vals) == 2:
                res, err, fig, coords_df = diag_test.analyze_roc(df, truth, score, 'delong' if 'DeLong' in method else 'hanley', pos_label_user=pos_label)
                if err: st.error(err)
                else:
                    rep = [
                        {'type':'text', 'data':f"Analysis: <b>{score}</b> vs <b>{truth}</b>"},
                        {'type':'plot', 'data':fig},
                        {'type':'table', 'header':'Statistics', 'data':pd.DataFrame([res]).T},
                        {'type':'table', 'header':'Performance', 'data':coords_df}
                    ]
                    html = diag_test.generate_report(f"ROC: {score}", rep)
                    st.session_state.html_output_roc = html
                    st.components.v1.html(html, height=800, scrolling=True)
            else:
                st.error("Invalid Target configuration.")

        with dl_col:
            if st.session_state.html_output_roc:
                st.download_button("📥 Download Report", st.session_state.html_output_roc, "roc_report.html", "text/html", key='dl_roc')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_roc')

    # --- Chi-Square ---
    with sub_tab2:
        st.markdown("##### Chi-Square Test")
        c1, c2 = st.columns(2)
        v1 = c1.selectbox("Var 1:", all_cols, key='chi_v1')
        v2 = c2.selectbox("Var 2:", all_cols, index=min(1,len(all_cols)-1), key='chi_v2')
        
        run_col, dl_col = st.columns([1, 1])
        if 'html_output_chi' not in st.session_state: st.session_state.html_output_chi = None

        if run_col.button("Run Chi-Square", key='btn_chi'):
            tab, msg = diag_test.calculate_chi2(df, v1, v2)
            if tab is not None:
                rep = [
                    {'type':'text', 'data':f"Result: {msg}"},
                    {'type':'table', 'header':'Contingency Table', 'data':tab.reset_index()}
                ]
                html = diag_test.generate_report(f"Chi2: {v1} vs {v2}", rep)
                st.session_state.html_output_chi = html
                st.components.v1.html(html, height=500, scrolling=True)
            else: st.error(msg)
        
        with dl_col:
            if st.session_state.html_output_chi:
                st.download_button("📥 Download Report", st.session_state.html_output_chi, "chi2.html", "text/html", key='dl_chi')
            else: st.button("📥 Download Report", disabled=True, key='ph_chi')

    # --- Descriptive ---
    with sub_tab3:
        st.markdown("##### Descriptive Stats")
        dv = st.selectbox("Select Variable:", all_cols, key='desc_v')
        run_col, dl_col = st.columns([1, 1])
        if 'html_output_desc' not in st.session_state: st.session_state.html_output_desc = None
        
        if run_col.button("Show Stats", key='btn_desc'):
            res = diag_test.calculate_descriptive(df, dv)
            if res is not None:
                html = diag_test.generate_report(f"Descriptive: {dv}", [{'type':'table', 'data':res}])
                st.session_state.html_output_desc = html
                st.components.v1.html(html, height=500, scrolling=True)
        
        with dl_col:
            if st.session_state.html_output_desc:
                st.download_button("📥 Download Report", st.session_state.html_output_desc, "desc.html", "text/html", key='dl_desc')
            else: st.button("📥 Download Report", disabled=True, key='ph_desc')
