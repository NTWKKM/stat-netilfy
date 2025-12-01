import streamlit as st
import pandas as pd
import correlation # Import from root

def render(df):
    st.subheader("3. Correlation Analysis")
    
    sub_tab1, sub_tab2 = st.tabs([
        "🎲 Chi-Square & Risk-RR,OR,NNT (Categorical)", 
        "📈 Pearson/Spearman (Continuous)"
    ])
    
    all_cols = df.columns.tolist()

    # ==================================================
    # SUB-TAB 1: Chi-Square & Risk Measures (Categorical)
    # ==================================================
    with sub_tab1:
        st.markdown("##### Chi-Square Test & Risk Analysis")
        st.info("""
            **💡 Guide:** Used to analyze the association between **two categorical variables**.
            * **Chi-Square Test:** Determines if there is a significant association between the variables (P-value).
            * **Risk/Odds Ratio:** For **2x2 tables**, the tool provides **automatically calculated** metrics: **Risk Ratio (RR)**, **Odds Ratio (OR)**, and **Number Needed to Treat (NNT)**.
            
            **Variable Selection:**
            * **Variable 1 (Row):** Typically the **Exposure**, **Risk Factor**, or **Intervention**.
            * **Variable 2 (Column):** Typically the **Outcome** or **Event** of interest.
        """)

        cc1, cc2, cc3 = st.columns(3)
        
        # 🟢 UPDATE 1: Auto-select V1 and V2
        v1_default_name = 'Hypertension'
        v2_default_name = 'Outcome_Disease'
        
        v1_idx = next((i for i, c in enumerate(all_cols) if c == v1_default_name), 0)
        v2_idx = next((i for i, c in enumerate(all_cols) if c == v2_default_name), min(1, len(all_cols)-1))
        
        v1 = cc1.selectbox("Variable 1 (Exposure/Row):", all_cols, index=v1_idx, key='chi1_corr_tab') 
        v2 = cc2.selectbox("Variable 2 (Outcome/Col):", all_cols, index=v2_idx, key='chi2_corr_tab')
        
        correction_flag = cc3.radio("Correction Method (for 2x2):", 
                                    ['Pearson (Standard)', "Yates' correction"], 
                                    index=0, key='chi_corr_method_tab') == "Yates' correction"

        # 🟢 NEW: Positive Label Selectors
        st.markdown("---")
        st.caption("Select Positive Label for Risk/Odds Ratio calculation (default is '1'):")

        # Helper function to get unique values and set default index (Duplicated for tab_corr)
        def get_pos_label_settings(df, col_name):
            unique_vals = [str(x) for x in df[col_name].dropna().unique()]
            unique_vals.sort()
            default_idx = 0
            if '1' in unique_vals:
                default_idx = unique_vals.index('1')
            return unique_vals, default_idx

        # Selector for V1 Positive Label
        cc4, cc5, cc6 = st.columns(3)
        v1_uv, v1_default_idx = get_pos_label_settings(df, v1)
        v1_pos_label = cc4.selectbox(f"Positive Label (Row: {v1}):", v1_uv, index=v1_default_idx, key='chi_v1_pos_corr')

        # Selector for V2 Positive Label (Outcome)
        v2_uv, v2_default_idx = get_pos_label_settings(df, v2)
        v2_pos_label = cc5.selectbox(f"Positive Label (Col: {v2}):", v2_uv, index=v2_default_idx, key='chi_v2_pos_corr')
        
        # Add a placeholder column to maintain alignment
        cc6.empty()
        st.markdown("---")

        run_col, dl_col = st.columns([1, 1])
        if 'html_output_corr_cat' not in st.session_state: st.session_state.html_output_corr_cat = None

        if run_col.button("🚀 Run Analysis (Chi-Square)", key='btn_chi_run'):
            # 🟢 UPDATE 4: Pass new parameters to calculate_chi2
            tab, stats, msg, risk_df = correlation.calculate_chi2(
                df, v1, v2, 
                correction=correction_flag,
                v1_pos=v1_pos_label, # <--- NEW PARAMETER
                v2_pos=v2_pos_label  # <--- NEW PARAMETER
            )

            tab, stats, msg, risk_df = correlation.calculate_chi2(df, v1, v2, correction=correction_flag)
            
            if tab is not None:
                rep = [
                    {'type': 'text', 'data': f"<b>Analysis:</b> Chi-Square & Risk<br><b>Variables:</b> {v1} vs {v2}"},
                    {'type': 'text', 'data': f"<b>Main Result:</b> {msg}"},
                    
                    # Contingency Table
                    {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab, 'outcome_col': v2},
                    
                    # Statistics
                    {'type': 'table', 'header': 'Detailed Statistics', 'data': pd.DataFrame([stats]).T} 
                ]
                
                # 🟢 UPDATE 2: เพิ่ม Risk Table ลงใน Report ถ้ามีข้อมูล
                if risk_df is not None:
                    rep.append({'type': 'table', 'header': 'Risk & Effect Measures (2x2 Table)', 'data': risk_df})

                html = correlation.generate_report(f"Chi-square: {v1} vs {v2}", rep)
                
                st.session_state.html_output_corr_cat = html
                st.components.v1.html(html, height=600, scrolling=True)
            else:
                st.error(msg) 

        with dl_col:
            if st.session_state.html_output_corr_cat:
                st.download_button("📥 Download Report", st.session_state.html_output_corr_cat, "chi_risk_report.html", "text/html", key='dl_btn_corr_cat')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_btn_corr_cat')

    # ==================================================
    # SUB-TAB 2: Pearson/Spearman (Continuous)
    # ==================================================
    with sub_tab2:
        st.markdown("##### Continuous Correlation Analysis")
        st.info("""
    **💡 Guide:** Measures the relationship between **two continuous (numeric) variables**.

    * **Pearson (r):** Assesses **linear** correlation; best for normally distributed data.
    * **Spearman (rho):** Assesses **monotonic** (directional) correlation; best for non-normal data or ranks/outliers.

    **Interpretation of Coefficient (r/rho):**
    * **Close to +1:** Strong positive association (Both variables increase together).
    * **Close to -1:** Strong negative association (One increases as the other decreases).
    * **Close to 0:** Weak or no association.

    **X/Y Axis (for Plotting):**
    * The coefficient (r/rho) is **symmetrical** (X,Y is the same as Y,X).
    * For visual clarity, the **Predictor (Independent)** should be on the **X-axis** and the **Outcome (Dependent)** on the **Y-axis**.
        """)
        
        c1, c2, c3 = st.columns(3)
        cm = c1.selectbox("Correlation Coefficient:", ["Pearson", "Spearman"], key='coeff_type_tab')
        
        # 🟢 UPDATE: Auto-select BMI and Inflammation_Marker
        cv1_default_name = 'BMI'
        cv2_default_name = 'Inflammation_Marker'
        
        cv1_idx = next((i for i, c in enumerate(all_cols) if c == cv1_default_name), 0)
        cv2_idx = next((i for i, c in enumerate(all_cols) if c == cv2_default_name), min(1, len(all_cols)-1))
        
        cv1 = c2.selectbox("Variable 1 (X-axis):", all_cols, index=cv1_idx, key='cv1_corr_tab')
        cv2 = c3.selectbox("Variable 2 (Y-axis):", all_cols, index=cv2_idx, key='cv2_corr_tab')
        
        run_col_cont, dl_col_cont = st.columns([1, 1])
        if 'html_output_corr_cont' not in st.session_state: st.session_state.html_output_corr_cont = None

        if run_col_cont.button("📉 Analyze Correlation", key='btn_run_cont'):
            if cv1 == cv2:
                st.error("Please select different variables.")
            else:
                m_key = 'pearson' if cm == 'Pearson' else 'spearman'
                res, err, fig = correlation.calculate_correlation(df, cv1, cv2, method=m_key)
                
                if err: 
                    st.error(err)
                else:
                    rep = [
                        {'type':'text', 'data':f"Method: {res['Method']}<br>Variables: {cv1} vs {cv2}"},
                        {'type':'table', 'header':'Statistics', 'data':pd.DataFrame([res])},
                        {'type':'plot', 'header':'Scatter Plot', 'data':fig}
                    ]
                    html = correlation.generate_report(f"Corr: {cv1} vs {cv2}", rep)
                    st.session_state.html_output_corr_cont = html
                    st.components.v1.html(html, height=600, scrolling=True)

        with dl_col_cont:
            if st.session_state.html_output_corr_cont:
                st.download_button("📥 Download Report", st.session_state.html_output_corr_cont, "correlation_cont_report.html", "text/html", key='dl_btn_corr_cont')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_btn_corr_cont')
