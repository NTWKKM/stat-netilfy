import streamlit as st
import pandas as pd
import correlation # Import from root

def render(df):
    st.subheader("3. Correlation Analysis")
    
    sub_tab1, sub_tab2 = st.tabs([
        "🎲 Chi-Square & Risk (Categorical)", 
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
            * **Risk/Odds Ratio: calculates **Risk Ratio (RR)**, **Odds Ratio (OR)**, and **Number Needed to Treat (NNT)** automatically ** For **2x2 tables.**
            
            **Variable Selection:**
            * **Variable 1 (Row):** Typically the **Exposure**, **Risk Factor**, or **Intervention**.
            * **Variable 2 (Column):** Typically the **Outcome** or **Event** of interest.
        """)

        cc1, cc2, cc3 = st.columns(3)
        v1 = cc1.selectbox("Variable 1 (Exposure/Row):", all_cols, key='chi1_corr_tab') 
        v2 = cc2.selectbox("Variable 2 (Outcome/Col):", all_cols, index=min(1, len(all_cols)-1), key='chi2_corr_tab')
        
        correction_flag = cc3.radio("Correction Method (for 2x2):", 
                                    ['Pearson (Standard)', "Yates' correction"], 
                                    index=0, key='chi_corr_method_tab') == "Yates' correction"

        run_col, dl_col = st.columns([1, 1])
        if 'html_output_corr_cat' not in st.session_state: st.session_state.html_output_corr_cat = None

        if run_col.button("🚀 Run Analysis (Chi-Square)", key='btn_chi_run'):
            # 🟢 UPDATE 1: รับค่า 4 ตัว (เพิ่ม risk_df)
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
    # (ส่วนนี้เหมือนเดิม ไม่ต้องแก้)
    with sub_tab2:
        st.markdown("##### Continuous Correlation Analysis")
        st.info("""
            **💡 Guide:** Used to measure the relationship between **two continuous (numeric) variables**.
            * **Pearson (r):** Best for data that follows a normal distribution (Linear Relationship).
            * **Spearman (rho):** Best for non-normal data, ordinal data, or outliers (Monotonic Relationship).
             **Interpretation of Coefficient:**             * **Close to +1:** Strong positive relationship (Both increase together).
            * **Close to -1:** Strong negative relationship (One increases, the other decreases).
            * **Close to 0:** No relationship.
        """)
        
        c1, c2, c3 = st.columns(3)
        cm = c1.selectbox("Correlation Coefficient:", ["Pearson", "Spearman"], key='coeff_type_tab')
        cv1 = c2.selectbox("Variable 1 (X-axis):", all_cols, key='cv1_corr_tab')
        cv2 = c3.selectbox("Variable 2 (Y-axis):", all_cols, index=min(1,len(all_cols)-1), key='cv2_corr_tab')
        
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
