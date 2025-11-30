import streamlit as st
import pandas as pd
import correlation # Import จาก root

def render(df):
    st.subheader("3. Correlation Analysis")
    
    # Method Selector
    method = st.radio("Select Analysis Method:", ["Chi-Square Test", "Correlation (Pearson/Spearman)"], horizontal=True)
    all_cols = df.columns.tolist()

    if 'Chi-Square' in method:
        # --- UI ส่วน Chi-Square ที่คุณต้องการ ---
        st.markdown("##### Chi-Square Test & Risk Measures")
        st.markdown("""
            <p>
            The **Chi-Square Test** examines the association between two categorical variables. For $2\\times2$ tables, the analysis automatically includes **Risk Ratio (RR)**, **Risk Difference (RD)**, and **Number Needed to Treat (NNT)**.
            </p>
            <p>
            **Variable 1** is interpreted as the **Exposure/Treatment (Row)**, and **Variable 2** as the **Outcome/Event (Column)**.
            </p>
        """, unsafe_allow_html=True)

        # 🟢 3 Columns Layout
        cc1, cc2, cc3 = st.columns(3)
        
        v1 = cc1.selectbox("Variable 1 (Exposure/Group):", all_cols, key='chi1')
        v2 = cc2.selectbox("Variable 2 (Outcome/Event):", all_cols, index=min(1, len(all_cols)-1), key='chi2')
        
        # Correction Selection
        correction_flag = cc3.radio("Correction Method (for 2x2 table):", 
                                    ['Pearson (Standard)', "Yates' correction"], 
                                    index=0, key='chi_corr_method') == "Yates' correction"

        run_col_chi, download_col_chi = st.columns([1, 1])
        if 'html_output_corr' not in st.session_state: st.session_state.html_output_corr = None

        if run_col_chi.button("Run Chi-Square", key='btn_chi'):
            # 🟢 เรียกใช้ correlation.calculate_chi2 (ไม่ใช่ diag_test)
            tab, stats, msg = correlation.calculate_chi2(df, v1, v2, correction=correction_flag)
            
            if tab is not None:
                # Display Result
                st.success(f"Result: {msg}")
                st.table(tab)
                
                # Prepare Report
                display_tab = tab.reset_index()
                rep = [
                    {'type': 'text', 'data': f"<b>Chi-square Test Result:</b> {msg}"},
                    {'type': 'table', 'header': 'Contingency Table', 'data': display_tab},
                    {'type': 'table', 'header': 'Statistics', 'data': pd.DataFrame([stats])}
                ]
                html = correlation.generate_report(f"Chi-square: {v1} vs {v2}", rep)
                
                st.session_state.html_output_corr = html
                st.components.v1.html(html, height=500, scrolling=True)
            else:
                st.error(msg) # msg acts as error message here

    else:
        # --- UI ส่วน Pearson/Spearman (แบบเดิม) ---
        st.markdown("##### Continuous Correlation (Pearson/Spearman)")
        c1, c2, c3 = st.columns(3)
        cm = c1.selectbox("Coefficient:", ["Pearson", "Spearman"])
        cv1 = c2.selectbox("Variable 1:", all_cols, key='corr_v1')
        cv2 = c3.selectbox("Variable 2:", all_cols, index=min(1,len(all_cols)-1), key='corr_v2')
        
        run_col, dl_col = st.columns([1, 1])
        if 'html_output_corr' not in st.session_state: st.session_state.html_output_corr = None

        if run_col.button("📉 Analyze Correlation", key='btn_corr'):
            if cv1 == cv2:
                st.error("Select different variables.")
            else:
                m_key = 'pearson' if cm == 'Pearson' else 'spearman'
                res, err, fig = correlation.calculate_correlation(df, cv1, cv2, method=m_key)
                
                if err: 
                    st.error(err)
                else:
                    st.write(f"**{res['Method']}:** {res['Coefficient']:.4f} (p={res['P-value']:.4f})")
                    st.pyplot(fig)
                    rep = [
                        {'type':'text', 'data':f"Method: {res['Method']}"},
                        {'type':'table', 'header':'Statistics', 'data':pd.DataFrame([res])},
                        {'type':'plot', 'header':'Plot', 'data':fig}
                    ]
                    html = correlation.generate_report(f"Corr: {cv1} vs {cv2}", rep)
                    st.session_state.html_output_corr = html
                    st.components.v1.html(html, height=500, scrolling=True)

    # --- ส่วน Download Button (ใช้ร่วมกัน) ---
    with download_col_chi if 'Chi-Square' in method else dl_col:
        if st.session_state.html_output_corr:
            st.download_button("📥 Download Report", st.session_state.html_output_corr, "correlation_report.html", "text/html", key='dl_corr_common')
        else:
            st.button("📥 Download Report", disabled=True, key='ph_corr_common')
