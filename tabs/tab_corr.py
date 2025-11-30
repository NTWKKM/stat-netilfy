import streamlit as st
import pandas as pd
import correlation # Import จาก root

def render(df):
    st.subheader("3. Correlation Analysis")
    st.markdown("""
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 20px;">
            <b>Methods:</b> Chi-Square (Categorical), Pearson (Linear), Spearman (Monotonic)
        </div>
    """, unsafe_allow_html=True)

    all_cols = df.columns.tolist()
    c_m, c_v1, c_v2 = st.columns(3)
    
    method = c_m.selectbox("Method:", ["Chi-Square", "Pearson Correlation", "Spearman Correlation"])
    v1 = c_v1.selectbox("Variable 1:", all_cols, key='corr_v1')
    v2 = c_v2.selectbox("Variable 2:", all_cols, index=min(1,len(all_cols)-1), key='corr_v2')
    
    use_yates = True
    if method == "Chi-Square":
        use_yates = st.checkbox("Yates' Correction (for 2x2)", value=True)

    run_col, dl_col = st.columns([1, 1])
    if 'html_output_corr' not in st.session_state: st.session_state.html_output_corr = None

    if run_col.button("📉 Analyze Correlation", key='btn_corr'):
        if v1 == v2:
            st.error("Select different variables.")
        else:
            try:
                html = ""
                if method == "Chi-Square":
                    tab, stats, msg = correlation.calculate_chi2(df, v1, v2, correction=use_yates)
                    if tab is not None:
                        st.write(f"**Result:** {msg}")
                        st.table(tab)
                        rep = [
                            {'type':'text', 'data':f"Method: Chi-Square<br>Result: {msg}"},
                            {'type':'table', 'header':'Contingency Table', 'data':tab.reset_index()},
                            {'type':'table', 'header':'Stats', 'data':pd.DataFrame([stats])}
                        ]
                        html = correlation.generate_report(f"Corr: {v1} vs {v2}", rep)
                    else: st.error(msg)
                else:
                    # Pearson/Spearman
                    m_key = 'pearson' if 'Pearson' in method else 'spearman'
                    res, err, fig = correlation.calculate_correlation(df, v1, v2, method=m_key)
                    if err: st.error(err)
                    else:
                        st.write(f"**{res['Method']}:** {res['Coefficient']:.4f} (p={res['P-value']:.4f})")
                        st.pyplot(fig)
                        rep = [
                            {'type':'text', 'data':f"Method: {res['Method']}"},
                            {'type':'table', 'header':'Statistics', 'data':pd.DataFrame([res])},
                            {'type':'plot', 'header':'Plot', 'data':fig}
                        ]
                        html = correlation.generate_report(f"Corr: {v1} vs {v2}", rep)
                
                if html:
                    st.session_state.html_output_corr = html
                    st.components.v1.html(html, height=500, scrolling=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with dl_col:
        if st.session_state.html_output_corr:
            st.download_button("📥 Download Report", st.session_state.html_output_corr, "corr_report.html", "text/html", key='dl_corr')
        else: st.button("📥 Download Report", disabled=True, key='ph_corr')
