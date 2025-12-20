import streamlit as st
import pandas as pd
import correlation # Import from root
import diag_test # Import for ICC calculation
from typing import List, Tuple

# 🟢 NEW: Helper function to select between original and matched datasets
def _get_dataset_for_correlation(df: pd.DataFrame):
    """
    Helper: เลือกระหว่าง original vs matched dataset สำหรับ correlation analysis
    คืนค่า: (selected_df, label_str)
    """
    has_matched = (
        st.session_state.get("is_matched", False)
        and st.session_state.get("df_matched") is not None
    )

    if has_matched:
        col1, _ = st.columns([2, 1])
        with col1:
            data_source = st.radio(
                "📄 Select Dataset:",
                ["📊 Original Data", "✅ Matched Data (from PSM)"],
                index=1,  # default Matched สำหรับ correlation analysis
                horizontal=True,
                key="correlation_data_source",
            )

        if "✅" in data_source:
            selected_df = st.session_state.df_matched.copy()
            label = f"✅ Matched Data ({len(selected_df)} rows)"
        else:
            selected_df = df
            label = f"📊 Original Data ({len(df)} rows)"
    else:
        selected_df = df
        label = f"📊 Original Data ({len(df)} rows)"

    return selected_df, label


def render(df):
    """
    Render the Correlation & ICC section UI in Streamlit.
    
    Displays three subtabs for continuous-variable analyses: (1) Pearson/Spearman correlation between two selected variables with a scatter plot and statistics, (2) Intraclass Correlation Coefficient (ICC) reliability analysis for 2+ numeric columns, and (3) a reference & interpretation guide. Generated HTML reports are stored in st.session_state under 'html_output_corr_cont' and 'html_output_icc'.
    
    Parameters:
        df (pandas.DataFrame): Input dataset whose columns populate selectors and whose data are used for the analyses.
    
    Side effects:
        Renders Streamlit controls, informational text, analysis results, and plots; writes HTML reports to st.session_state keys 'html_output_corr_cont' and 'html_output_icc'.
    """
    st.subheader("🎯 Correlation & ICC")
    
    # 🟢 NEW: Display matched data status
    if st.session_state.get("is_matched", False):
        st.info("✅ มี Matched dataset จาก PSM แล้ว สามารถเลือกใช้ด้านล่างได้")
    
    # 🟢 NEW: Select dataset (original or matched)
    corr_df, corr_label = _get_dataset_for_correlation(df)
    st.write(f"**Using:** {corr_label}")
    st.write(f"**Rows:** {len(corr_df)} | **Columns:** {len(corr_df.columns)}")
    
    # 🟢 REORGANIZED: 3 subtabs (removed Chi-Square, added ICC)
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📉 Pearson/Spearman (Continuous Correlation)", 
        "📏 Reliability (ICC)",
        "ℹ️ Reference & Interpretation"
    ])
    
    all_cols = corr_df.columns.tolist()
    if not all_cols:
        st.warning("No columns available for correlation analysis.")
        return

    # ==================================================
    # SUB-TAB 1: Pearson/Spearman (Continuous)
    # ==================================================
    with sub_tab1:
        st.markdown("##### Continuous Correlation Analysis (Pearson & Spearman)")
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
        
        # 🟢 UPDATE: Auto-select default continuous variables
        cv1_default_name = 'Lab_HbA1c'
        cv2_default_name = 'Lab_Glucose'
        
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
                # res: dict with keys (Method, Coefficient, P-value, N), err: str, fig: Plotly Figure
                # 🟢 UPDATED: Use corr_df (selected dataset) instead of df
                res, err, fig = correlation.calculate_correlation(corr_df, cv1, cv2, method=m_key)
        
                if err: 
                    st.error(err)
                else:
                    # 🟢 FIX: res is dict, convert to DataFrame for table display
                    rep = [
                        {'type':'text', 'data':f"Method: {res['Method']}"},
                        {'type':'text', 'data':f"Variables: {cv1} vs {cv2}"},
                        {'type':'table', 'header':'Statistics', 'data':pd.DataFrame([res])}, 
                        {'type':'plot', 'header':'Scatter Plot', 'data':fig}
                    ]
                    html = correlation.generate_report(f"Corr: {cv1} vs {cv2}", rep)
                    st.session_state.html_output_corr_cont = html
                    st.components.v1.html(html, height=600, scrolling=True)

        with dl_col_cont:
            if st.session_state.html_output_corr_cont:
                st.download_button("📥 Download Report", st.session_state.html_output_corr_cont, "correlation_report.html", "text/html", key='dl_btn_corr_cont')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_btn_corr_cont')

    # ==================================================
    # SUB-TAB 2: Reliability (ICC) - MOVED FROM Tab 4
    # ==================================================
    with sub_tab2:
        st.markdown("##### Reliability Analysis (Intraclass Correlation Coefficient - ICC)")
        st.info("""
            **💡 Guide:** Evaluates the reliability/agreement between 2 or more raters/methods for **Numeric/Continuous** variables.
            
            **ICC Types (Most Common):**
            * **ICC(2,1) Absolute Agreement:** Use when you care if the absolute scores are the same (e.g., Method A vs Method B must produce same values).
            * **ICC(3,1) Consistency:** Use when you care if the ranking is consistent, even if absolute scores differ (e.g., systematic bias OK, just need same ranking).
            
            **Interpretation of ICC:**
            * **< 0.5:** Poor reliability
            * **0.5 - 0.75:** Moderate reliability
            * **0.75 - 0.9:** Good reliability
            * **> 0.9:** Excellent reliability
        """)
        
        # 🟢 Auto-select numeric columns only (from corr_df)
        numeric_cols = corr_df.select_dtypes(include="number").columns.tolist()
        
        # Auto-select columns with 'measurement', 'rater', 'machine', 'score', 'read' in name
        default_icc_cols = [c for c in numeric_cols if any(k in c.lower() for k in ['measure', 'machine', 'rater', 'read', 'icc'])]
        if len(default_icc_cols) < 2:
            default_icc_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else []
        
        icc_cols = st.multiselect(
            "Select Variables (Raters/Methods) - Select 2+ for ICC:", 
            numeric_cols, 
            default=default_icc_cols, 
            key='icc_vars_corr',
            help="Select 2 or more numeric columns representing different raters/methods measuring the same construct."
        )
        
        icc_run, icc_dl = st.columns([1, 1])
        if 'html_output_icc' not in st.session_state: 
            st.session_state.html_output_icc = None
        
        if icc_run.button("📏 Calculate ICC", key='btn_icc_run', help="Calculates Intraclass Correlation Coefficient for reliability"):
            if len(icc_cols) < 2:
                st.error("❌ Please select at least 2 numeric columns for ICC calculation.")
                st.stop()
            # 🟢 UPDATED: Use corr_df (selected dataset) instead of df
            res_df, err, anova_df = diag_test.calculate_icc(corr_df, icc_cols)
            
            if err:
                st.error(err)
            else:
                rep_elements = [
                    {'type': 'text', 'data': f"ICC Analysis: {', '.join(icc_cols)}"},
                    {'type': 'table', 'header': 'ICC Results (Single Measures)', 'data': res_df},
                    {'type': 'table', 'header': 'ANOVA Table (Reference)', 'data': anova_df}
                ]
                html = diag_test.generate_report("ICC Reliability Analysis", rep_elements)
                st.session_state.html_output_icc = html
                st.components.v1.html(html, height=500, scrolling=True)
                
        with icc_dl:
            if st.session_state.html_output_icc:
                st.download_button("📥 Download Report", st.session_state.html_output_icc, "icc_report.html", "text/html", key='dl_icc_corr')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_icc_corr')

    # ==================================================
    # SUB-TAB 3: Reference & Interpretation
    # ==================================================
    with sub_tab3:
        st.markdown("##### Quick Reference: Correlation vs ICC")
        
        st.info("""
        **📊 When to Use What:**
        
        | Test | Variables | Purpose | Example |
        |------|-----------|---------|----------|
        | **Pearson** | 2 continuous | Linear relationship | Age vs Blood Pressure |
        | **Spearman** | 2 continuous | Monotonic relationship (rank-based) | Severity score vs Hospital Stay |
        | **ICC** | 2+ continuous | Reliability/agreement between raters/methods | Agreement between 2 doctors rating images |
        
        **Key Differences:**
        - **Correlation (r/rho)** = Association strength between TWO variables
        - **ICC** = Agreement strength between 2+ RATERS/METHODS measuring same thing
        """)
        
        st.markdown("### Interpretation Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Correlation Coefficient (r, rho)**")
            st.markdown("""
            - **|r| = 0.7 to 1.0** → Strong
            - **|r| = 0.4 to 0.7** → Moderate  
            - **|r| = 0.2 to 0.4** → Weak
            - **|r| < 0.2** → Very weak/negligible
            - **p < 0.05** → Statistically significant
            """)
        
        with col2:
            st.markdown("**ICC Value**")
            st.markdown("""
            - **0.90 to 1.00** → Excellent
            - **0.75 to 0.89** → Good
            - **0.50 to 0.74** → Moderate
            - **0.25 to 0.49** → Fair
            - **< 0.25** → Poor
            """)
        
        st.markdown("""---
        **📝 Note:** Chi-Square test (for categorical association) has been moved to **Tab 4: Diagnostic Tests (ROC)**.
        
        **✨ NEW:** Can now analyze both Original and Matched datasets!
        """)
