import streamlit as st
import pandas as pd
import numpy as np  # ✅ เพิ่มการ import numpy
import correlation # Import from root
import diag_test # Import for ICC calculation
from typing import List, Tuple

# 🟢 NEW: Helper function to select between original and matched datasets
def _get_dataset_for_correlation(df: pd.DataFrame):
    """
    Choose and return the dataset to use for correlation analysis (original or matched).
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

# ✅ ปรับแก้ให้รับ var_meta=None เพื่อรองรับการเรียกจาก app.py
def render(df, var_meta=None):
    """
    Render the "Correlation & ICC" user interface for exploring correlations and intraclass correlation coefficients.
    
    Parameters:
        df (pd.DataFrame): Source dataset used for analysis and selection between original/matched variants.
        var_meta (dict, optional): Mapping of column names to metadata (e.g., {'col': {'label': 'Friendly Name'}}). When provided, friendly labels from this mapping are used in selection controls and report headings.
    """
    # จัดการ var_meta ถ้าเป็น None ให้เป็น dict ว่าง
    if var_meta is None:
        var_meta = {}

    # Helper function เพื่อดึงชื่อ Label สวยๆ
    def get_label(col):
        return var_meta.get(col, {}).get('label', col)

    st.subheader("📈 Correlation & ICC")
    
    # 🟢 NEW: Display matched data status
    if st.session_state.get("is_matched", False):
        st.info("✅ **Matched Dataset Available** - You can select it below for analysis")
    
    # 🟢 NEW: Select dataset (original or matched)
    corr_df, corr_label = _get_dataset_for_correlation(df)
    st.write(f"**Using:** {corr_label}")
    st.write(f"**Rows:** {len(corr_df)} | **Columns:** {len(corr_df.columns)}")
    
    # 🟢 REORGANIZED: 3 subtabs
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📉 Pearson/Spearman (Continuous Correlation)", 
        "📏 Reliability (ICC)",
        "ℹ️ Reference & Interpretation"
    ])
    
    # ดึงเฉพาะคอลัมน์ที่เป็นตัวเลขจริงๆ เพื่อป้องกัน Error เวลาคำนวณ
    # ใช้ np.number ได้แล้วเพราะ import numpy as np แล้ว
    numeric_cols = corr_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns available for correlation analysis.")
        return

    # ==================================================
    # SUB-TAB 1: Pearson/Spearman (Continuous)
    # ==================================================
    with sub_tab1:
        st.markdown("##### Continuous Correlation Analysis")
        # 🧹 Removed detailed description (Moved to Tab 3)
        
        c1, c2, c3 = st.columns(3)
        cm = c1.selectbox("Correlation Coefficient:", ["Pearson", "Spearman"], key='coeff_type_tab')
        
        # 🟢 UPDATE: Auto-select default continuous variables
        cv1_default_name = 'Lab_HbA1c'
        cv2_default_name = 'Lab_Glucose'
        
        cv1_idx = next((i for i, c in enumerate(numeric_cols) if c == cv1_default_name), 0)
        cv2_idx = next((i for i, c in enumerate(numeric_cols) if c == cv2_default_name), min(1, len(numeric_cols)-1))
        
        # เพิ่ม format_func=get_label เพื่อแสดงชื่อไทย/ชื่อเต็ม
        cv1 = c2.selectbox("Variable 1 (X-axis):", numeric_cols, index=cv1_idx, key='cv1_corr_tab', format_func=get_label)
        cv2 = c3.selectbox("Variable 2 (Y-axis):", numeric_cols, index=cv2_idx, key='cv2_corr_tab', format_func=get_label)
        
        run_col_cont, dl_col_cont = st.columns([1, 1])
        if 'html_output_corr_cont' not in st.session_state: st.session_state.html_output_corr_cont = None

        if run_col_cont.button("📉 Analyze Correlation", type="primary", key='btn_run_cont'):
            if cv1 == cv2:
                st.error("Please select different variables.")
            else:
                m_key = 'pearson' if cm == 'Pearson' else 'spearman'
                # 🟢 UPDATED: Use corr_df (selected dataset)
                res, err, fig = correlation.calculate_correlation(corr_df, cv1, cv2, method=m_key)
        
                if err: 
                    st.error(err)
                else:
                    rep = [
                        {'type':'text', 'data':f"Method: {res['Method']}"},
                        {'type':'text', 'data':f"Variables: {get_label(cv1)} vs {get_label(cv2)}"},
                        {'type':'table', 'header':'Statistics', 'data':pd.DataFrame([res])}, 
                        {'type':'plot', 'header':'Scatter Plot', 'data':fig}
                    ]
                    html = correlation.generate_report(f"Corr: {get_label(cv1)} vs {get_label(cv2)}", rep)
                    st.session_state.html_output_corr_cont = html
                    st.components.v1.html(html, height=600, scrolling=True)

        with dl_col_cont:
            if st.session_state.html_output_corr_cont:
                st.download_button("📥 Download Report", st.session_state.html_output_corr_cont, "correlation_report.html", "text/html", key='dl_btn_corr_cont')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_btn_corr_cont')

    # ==================================================
    # SUB-TAB 2: Reliability (ICC)
    # ==================================================
    with sub_tab2:
        st.markdown("##### Reliability Analysis (ICC)")
        # 🧹 Removed detailed description (Moved to Tab 3)
        
        # Auto-select columns
        default_icc_cols = [c for c in numeric_cols if any(k in c.lower() for k in ['measure', 'machine', 'rater', 'read', 'icc'])]
        if len(default_icc_cols) < 2:
            default_icc_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else []
        
        icc_cols = st.multiselect(
            "Select Variables (Raters/Methods) - Select 2+ for ICC:", 
            numeric_cols, 
            default=default_icc_cols, 
            key='icc_vars_corr',
            format_func=get_label, # แสดงชื่อเต็ม
            help="Select 2 or more numeric columns representing different raters/methods measuring the same construct."
        )
        
        icc_run, icc_dl = st.columns([1, 1])
        if 'html_output_icc' not in st.session_state: 
            st.session_state.html_output_icc = None
        
        if icc_run.button("📏 Calculate ICC", type="primary", key='btn_icc_run', help="Calculates Intraclass Correlation Coefficient for reliability"):
            if len(icc_cols) < 2:
                st.error("❌ Please select at least 2 numeric columns for ICC calculation.")
                st.stop()
            
            # 🟢 UPDATED: Use corr_df
            res_df, err, anova_df = diag_test.calculate_icc(corr_df, icc_cols)
            
            if err:
                st.error(err)
            else:
                rep_elements = [
                    {'type': 'text', 'data': f"ICC Analysis: {', '.join([get_label(c) for c in icc_cols])}"},
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
    # SUB-TAB 3: Reference & Interpretation (Updated)
    # ==================================================
    with sub_tab3:
        st.markdown("## 📚 Reference & Interpretation Guide")
        
        st.info("💡 **Tip:** Use this tab to understand which test to choose and how to interpret the results.")
        
        col1, col2 = st.columns(2)
        
        # --- Column 1: Correlation ---
        with col1:
            st.markdown("### 📉 Correlation (Relationship)")
            st.markdown("""
            **Concept:** Measures the strength and direction of the relationship between **two continuous variables**.
            
            **1. Pearson (r):**
            * **Best for:** Linear relationships (straight line), normally distributed data.
            * **Sensitive to:** Outliers.
            
            **2. Spearman (rho):**
            * **Best for:** Monotonic relationships (variables increase together but not necessarily at a constant rate), non-normal data, or ranks.
            * **Robust to:** Outliers.
            
            **Interpretation of Coefficient (r or rho):**
            * **+1.0:** Perfect Positive (As X goes up, Y goes up).
            * **-1.0:** Perfect Negative (As X goes up, Y goes down).
            * **0.0:** No relationship.
            
            **Strength Guidelines:**
            * **0.7 - 1.0:** Strong
            * **0.4 - 0.7:** Moderate
            * **0.2 - 0.4:** Weak
            * **< 0.2:** Negligible
            """)

        # --- Column 2: ICC ---
        with col2:
            st.markdown("### 📏 ICC (Reliability)")
            st.markdown("""
            **Concept:** Measures the reliability or agreement between **two or more raters/methods** measuring the same thing. Unlike correlation, ICC accounts for systematic error (bias).
            
            **Common Types:**
            * **ICC(2,1) Absolute Agreement:** Use when you care if the **exact scores** match. (e.g., Doctor A gives 5, Doctor B must give 5).
            * **ICC(3,1) Consistency:** Use when you care if the **ranking** is consistent, even if scores differ. (e.g., Doctor A is consistently 1 point higher than Doctor B).
            
            **Interpretation of ICC Value:**
            * **> 0.90:** Excellent Reliability ✅
            * **0.75 - 0.90:** Good Reliability
            * **0.50 - 0.75:** Moderate Reliability ⚠️
            * **< 0.50:** Poor Reliability ❌
            """)
        
        st.markdown("---")
        st.markdown("### 📝 Common Questions")
        st.markdown("""
        **Q: Why use ICC instead of Pearson correlation for reliability?**
        * **A:** Pearson only measures linearity. If Rater A always gives a score exactly 10 points higher than Rater B, Pearson correlation will be 1.0 (perfect), but they don't agree! ICC accounts for this difference and would show lower agreement.
        
        **Q: What if my p-value is significant (< 0.05) but r is low (0.1)?**
        * **A:** A significant p-value just means the correlation is likely not zero. With large sample sizes, even tiny correlations can be "significant". **Focus on the r-value magnitude** for clinical relevance.
        """)
        
        st.caption("Note: Chi-Square test (for categorical association) can be found in **Tab 4: Diagnostic Tests**.")
