import streamlit as st
import pandas as pd
import diag_test # ✅ ใช้ diag_test ตัวเดียว
from typing import List, Tuple

def render(df, _var_meta=None):  # var_meta reserved for future use
    """
    Render Streamlit UI panels for diagnostic tests and statistics.
    
    Displays five interactive tabs for diagnostic analyses:
    - ROC Curve & AUC
    - Chi-Square & Risk (2x2) with RR/OR/NNT - THE CHI-SQUARE HOME
    - Agreement (Cohen's Kappa)
    - Descriptive statistics
    - Reference & Interpretation
    
    Each tab provides controls for selecting columns from `df`, running the analysis, viewing results as embedded HTML, and downloading an HTML report. Generated report HTML is stored in Streamlit session state under keys: `html_output_roc`, `html_output_chi`, `html_output_kappa`, and `html_output_desc`.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing the variables to analyze; column names are used for UI selections.
        _var_meta (Any): Metadata about variables (unused for visible output selection unless integrated by UI); present for potential future use.
    """
    st.subheader("🧪 Diagnostic Tests (ROC)")
    
    # 🟢 IMPORTANT: Now 5 subtabs (added Reference & Interpretation)
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "📈 ROC Curve & AUC", 
        "🎲 Chi-Square & Risk Analysis (2x2)", 
        "🤝 Agreement (Kappa)", 
        "📊 Descriptive",
        "ℹ️ Reference & Interpretation"
    ])
    
    all_cols = df.columns.tolist()
    if not all_cols:
        st.error("Dataset has no columns to analyze.")
        return
        
    # --- ROC ---
    with sub_tab1:
        st.markdown("##### ROC Curve Analysis")
        st.info("""
    **💡 Guide:** Evaluates the performance of a **continuous diagnostic test** (e.g., 'lab value' or 'risk index') against a **binary Gold Standard** (e.g., 'disease' or 'no disease').

    * **AUC (Area Under Curve):** Measures overall test discrimination ability (0.5 = random guess, 1.0 = perfect test).
    * **Youden Index (J):** Identifies the **optimal cut-off point** by maximizing the difference between Sensitivity and (1 - Specificity).
    * **P-value:** Tests the null hypothesis that the AUC is equal to 0.5 (i.e., the test performs better than chance).
        """)
        
        rc1, rc2, rc3, rc4 = st.columns(4)
        
        def_idx = 0
        for i, c in enumerate(all_cols):
            cl = c.lower()
            if "gold" in cl or "standard" in cl:
                def_idx = i
                break
        
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
                        {'type':'text', 'data':f"Analysis: {score} vs {truth}"},
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

    # --- Chi-Square & Risk Analysis (2x2) ---
    with sub_tab2:
        st.markdown("##### 🎲 Chi-Square & Risk Analysis (2x2 Contingency Table)")
        st.info("""
            **💡 Guide - THE HOME OF CHI-SQUARE ANALYSIS:** Used to analyze the association between **two categorical variables**.
            
            ### 📊 What You Get from 2x2 Tables:
            
            **Association Test:**
            * **Chi-Square Test:** Determines if there is a significant association between the variables (P-value).
            * **Method Options:** Pearson (standard), Yates' correction (conservative), Fisher's Exact Test (for small samples)
            
            **Effect/Risk Metrics (automatically calculated for 2x2):**
            * **Odds Ratio (OR):** Odds of outcome in exposed vs unexposed groups
            * **Risk Ratio (RR):** Risk of outcome in exposed vs unexposed (for cohort studies)
            * **Number Needed to Treat (NNT):** How many patients to treat to prevent 1 outcome
            * **Confidence Intervals (95% CI):** For all metrics
            
            **Diagnostic Metrics (if applicable context):**
            * **Sensitivity/Specificity:** Test accuracy
            * **PPV/NPV:** Predictive values
            * **LR+/LR-:** Likelihood ratios
            
            **Variable Selection:**
            * **Variable 1 (Row):** Typically the **Exposure**, **Risk Factor**, **Intervention**, or **Test Result**
            * **Variable 2 (Column):** Typically the **Outcome**, **Event**, **Gold Standard**, or **Disease Status**
        """)

        c1, c2, c3 = st.columns(3)
        
        # Auto-select V1 and V2
        v1_default_name = 'Treatment_Group'
        v2_default_name = 'Outcome_Cured'
        v1_idx = next((i for i, c in enumerate(all_cols) if c == v1_default_name), 0)
        v2_idx = next((i for i, c in enumerate(all_cols) if c == v2_default_name), min(1, len(all_cols)-1))
        
        v1 = c1.selectbox("Variable 1 (Exposure/Row):", all_cols, index=v1_idx, key='chi_v1_diag')
        v2 = c2.selectbox("Variable 2 (Outcome/Col):", all_cols, index=v2_idx, key='chi_v2_diag')
        
        method_choice = c3.radio(
            "Test Method (for 2x2):", 
            ['Pearson (Standard)', "Yates' correction", "Fisher's Exact Test"], 
            index=0, 
            key='chi_corr_method_diag',
            help="Pearson: Best for large samples. Yates: Conservative correction. Fisher: Exact test, MUST use if any expected count < 5."
        )
        
        # Positive Label Selectors
        def get_pos_label_settings(df: pd.DataFrame, col_name: str) -> Tuple[List[str], int]:
            """
            Return sorted non-null unique string values from a DataFrame column and a sensible default selection index.
            
            Parameters:
                df (pd.DataFrame): DataFrame containing the column.
                col_name (str): Name of the column to extract values from.
            
            Returns:
                tuple(list[str], int): A tuple where the first element is a sorted list of the column's unique non-null values as strings, and the second element is the default index to select (index of '1' if present, otherwise index of '0' if present, otherwise 0).
            """
            # 🟢 NOTE: Need to handle the case where the column might be empty after dropna
            # Convert to string and drop NA values before getting unique values
            unique_vals = [str(x) for x in df[col_name].dropna().unique()]
            unique_vals.sort()
    
            default_idx = 0
            if '1' in unique_vals:
                # Default to '1' if available
                default_idx = unique_vals.index('1')
            elif len(unique_vals) > 0 and '0' in unique_vals:
                # Otherwise, default to '0' if available and there are unique values
                default_idx = unique_vals.index('0')
        
            return unique_vals, default_idx

        c4, c5, c6 = st.columns(3)
        v1_uv, v1_default_idx = get_pos_label_settings(df, v1)
        if not v1_uv:
            c4.warning(f"No non-null values in {v1}.")
            v1_pos_label = None
        else:
            v1_pos_label = c4.selectbox(
                f"Positive Label (Row: {v1}):",
                v1_uv,
                index=v1_default_idx,
                key='chi_v1_pos_diag',
            )

        v2_uv, v2_default_idx = get_pos_label_settings(df, v2)
        if not v2_uv:
            c5.warning(f"No non-null values in {v2}.")
            v2_pos_label = None
        else:
            v2_pos_label = c5.selectbox(
                f"Positive Label (Col: {v2}):",
                v2_uv,
                index=v2_default_idx,
                key='chi_v2_pos_diag',
            )
        
        # 🛑 จุดที่เพิ่ม 1: ถ้าเลือกไม่ได้ (เป็น None) ให้หยุดทำงาน อย่าฝืนคำนวณต่อ
        inputs_ok = not (v1_pos_label is None or v2_pos_label is None)
        if not inputs_ok:
            st.warning("Chi-Square disabled: one of the selected columns has no non-null values.")

        c6.empty()
        st.caption("Select Positive Label for Risk/Odds Ratio calculation (default is '1'):")
        
        run_col, dl_col = st.columns([1, 1])
        
        if 'html_output_chi' not in st.session_state: 
            st.session_state.html_output_chi = None

        if run_col.button("🚀 Run Analysis (Chi-Square)", key='btn_chi_run_diag', disabled=not inputs_ok):
            
            # --- 🟢 จุดที่ 3 เพิ่มตรงนี้ครับ ---
            # CodeRabbit เตือนว่า selectbox คืนค่าเป็น String (เช่น "1") 
            # แต่ในตารางอาจเป็น Int (เช่น 1) ทำให้เทียบกันไม่ติด
            # เราจึงต้องสร้างตารางจำลอง (df_calc) และแปลงข้อมูลเป็น String ก่อนส่งไปคำนวณ
            df_calc = df.copy()
            df_calc[v1] = df_calc[v1].astype("string")
            df_calc[v2] = df_calc[v2].astype("string")
            # --------------------------------

            # ⚠️ อย่าลืมเปลี่ยน parameter ตัวแรกจาก df เป็น df_calc ด้วยนะครับ
            tab, stats, msg, risk_df = diag_test.calculate_chi2(
                df_calc, v1, v2,  # <--- เปลี่ยนตรงนี้เป็น df_calc
                method=method_choice,
                v1_pos=v1_pos_label,
                v2_pos=v2_pos_label
            )
            
            if tab is not None:
                # 🟢 UPDATE 1: จัดการข้อความสถานะ/หมายเหตุ (Warning/Note)
                # เมื่อ tab is not None, msg จะมีแค่ข้อความเตือน (Warning/Note) หรือเป็นสตริงว่างเปล่าเท่านั้น
                if msg.strip():
                    status_text = f"Note: {msg.strip()}"
                else:
                    status_text = "Analysis Status: Completed successfully."
                
                # 🟢 FIX: แยกข้อความและลบแท็ก HTML (<b>, <br>) ออก
                # และใช้ตัวแปร rep_elements ให้สอดคล้องกับโค้ดส่วนอื่น
                rep_elements = [ 
                    {'type': 'text', 'data': "Analysis: Diagnostic Test / Chi-Square"},
                    {'type': 'text', 'data': f"Variables: {v1} vs {v2}"},
                    {'type': 'text', 'data': status_text}, # แสดงสถานะหรือข้อความเตือน
                    
                    # Contingency Table
                    {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab, 'outcome_col': v2},
                ]
                
                # 🟢 NOTE: ใช้โครงสร้างเดิมในการเพิ่ม Statistics และ Risk/Effect Measures
                if stats is not None:
                    # เดิม: rep_elements.append({'type': 'table', 'header': 'Statistics', 'data': stats})
                    rep_elements.append({'type': 'table', 'header': 'Statistics', 'data': stats})
                if risk_df is not None:
                    rep_elements.append({'type': 'table', 'header': 'Risk & Effect Measures (2x2 Table)', 'data': risk_df})
                
                html = diag_test.generate_report(f"Chi2: {v1} vs {v2}", rep_elements)
                st.session_state.html_output_chi = html
                st.components.v1.html(html, height=600, scrolling=True)
            else: 
                # กรณีนี้ msg คือ Fatal Error ซึ่งจะถูกแสดงด้วย Streamlit error
                st.error(msg)
        
        with dl_col:
            if st.session_state.html_output_chi:
                st.download_button("📥 Download Report", st.session_state.html_output_chi, "chi2_diag.html", "text/html", key='dl_chi_diag')
            else: 
                st.button("📥 Download Report", disabled=True, key='ph_chi_diag')
       
    # --- Agreement (Kappa) ---
    with sub_tab3:
        st.markdown("##### Agreement Analysis (Cohen's Kappa)")
        st.info("""
             **💡 Guide:** Evaluates the **agreement** between two raters or two methods classifying items into categories.
             * **Cohen's Kappa (κ):** Measures agreement adjusting for chance. 
             * **Interpretation:** 
                 * < 0: Poor
                 * 0.01 - 0.20: Slight
                 * 0.21 - 0.40: Fair
                 * 0.41 - 0.60: Moderate
                 * 0.61 - 0.80: Substantial
                 * 0.81 - 1.00: Perfect
         """)
        
        # 🟢 เพิ่ม Logic การเลือกอัตโนมัติ (Auto-select)
        # all_cols already defined above in render()
        
        # 1. Logic หาคอลัมน์ที่น่าจะเป็น Rater A
        kv1_default_idx = 0
        kv2_default_idx = min(1, len(all_cols) - 1)
        
        # --- เริ่มค้นหา Rater A ---
        for i, col in enumerate(all_cols):
            # ตรวจสอบชื่อที่มีคำว่า 'Dr_A', 'Rater_1', 'Diagnosis_A'
            if 'dr_a' in col.lower() or 'rater_1' in col.lower() or 'diagnosis_a' in col.lower():
                kv1_default_idx = i
                break
        
        # --- เริ่มค้นหา Rater B ---
        for i, col in enumerate(all_cols):
            if 'dr_b' in col.lower() or 'rater_2' in col.lower() or 'diagnosis_b' in col.lower():
                kv2_default_idx = i
                break

        # ป้องกันไม่ให้ Rater 1 และ Rater 2 เป็นคอลัมน์เดียวกัน ถ้าเจอชื่อที่ตรงกัน
        if kv1_default_idx == kv2_default_idx and len(all_cols) > 1:
            # ถ้าชื่อซ้ำกัน ให้ Rater 2 เลือกคอลัมน์ถัดไป
            kv2_default_idx = min(kv1_default_idx + 1, len(all_cols) - 1)
            
        k1, k2 = st.columns(2)
        # 🟢 FIX BUG: ใช้ index ที่คำนวณได้
        kv1 = k1.selectbox("Rater/Method 1:", all_cols, index=kv1_default_idx, key='kappa_v1_diag')
        kv2 = k2.selectbox("Rater/Method 2:", all_cols, index=kv2_default_idx, key='kappa_v2_diag')
        if kv1 == kv2:
            st.warning("Please select two different columns for Kappa.")
        
        k_run, k_dl = st.columns([1, 1])
        if 'html_output_kappa' not in st.session_state:
            st.session_state.html_output_kappa = None
        
        if k_run.button("🤝 Calculate Kappa", key='btn_kappa_run'):
            res_df, err, conf_mat = diag_test.calculate_kappa(df, kv1, kv2)
            if err:
                st.error(err)
            else:
                rep_elements = [
                    {'type': 'text', 'data': f"Agreement Analysis: {kv1} vs {kv2}"},
                    {'type': 'table', 'header': 'Kappa Statistics', 'data': res_df},
                    {'type': 'contingency_table', 'header': 'Confusion Matrix (Crosstab)', 'data': conf_mat, 'outcome_col': kv2}
                ]
                html = diag_test.generate_report(f"Kappa: {kv1} vs {kv2}", rep_elements)
                st.session_state.html_output_kappa = html
                st.components.v1.html(html, height=500, scrolling=True)
                
        with k_dl:
            if st.session_state.html_output_kappa:
                st.download_button("📥 Download Report", st.session_state.html_output_kappa, "kappa_report.html", "text/html", key='dl_kappa_diag')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_kappa_diag')

    # --- Descriptive ---
    with sub_tab4:
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
        
        # [แก้ไข] จัด Indentation ให้ตรงกับ if run_col.button ด้านบน (8 spaces)
        with dl_col:
            if st.session_state.html_output_desc:
                st.download_button("📥 Download Report", st.session_state.html_output_desc, "desc.html", "text/html", key='dl_desc_diag')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_desc_diag')

    # --- Reference & Interpretation (NEW) ---
    with sub_tab5:
        st.markdown("##### 📚 Quick Reference: Diagnostic Tests")
        
        st.info("""
        **When to Use Which Test:**
        
        | Test | Variables | Purpose | Example |
        |------|-----------|---------|----------|
        | **ROC AUC** | 1 continuous + 1 binary | Diagnostic test performance | Blood glucose vs diabetes diagnosis |
        | **Chi-Square** | 2 categorical | Association between categories | Treatment group vs Outcome (Yes/No) |
        | **Kappa** | 2 categorical (same categories) | Agreement between raters | Doctor A diagnosis vs Doctor B diagnosis |
        | **Descriptive** | Any single variable | Data distribution & summary | Patient age, gender, lab values |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ROC Curve (AUC)")
            st.markdown("""
            **When to Use:**
            - Evaluating diagnostic test performance
            - Finding optimal cut-off thresholds
            - Comparing multiple diagnostic tests
            
            **Interpretation:**
            - AUC = 0.9-1.0: Excellent test ✅
            - AUC = 0.8-0.9: Good test ✔️
            - AUC = 0.7-0.8: Fair test ⚠️
            - AUC < 0.7: Poor test ❌
            
            **Common Mistakes:**
            - Using non-continuous predictor (should be numeric score)
            - Not validating on independent test set
            - Ignoring confidence intervals
            """)
            
            st.markdown("### Chi-Square & Risk Analysis")
            st.markdown("""
            **Test Selection:**
            - Pearson: Large samples (n > 40)
            - Yates' correction: Small samples
            - Fisher's Exact: Expected count < 5
            
            **Interpretation:**
            - p < 0.05: Significant association ✅
            - p ≥ 0.05: No association ❌
            
            **Metrics:**
            - **RR > 1**: Increased risk
            - **OR > 1**: Increased odds
            - **NNT < 10**: Excellent ✅
            - **NNT > 50**: Marginal ⚠️
            """)
        
        with col2:
            st.markdown("### Agreement (Kappa)")
            st.markdown("""
            **Interpretation (Landis & Koch):**
            - κ < 0: Poor ❌
            - κ 0.01-0.20: Slight
            - κ 0.21-0.40: Fair
            - κ 0.41-0.60: Moderate ✔️
            - κ 0.61-0.80: Substantial ✅
            - κ 0.81-1.00: Perfect 🏆
            
            **Common Mistakes:**
            - Using Kappa for continuous data (use ICC instead)
            - Not checking if categories are the same
            - Interpreting raw agreement % (need chance adjustment)
            """)
            
            st.markdown("### Descriptive Statistics")
            st.markdown("""
            **For Numeric Data:**
            - Mean ± SD (if normal) ✅
            - Median ± IQR (if non-normal) ✅
            - Check normality with Shapiro-Wilk test
            
            **For Categorical Data:**
            - Frequency counts
            - Percentages
            
            **Common Mistakes:**
            - Mean ± SD for non-normal data ❌
            - Not checking for outliers
            - Ignoring missing data patterns
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Quick Decision Guide
        
        **Question: My test predicts disease (continuous score vs binary disease status)?**
        → Use **ROC Curve & AUC** (Tab 1)
        
        **Question: Two categorical variables - are they associated?**
        → Use **Chi-Square** (Tab 2)
        
        **Question: Do two raters/methods agree on classification?**
        → Use **Kappa** (Tab 3)
        
        **Question: I just want to understand my data distribution?**
        → Use **Descriptive Statistics** (Tab 4)
        """)
