import streamlit as st
import pandas as pd
import diag_test # ✅ ใช้ diag_test ตัวเดียว

def render(df, _var_meta=None):  # var_meta reserved for future use
    """
    Render Streamlit UI panels for diagnostic tests and statistics.
    
    Displays five interactive tabs for common diagnostic analyses:
    - ROC Curve & AUC
    - Chi-Square & Risk (2x2) with RR/OR/NNT
    - Agreement (Cohen's Kappa)
    - Reliability (Intraclass Correlation Coefficient, ICC)
    - Descriptive statistics
    
    Each tab provides controls for selecting columns from `df`, running the analysis, viewing results as embedded HTML, and downloading an HTML report. Generated report HTML is stored in Streamlit session state under keys: `html_output_roc`, `html_output_chi`, `html_output_kappa`, `html_output_icc`, and `html_output_desc`.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing the variables to analyze; column names are used for UI selections.
        _var_meta (Any): Metadata about variables (unused for visible output selection unless integrated by UI); present for potential future use.
    """
    st.subheader("2. Diagnostic Test & Statistics")
    # 🟢 UPDATE: เพิ่ม Tab "Reliability (ICC)" เป็น Tab ที่ 4
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "📈 ROC Curve & AUC", 
        "🎲 Chi-Square & Risk-RR,OR,NNT (Categorical)", 
        "🤝 Agreement (Kappa)", 
        "📏 Reliability (ICC)", # 🟢 NEW TAB
        "📊 Descriptive"
    ])
    all_cols = df.columns.tolist()
    if not all_cols:
        st.error("Dataset has no columns to analyze.")
        return
        
    # --- ROC ---
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
        
        # Auto-select V1 and V2
        v1_default_name = 'Group_Treatment'
        v2_default_name = 'Status_Death'
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
        def get_pos_label_settings(df: pd.DataFrame, col_name: str) -> tuple[list[str], int]:
            """
            Compute candidate positive-label values for a column and determine a default selection index.
            
            Parameters:
                df (pandas.DataFrame): DataFrame containing the column.
                col_name (str): Name of the column to inspect.
            
            Returns:
                tuple:
                    unique_vals (list[str]): Sorted list of the column's unique non-null values as strings.
                    default_idx (int): Index into `unique_vals` to use as the default positive label (index of value `'1'` if present; otherwise `0`).
            """
            unique_vals = [str(x) for x in df[col_name].dropna().unique()]
            unique_vals.sort()
            default_idx = 0
            if '1' in unique_vals:
                default_idx = unique_vals.index('1')
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
                rep_elements = [
                    {'type': 'text', 'data': f"<b>Result:</b> {msg}"},
                    {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab, 'outcome_col': v2},
                ]
                if stats is not None:
                    rep_elements.append({'type': 'table', 'header': 'Statistics', 'data': pd.DataFrame([stats]).T})
                if risk_df is not None:
                    rep_elements.append({'type': 'table', 'header': 'Risk & Effect Measures (2x2 Table)', 'data': risk_df})
                
                html = diag_test.generate_report(f"Chi2: {v1} vs {v2}", rep_elements)
                st.session_state.html_output_chi = html
                st.components.v1.html(html, height=600, scrolling=True)
            else: 
                st.error(msg)
        
        with dl_col:
            if st.session_state.html_output_chi:
                st.download_button("📥 Download Report", st.session_state.html_output_chi, "chi2_diag.html", "text/html", key='dl_chi_diag')
            else: 
                st.button("📥 Download Report", disabled=True, key='ph_chi_diag')
       
    # --- 🟢 NEW: Agreement (Kappa) ---
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
                    {'type': 'text', 'data': f"<b>Agreement Analysis:</b> {kv1} vs {kv2}"},
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

    # --- 🟢 NEW: Reliability (ICC) ---
    with sub_tab4:
        st.markdown("##### Reliability Analysis (Intraclass Correlation Coefficient - ICC)")
        st.info("""
            **💡 Guide:** Evaluates the reliability/agreement between 2 or more raters/methods for **Numeric/Continuous** variables.
            * **ICC(2,1) Absolute Agreement:** Use when you care if the absolute scores are the same (e.g., Method A vs Method B).
            * **ICC(3,1) Consistency:** Use when you care if the ranking is consistent, even if absolute scores differ (e.g., systematic bias).
        """)
        
        # 🟢 Auto-select เฉพาะ Numeric Columns เพื่อความปลอดภัย
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        
        # Auto-select columns with 'measurement' or 'rater' or 'machine'
        default_icc_cols = [c for c in numeric_cols if any(k in c.lower() for k in ['measure', 'machine', 'rater', 'score', 'read'])]
        if len(default_icc_cols) < 2:
            default_icc_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else []
        
        icc_cols = st.multiselect("Select Variables (Raters/Methods):", numeric_cols, default=default_icc_cols, key='icc_vars_diag')
        
        icc_run, icc_dl = st.columns([1, 1])
        if 'html_output_icc' not in st.session_state: 
            st.session_state.html_output_icc = None
        
        if icc_run.button("📏 Calculate ICC", key='btn_icc_run'):
            if len(icc_cols) < 2:
                st.error("Please select at least 2 numeric columns for ICC.")
                st.stop()
            res_df, err, anova_df = diag_test.calculate_icc(df, icc_cols)
            
            if err:
                st.error(err)
            else:
                rep_elements = [
                    {'type': 'text', 'data': f"<b>ICC Analysis:</b> {', '.join(icc_cols)}"},
                    {'type': 'table', 'header': 'ICC Results (Single Measures)', 'data': res_df},
                    {'type': 'table', 'header': 'ANOVA Table (Reference)', 'data': anova_df}
                ]
                html = diag_test.generate_report("ICC Analysis", rep_elements)
                st.session_state.html_output_icc = html
                st.components.v1.html(html, height=400, scrolling=True)
                
        with icc_dl:
            if st.session_state.html_output_icc:
                st.download_button("📥 Download Report", st.session_state.html_output_icc, "icc_report.html", "text/html", key='dl_icc_diag')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_icc_diag')
                
    # --- Descriptive (ย้ายมาเป็น sub_tab5) ---
    with sub_tab5:
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
