import streamlit as st
import pandas as pd
import diag_test # ✅ ใช้ diag_test ตัวเดียว
from typing import List, Tuple

# 🟢 NEW: Helper function to select between original and matched datasets
def _get_dataset_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Choose which dataset to use for downstream analysis and return it with a human-readable label.
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
                index=1,  # ถ้ามี matched แล้ว default เป็น matched
                horizontal=True,
                key="diag_data_source",
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


def render(df, _var_meta=None):  # var_meta reserved for future use
    """
    Render the Streamlit UI for interactive diagnostic analyses and report generation.
    """
    st.subheader("🧪 Diagnostic Tests (ROC)")

    # 🟢 NEW: Display matched data status if available
    if st.session_state.get("is_matched", False):
        st.info("✅ **Matched Dataset Available** - You can select it below for analysis")

    # 🟢 NEW: Select dataset (original or matched)
    selected_df, data_label = _get_dataset_for_analysis(df)
    st.write(f"**Using:** {data_label}")
    st.write(f"**Rows:** {len(selected_df)} | **Columns:** {len(selected_df.columns)}")
    
    # 🟢 IMPORTANT: Now 5 subtabs (added Reference & Interpretation)
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "📈 ROC Curve & AUC", 
        "🎲 Chi-Square & Risk Analysis (2x2)", 
        "🤝 Agreement (Kappa)", 
        "📊 Descriptive",
        "ℹ️ Reference & Interpretation"
    ])
    
    all_cols = selected_df.columns.tolist()
    if not all_cols:
        st.error("Dataset has no columns to analyze.")
        return
        
    # --- ROC ---
    with sub_tab1:
        st.markdown("##### ROC Curve Analysis")
        # 🧹 Removed detailed description (Moved to Tab 5)
        
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
        unique_vals = selected_df[truth].dropna().unique()
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
        
        if run_col.button("🚀 Analyze ROC", type="primary", key='btn_roc_diag'):
            if pos_label and len(unique_vals) == 2:
                # Call analyze_roc from diag_test (using selected_df)
                res, err, fig, coords_df = diag_test.analyze_roc(selected_df, truth, score, 'delong' if 'DeLong' in method else 'hanley', pos_label_user=pos_label)
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
        # 🧹 Removed detailed description (Moved to Tab 5)

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
            help="See Tab 5 for detailed guidance on choosing a method."
        )
        
        # Positive Label Selectors
        def get_pos_label_settings(df_input: pd.DataFrame, col_name: str) -> Tuple[List[str], int]:
            unique_vals = [str(x) for x in df_input[col_name].dropna().unique()]
            unique_vals.sort()
            default_idx = 0
            if '1' in unique_vals:
                default_idx = unique_vals.index('1')
            elif len(unique_vals) > 0 and '0' in unique_vals:
                default_idx = unique_vals.index('0')
            return unique_vals, default_idx

        c4, c5, c6 = st.columns(3)
        v1_uv, v1_default_idx = get_pos_label_settings(selected_df, v1)
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

        v2_uv, v2_default_idx = get_pos_label_settings(selected_df, v2)
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
        
        inputs_ok = not (v1_pos_label is None or v2_pos_label is None)
        if not inputs_ok:
            st.warning("Chi-Square disabled: one of the selected columns has no non-null values.")

        c6.empty()
        st.caption("Select Positive Label for Risk/Odds Ratio calculation (default is '1'):")
        
        run_col, dl_col = st.columns([1, 1])
        
        if 'html_output_chi' not in st.session_state: 
            st.session_state.html_output_chi = None

        if run_col.button("🚀 Analyze Chi-Square", type="primary", key='btn_chi_run_diag', disabled=not inputs_ok):
            
            df_calc = selected_df.copy()
            df_calc[v1] = df_calc[v1].astype("string")
            df_calc[v2] = df_calc[v2].astype("string")
            
            tab, stats, msg, risk_df = diag_test.calculate_chi2(
                df_calc, v1, v2, 
                method=method_choice,
                v1_pos=v1_pos_label,
                v2_pos=v2_pos_label
            )
            
            if tab is not None:
                if msg.strip():
                    status_text = f"Note: {msg.strip()}"
                else:
                    status_text = "Analysis Status: Completed successfully."
                
                rep_elements = [ 
                    {'type': 'text', 'data': "Analysis: Diagnostic Test / Chi-Square"},
                    {'type': 'text', 'data': f"Variables: {v1} vs {v2}"},
                    {'type': 'text', 'data': status_text},
                    {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab, 'outcome_col': v2},
                ]
                
                if stats is not None:
                    rep_elements.append({'type': 'table', 'header': 'Statistics', 'data': stats})
                if risk_df is not None:
                    rep_elements.append({'type': 'table', 'header': 'Risk & Effect Measures (2x2 Table)', 'data': risk_df})
                
                html = diag_test.generate_report(f"Chi2: {v1} vs {v2}", rep_elements)
                st.session_state.html_output_chi = html
                st.components.v1.html(html, height=600, scrolling=True)
                st.success("✅ Chi-Square analysis complete!")
            else: 
                st.error(msg)
        
        with dl_col:
            if st.session_state.html_output_chi:
                st.download_button("📥 Download Report", st.session_state.html_output_chi, "chi2_diag.html", "text/html", key='dl_chi_diag')
            else: 
                st.button("📥 Download Report", disabled=True, key='ph_chi_diag')
        
    # --- Agreement (Kappa) ---
    with sub_tab3:
        st.markdown("##### Agreement Analysis (Cohen's Kappa)")
        # 🧹 Removed detailed description (Moved to Tab 5)
        
        kv1_default_idx = 0
        kv2_default_idx = min(1, len(all_cols) - 1)
        
        for i, col in enumerate(all_cols):
            if 'dr_a' in col.lower() or 'rater_1' in col.lower() or 'diagnosis_a' in col.lower():
                kv1_default_idx = i
                break
        
        for i, col in enumerate(all_cols):
            if 'dr_b' in col.lower() or 'rater_2' in col.lower() or 'diagnosis_b' in col.lower():
                kv2_default_idx = i
                break

        if kv1_default_idx == kv2_default_idx and len(all_cols) > 1:
            kv2_default_idx = min(kv1_default_idx + 1, len(all_cols) - 1)
            
        k1, k2 = st.columns(2)
        kv1 = k1.selectbox("Rater/Method 1:", all_cols, index=kv1_default_idx, key='kappa_v1_diag')
        kv2 = k2.selectbox("Rater/Method 2:", all_cols, index=kv2_default_idx, key='kappa_v2_diag')
        if kv1 == kv2:
            st.warning("Please select two different columns for Kappa.")
        
        k_run, k_dl = st.columns([1, 1])
        if 'html_output_kappa' not in st.session_state:
            st.session_state.html_output_kappa = None
        
        if k_run.button("🚀 Calculate Kappa", type="primary", key='btn_kappa_run'):
            res_df, err, conf_mat = diag_test.calculate_kappa(selected_df, kv1, kv2)
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
        # 🧹 Removed detailed description (Moved to Tab 5)

        dv = st.selectbox("Select Variable:", all_cols, key='desc_v_diag')
        run_col, dl_col = st.columns([1, 1])
        if 'html_output_desc' not in st.session_state: st.session_state.html_output_desc = None
        
        if run_col.button("Show Stats", key='btn_desc_diag'):
            res = diag_test.calculate_descriptive(selected_df, dv)
            if res is not None:
                html = diag_test.generate_report(f"Descriptive: {dv}", [{'type':'table', 'data':res}])
                st.session_state.html_output_desc = html
                st.components.v1.html(html, height=500, scrolling=True)
        
        with dl_col:
            if st.session_state.html_output_desc:
                st.download_button("📥 Download Report", st.session_state.html_output_desc, "desc.html", "text/html", key='dl_desc_diag')
            else:
                st.button("📥 Download Report", disabled=True, key='ph_desc_diag')

    # --- Reference & Interpretation (Detailed Update) ---
    with sub_tab5:
        st.markdown("## 📚 Reference & Interpretation Guide")
        
        st.info("💡 **Tip:** This section provides detailed explanations and interpretation rules for all the diagnostic tests available in the other tabs.")
        
        # 💡 Decision Guide (First for quick access)
        st.markdown("### 🚦 Quick Decision Guide")
        st.markdown("""
        | **Question** | **Recommended Test** | **Example** |
        | :--- | :--- | :--- |
        | My test is a **score** (e.g., 0-100) and I want to see how well it predicts a **disease** (Yes/No)? | **ROC Curve & AUC** | Risk Score vs Diabetes |
        | I want to find the **best cut-off** value for my test score? | **ROC Curve (Youden Index)** | Finding optimal BP for Hypertension |
        | Are these two **groups** (e.g., Treatment vs Control) different in outcome (Cured vs Not Cured)? | **Chi-Square** | Drug A vs Placebo on Recovery |
        | Do two doctors **agree** on the same diagnosis? | **Cohen's Kappa** | Radiologist A vs Radiologist B |
        | I just want to summarize **one variable** (Mean, Count)? | **Descriptive** | Age distribution |
        """)
        
        st.divider()

        col1, col2 = st.columns(2)
        
        # --- Column 1: ROC & Chi-Square ---
        with col1:
            st.markdown("### 📈 ROC Curve & AUC")
            st.markdown("""
            **Concept:** Evaluates how well a continuous test discriminates between two groups (Gold Standard).
            
            **Key Metrics:**
            * **AUC (Area Under Curve):** The single best number to summarize performance.
                * `0.5`: Random guessing (Flip a coin) ❌
                * `1.0`: Perfect prediction 🏆
            * **Youden Index (J):** Used to find the "Optimal Cut-off".
                * Formula: $J = Sensitivity + Specificity - 1$
                * The score with the highest J is often chosen as the cut-point.
            * **P-value:** Tests if AUC is significantly different from 0.5.

            **Interpretation Rules:**
            | AUC Range | Performance |
            | :--- | :--- |
            | 0.90 - 1.00 | Excellent ✅ |
            | 0.80 - 0.89 | Good ✔️ |
            | 0.70 - 0.79 | Fair ⚠️ |
            | < 0.70 | Poor ❌ |
            """)
            
            st.markdown("### 🎲 Chi-Square & Risk Analysis")
            st.markdown("""
            **Concept:** Tests association between "Exposure" (Row) and "Outcome" (Col).
            
            **Choosing the Right Test:**
            * **Pearson Chi-Square:** Standard test. Use when sample size is large (Expected count > 5).
            * **Yates' Correction:** Conservative version of Pearson. Use for small samples.
            * **Fisher's Exact Test:** Use when any cell has **Expected count < 5** (Very small sample).

            **Effect Measures:**
            * **Odds Ratio (OR):** Probability of event in Exposed / Unexposed. (Common in Case-Control).
            * **Risk Ratio (RR):** Probability of event in Exposed / Unexposed. (Common in Cohort).
            * **NNT (Number Needed to Treat):** How many people you need to treat to prevent 1 bad outcome.
                * *Ideal NNT:* < 10 (Very effective)
            
            **Interpretation:**
            * **P < 0.05:** Significant association (Groups are different).
            * **OR/RR > 1:** Exposure is a **Risk Factor**.
            * **OR/RR < 1:** Exposure is **Protective**.
            """)

        # --- Column 2: Kappa & Descriptive ---
        with col2:
            st.markdown("### 🤝 Agreement (Cohen's Kappa)")
            st.markdown("""
            **Concept:** Measures agreement between two raters, **removing the agreement that could happen by chance**.
            
            * *Example:* Even two blindfolded doctors might agree on a diagnosis by random luck. Kappa removes this "luck" factor.
            
            **Interpretation (Landis & Koch Scale):**
            | Kappa (κ) | Strength of Agreement |
            | :--- | :--- |
            | < 0.00 | **Poor** (Worse than chance) ❌ |
            | 0.00 - 0.20 | **Slight** |
            | 0.21 - 0.40 | **Fair** |
            | 0.41 - 0.60 | **Moderate** ✔️ |
            | 0.61 - 0.80 | **Substantial** ✅ |
            | 0.81 - 1.00 | **Perfect** 🏆 |
            """)
            
            st.markdown("### 📊 Descriptive Statistics")
            st.markdown("""
            **Concept:** Summarizes the central tendency and spread of data.
            
            **For Numeric Data (e.g., Age, BMI):**
            * **Normal Distribution:** Use **Mean ± SD**.
            * **Skewed Distribution:** Use **Median (IQR)**.
            * *Tip:* Always check the Histogram or Shapiro-Wilk test to decide.
            
            **For Categorical Data (e.g., Gender, Grade):**
            * Report **Frequency (n)** and **Percentage (%)**.
            """)
