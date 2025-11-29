# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Import Module ที่เราแยกไฟล์ไว้
from logic import process_data_and_generate_html
import diag_test
import table_one

# --- CONFIGURATION ---
st.set_page_config(page_title="Medical Stat Tool", layout="wide")

st.title("🏥 Medical Statistical Tool")

# --- INITIALIZE STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {} 
    
# 🟢 FIX: Initialize all output variables at the top
if 'html_output_t1' not in st.session_state:
    st.session_state.html_output_t1 = None
if 'html_output_roc' not in st.session_state:
    st.session_state.html_output_roc = None
if 'html_output_chi' not in st.session_state:
    st.session_state.html_output_chi = None
if 'html_output_desc' not in st.session_state:
    st.session_state.html_output_desc = None
if 'html_output_logit' not in st.session_state:
    st.session_state.html_output_logit = None


# --- HELPER FUNCTIONS ---
def check_perfect_separation(df, target_col):
    """ตรวจสอบว่าตัวแปรไหนแยกกลุ่มผลลัพธ์ได้สมบูรณ์เกินไป (เสี่ยง Error ใน Logistic)"""
    risky_vars = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: return []
    except: return []

    for col in df.columns:
        if col == target_col: continue
        # เช็คเฉพาะ Categorical หรือตัวแปรที่มีกลุ่มน้อยๆ
        if df[col].nunique() < 10: 
            try:
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except: pass
    return risky_vars

def safe_rerun():
    """รองรับคำสั่ง Rerun ทุกเวอร์ชันของ Streamlit"""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ==========================================
# 1. SIDEBAR: DATA & SETTINGS
# ==========================================
st.sidebar.title("MENU")

# --- 1.1 DATA LOADER ---
st.sidebar.header("1. Data Management")

# 🟢 SUPER EXAMPLE DATA GENERATOR
if st.sidebar.button("📄 Load Example Data"):
    # สร้างข้อมูลจำลอง 150 เคส ที่เหมาะกับทุกสถิติ
    np.random.seed(42)
    n = 150
    
    # 1. Group (สำหรับ Table 1)
    groups = np.random.choice(['Standard Care', 'New Drug'], n, p=[0.5, 0.5])
    
    # 2. Continuous Vars (สำหรับ T-test/ANOVA)
    age = np.random.normal(60, 12, n).astype(int)
    bmi = np.random.normal(25, 4, n).round(1)
    
    # 3. Categorical Vars (สำหรับ Chi-square)
    sex = np.random.choice([0, 1], n) # 0=F, 1=M
    
    # โรคประจำตัว (สัมพันธ์กับอายุ)
    ht_prob = (age - 20) / 80  # อายุมากเสี่ยงมาก
    ht = np.random.binomial(1, np.clip(ht_prob, 0.1, 0.9))
    
    # 4. Diagnostic Test Score (สำหรับ ROC)
    # สร้าง Score ที่มีความสามารถในการแยกโรค (AUC ~ 0.85)
    risk_score = np.random.normal(5, 2, n)
    
    # คนที่มี Score สูง มีโอกาสเป็นโรคสูง (Logistic function)
    # เพิ่มความชันเพื่อให้ ROC สวยและ Significant
    prob_disease = 1 / (1 + np.exp(-(risk_score - 6)*0.8))
    outcome = np.random.binomial(1, prob_disease)
    
    # Combine Data
    data = {
        'ID': range(1, n+1),
        'Group_Treatment': groups,
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Hypertension': ht,
        'Risk_Score': risk_score.round(2),
        'Outcome_Disease': outcome
    }
    
    st.session_state.df = pd.DataFrame(data)
    
    # Pre-set Metadata (Labeling)
    st.session_state.var_meta = {
        'Sex': {'type': 'Categorical', 'map': {0:'Female', 1:'Male'}},
        'Hypertension': {'type': 'Categorical', 'map': {0:'No', 1:'Yes'}},
        'Outcome_Disease': {'type': 'Categorical', 'map': {0:'Healthy', 1:'Disease'}},
        'Group_Treatment': {'type': 'Categorical', 'map': {}} 
    }
    
    st.sidebar.success("Loaded! Ready for all tabs.")
    safe_rerun()

# Upload File
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded_file)
        else: st.session_state.df = pd.read_excel(uploaded_file)
        st.sidebar.success("File Uploaded!")
    except Exception as e: st.sidebar.error(f"Error: {e}")

# --- 1.2 VARIABLE SETTINGS ---
if st.session_state.df is not None:
    st.sidebar.header("2. Variable Settings")
    df = st.session_state.df
    all_cols = df.columns.tolist()
    
    selected_var = st.sidebar.selectbox("Edit Variable:", ["Select..."] + all_cols)
    
    if selected_var != "Select...":
        # Load current settings
        meta = st.session_state.var_meta.get(selected_var, {})
        curr_type = meta.get('type', 'Auto-detect')
        curr_map = meta.get('map', {})
        
        # Edit Type
        new_type = st.sidebar.radio("Type:", ['Auto-detect', 'Categorical', 'Continuous'], 
                                    index=['Auto-detect', 'Categorical', 'Continuous'].index(curr_type))
        
        # Edit Labels
        map_str = "\n".join([f"{k}={v}" for k, v in curr_map.items()])
        st.sidebar.markdown("Labels (e.g. 0=No):")
        new_labels_str = st.sidebar.text_area("Label Map", value=map_str, height=80, label_visibility="collapsed")
        
        if st.sidebar.button("💾 Save Settings"):
            # Parse Labels
            new_map = {}
            for line in new_labels_str.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    try:
                        k = k.strip()
                        if k.replace('.','',1).isdigit():
                            k = float(k) if '.' in k else int(k)
                        new_map[k] = v.strip()
                    except: pass
            
            # Save to Session State
            if selected_var not in st.session_state.var_meta:
                st.session_state.var_meta[selected_var] = {}
            
            st.session_state.var_meta[selected_var]['type'] = new_type
            st.session_state.var_meta[selected_var]['map'] = new_map
            st.sidebar.success("Saved!")
            safe_rerun()

# 🟢 NEW: เพิ่มเครดิตที่ส่วนท้ายของ Sidebar (เฉพาะเมื่อมีการโหลดข้อมูลแล้ว)
st.sidebar.markdown("""
<hr style="margin-top: 20px; margin-bottom: 10px; border-color: var(--border-color);">
<div style='text-align: center; font-size: 0.7em; color: var(--text-color);'>
  &copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit
</div>
""", unsafe_allow_html=True)


# ==========================================
# 2. MAIN AREA: NAVIGATION & CONTENT
# ==========================================

if st.session_state.df is not None:
    df = st.session_state.df
    all_cols = df.columns.tolist()

    # --- TOP NAVIGATION (Reordered Tabs) ---
    main_tab0, main_tab1, main_tab2, main_tab3 = st.tabs([
        "📄 Raw Data", 
        "📋 Baseline Table 1", 
        "🔬 Diagnostic Test (ROC/Chi2)", 
        "📊 Logistic Regression" 
    ])

    # -----------------------------------------------
    # TAB 0: RAW DATA
    # -----------------------------------------------
    with main_tab0:
        st.subheader("Raw Data Table")
        st.info("💡 You can view, scroll, and edit your raw data directly in this table.")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=500, key='editor_raw')
        st.session_state.df = edited_df 
        df = st.session_state.df 

    # -----------------------------------------------
    # TAB 1: BASELINE CHARACTERISTICS (TABLE 1)
    # -----------------------------------------------
    with main_tab1:
        st.subheader("1. Baseline Characteristics (Table 1)")
        st.markdown("""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 20px;">
                <p>
                The **Baseline Characteristics (Table 1)** tool generates a summary table of the study population, typically stratifying variables by treatment or outcome groups.
                </p>
                <p>
                It reports continuous variables as **Mean ± SD** and categorical variables as **Counts (%)**, providing **P-values** to test for significant differences between groups (using T-test/ANOVA for continuous and Chi-square/Fisher's exact test for categorical variables).
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Try to find group column
        grp_idx = 0
        for i, c in enumerate(all_cols):
            if 'group' in c.lower() or 'treat' in c.lower(): grp_idx = i; break
        
        c_t1, c_t2 = st.columns([1, 2])
        with c_t1:
            col_group = st.selectbox("Group By (Column):", ["None"] + all_cols, index=grp_idx+1, key='t1_group')
        
        with c_t2:
            def_vars = [c for c in all_cols if c != col_group]
            selected_vars = st.multiselect("Include Variables:", all_cols, default=def_vars, key='t1_vars')
            
        run_col_t1, download_col_t1 = st.columns([1, 1])
        # if 'html_output_t1' not in st.session_state: # REMOVED: Initialized at top

        if run_col_t1.button("📊 Generate Table 1", type="primary"):
            with st.spinner("Generating Table 1..."):
                try:
                    grp = None if col_group == "None" else col_group
                    html_t1 = table_one.generate_table(df, selected_vars, grp, st.session_state.var_meta)
                    st.session_state.html_output_t1 = html_t1 # Store HTML
                    st.components.v1.html(html_t1, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.html_output_t1 = None # Clear state on error
                    
        with download_col_t1:
            if st.session_state.html_output_t1:
                st.download_button("📥 Download HTML Report", st.session_state.html_output_t1, "table1.html", "text/html", key='dl_btn_t1')
            else:
                # Placeholder button to maintain alignment
                st.button("📥 Download HTML Report", disabled=True, key='placeholder_t1')

    # -----------------------------------------------
    # TAB 2: DIAGNOSTIC TEST (Fixed Alignment and Logic)
    # -----------------------------------------------
    with main_tab2:
        st.subheader("2. Diagnostic Test & Statistics")
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📈 ROC Curve & AUC", "🎲 2x2 analysis : Chi square, RR, NNT", "📊 Descriptive"])
        
        # --- ROC ---
        with sub_tab1:
            st.markdown("##### ROC Curve Analysis")
            st.markdown("""
                <p>
                The **Receiver Operating Characteristic (ROC) Curve** evaluates the performance of a continuous test score in predicting a binary outcome.
                </p>
                <ul>
                    <li>**AUC (Area Under the Curve):** Measures the overall discriminative ability (0.5 = random chance, 1.0 = perfect separation).</li>
                    <li>**Youden Index:** Used to find the optimal cut-off point that maximizes both Sensitivity and Specificity.</li>
                    <li>**P-value (H0: AUC=0.5):** Tests if the AUC is statistically better than random chance.</li>
                </ul>
            """, unsafe_allow_html=True)
            
            # Use 4 columns for ROC inputs
            rc1, rc2, rc3, rc4 = st.columns(4)
            # Find default outcome index
            def_idx = 0
            for i, c in enumerate(all_cols):
                if 'outcome' in c.lower() or 'died' in c.lower():
                    def_idx = i; break
            
            truth = rc1.selectbox("Gold Standard (Binary):", all_cols, index=def_idx, key='roc_truth')
            
            # Find default score index
            score_idx = 0
            for i, c in enumerate(all_cols):
                if 'score' in c.lower(): score_idx = i; break
            score = rc2.selectbox("Test Score (Continuous):", all_cols, index=score_idx, key='roc_score')
            
            method = rc3.radio("CI Method:", ["DeLong et al.", "Binomial (Hanley)"])

            # Positive Label Selection
            pos_label = None
            unique_truth_vals = df[truth].dropna().unique()
            if len(unique_truth_vals) == 2:
                # เรียงค่าเพื่อให้ผู้ใช้เลือกได้ง่ายขึ้น
                sorted_vals = sorted([str(x) for x in unique_truth_vals]) 
                selected_pos_label = rc4.selectbox("Positive Label (1):", sorted_vals, key='roc_pos_label')
                pos_label = selected_pos_label
            elif len(unique_truth_vals) != 2:
                rc4.warning("Requires 2 unique values.")

            
            run_col_roc, download_col_roc = st.columns([1, 1])
            # if 'html_output_roc' not in st.session_state: # REMOVED: Initialized at top
            
            if run_col_roc.button("📉 Analyze ROC", key='btn_roc'):
                if pos_label is None and len(unique_truth_vals) == 2:
                    st.error("Error: Please select the Positive Label (1).")
                    st.session_state.html_output_roc = None
                elif len(unique_truth_vals) != 2:
                    st.error("Error: Gold Standard must have exactly 2 classes.")
                    st.session_state.html_output_roc = None
                else:
                    # Pass the selected pos_label to diag_test
                    res, err, fig, coords_df = diag_test.analyze_roc(df, truth, score, 'delong' if 'DeLong' in method else 'hanley', pos_label_user=pos_label)
                    
                    if err: 
                        st.error(err)
                        st.session_state.html_output_roc = None 
                    else:
                        report_elements = [
                            {'type': 'text', 'data': f"Analysis of Test Score: <b>{score}</b> vs Gold Standard: <b>{truth}</b>"},
                            {'type': 'plot', 'header': 'ROC Curve', 'data': fig},
                            {'type': 'table', 'header': 'Key Statistics', 'data': pd.DataFrame([res]).T},
                            {'type': 'table', 'header': 'Diagnostic Performance (All Cut-offs)', 'data': coords_df}
                        ]
                        html_report = diag_test.generate_report(f"ROC Analysis: {score}", report_elements)
                        st.session_state.html_output_roc = html_report # Store HTML
                        st.components.v1.html(html_report, height=800, scrolling=True)

            with download_col_roc:
                if st.session_state.html_output_roc:
                    st.download_button("📥 Download HTML Report", st.session_state.html_output_roc, "roc_report.html", "text/html", key='dl_btn_roc')
                else:
                    # Placeholder button to maintain alignment
                    st.button("📥 Download HTML Report", disabled=True, key='placeholder_roc')


        # --- Chi-Square ---
        with sub_tab2:
            st.markdown("##### Chi-Square Test & Risk Measures")
            st.markdown("""
                <p>
                The **Chi-Square Test** examines the association between two categorical variables. For $2\times2$ tables, the analysis automatically includes **Risk Ratio (RR)**, **Risk Difference (RD)**, and **Number Needed to Treat (NNT)**.
                </p>
                <p>
                **Variable 1** is interpreted as the **Exposure/Treatment (Row)**, and **Variable 2** as the **Outcome/Event (Column)**.
                </p>
            """, unsafe_allow_html=True)
            
            cc1, cc2 = st.columns(2)
            v1 = cc1.selectbox("Variable 1 (Exposure/Group):", all_cols, key='chi1')
            v2 = cc2.selectbox("Variable 2 (Outcome/Event):", all_cols, index=min(1, len(all_cols)-1), key='chi2')

            # ADDED: Correction selection
            correction_flag = st.radio("Correction Method (for 2x2 table):", 
                                         ['Pearson (Standard)', "Yates' correction"], 
                                         index=0, key='chi_corr_method') == "Yates' correction"
            
            run_col_chi, download_col_chi = st.columns([1, 1])
            # if 'html_output_chi' not in st.session_state: # REMOVED: Initialized at top
            
            if run_col_chi.button("Run Chi-Square", key='btn_chi'):
                tab_res, results = diag_test.calculate_chi2(df, v1, v2) 
                
                if tab_res is not None and 'error' not in results:
                    
                    report_elements = [
                        {'type': 'text', 'data': f"<b>Chi-square Test Result:</b> {results.get('chi2_msg', 'N/A')}"},
                        
                        # 🟢 FIX: ใช้ type: 'contingency_table' เพื่อ trigger Multi-level Header
                        {'type': 'contingency_table', 
                         'header': 'Contingency Table [Count (Row %) / (Total %)]', 
                         'data': tab_res,
                         'outcome_col': v2 # ส่งชื่อ Outcome Column ไปสำหรับ Header
                        }
                    ]
                    
                    # ADD RR/ARR/NNT IF IT'S A 2x2 TABLE
                    if results.get('Is_2x2', False):
                        rr = results.get('RR', np.nan)
                        rd = results.get('RD', np.nan)
                        nnt = results.get('NNT', np.nan)
                        
                        # Prepare RR/NNT table
                        rr_stats = pd.DataFrame({
                            "Statistic": [
                                f"Risk in {results['R_exp_label']} (R1)",
                                f"Risk in {results['R_unexp_label']} (R0)",
                                "Risk Ratio (RR)",
                                "Risk Difference (RD / ARR)",
                                "Number Needed to Treat (NNT)"
                            ],
                            "Value": [
                                f"{results.get('R_exp', np.nan):.4f}",
                                f"{results.get('R_unexp', np.nan):.4f}",
                                f"{rr:.4f}",
                                f"{rd:.4f}",
                                f"{nnt:.0f}" if pd.notna(nnt) and nnt != np.inf else "Inf"
                            ],
                            "Interpretation": [
                                "Risk of Event in Group 1",
                                "Risk of Event in Group 0 (Baseline)",
                                f"Risk in Gp 1 / Risk in Gp 0 (Event: {results['Event_label']})",
                                "Absolute difference in risk (R1 - R0)",
                                "Number of patients to treat to prevent 1 Event (if RD < 0)"
                            ]
                        })
                        
                        # Add Interpretation Note
                        exp_label = results['R_exp_label']
                        unexp_label = results['R_unexp_label']
                        event_label = results['Event_label']
                        
                        if pd.notna(rr) and pd.notna(rd):
                            rr_interpret = f"RR/RD Interpretation (Assuming Row 0: '{exp_label}' is Exposure/Treatment, and Column 0: '{event_label}' is Event):"
                            if rr > 1:
                                rr_interpret += f" RR > 1: The risk of '{event_label}' is {rr:.2f} times higher in '{exp_label}' compared to '{unexp_label}'."
                            elif rr < 1:
                                rr_interpret += f" RR < 1: The risk of '{event_label}' is reduced by {(1 - rr) * 100:.1f}% in '{exp_label}' compared to '{unexp_label}'."
                            else:
                                rr_interpret += " RR = 1: No difference in risk."

                            report_elements.append({'type': 'table', 'header': 'Risk & Effect Measures (2x2 Table)', 'data': rr_stats})
                            report_elements.append({'type': 'text', 'data': f"<b>Interpretation Note:</b> {rr_interpret}"})
                        
                    # Generate report
                    html_report = diag_test.generate_report(f"Chi-square & Risk Analysis: {v1} vs {v2}", report_elements)
                    st.session_state.html_output_chi = html_report # Store HTML
                    st.components.v1.html(html_report, height=700, scrolling=True)
                else:
                    st.error(results.get('error', 'Analysis Failed. Check if data is categorical and has non-zero counts.'))
                    st.session_state.html_output_chi = None # Clear state on error
                    
            with download_col_chi:
                if st.session_state.html_output_chi:
                    st.download_button("📥 Download HTML Report", st.session_state.html_output_chi, "chi2_report.html", "text/html", key='dl_btn_chi')
                else:
                    st.button("📥 Download HTML Report", disabled=True, key='placeholder_chi')


        # --- Descriptive ---
        with sub_tab3:
            st.markdown("##### Descriptive Statistics")
            st.markdown("""
                <p>
                **Descriptive Statistics** summarizes the basic features of the data in a study. 
                </p>
                <p>
                For **Continuous** variables, it reports Count, Mean, Standard Deviation (SD), and Quartiles. For **Categorical** variables, it reports Counts and Percentages.
                </p>
            """, unsafe_allow_html=True)
            
            dv = st.selectbox("Select Variable:", all_cols, key='desc_var')
            
            run_col_desc, download_col_desc = st.columns([1, 1])
            # if 'html_output_desc' not in st.session_state: # REMOVED: Initialized at top
            
            if run_col_desc.button("Show Stats", key='btn_desc'):
                res_df = diag_test.calculate_descriptive(df, dv)
                if res_df is not None:
                    report_elements = [
                        {'type': 'table', 'header': '', 'data': res_df}
                    ]
                    html_report = diag_test.generate_report(f"Descriptive Statistics: {dv}", report_elements)
                    st.session_state.html_output_desc = html_report # Store HTML
                    st.components.v1.html(html_report, height=500, scrolling=True)
                else:
                    st.session_state.html_output_desc = None # Clear state on error
                    
            with download_col_desc:
                if st.session_state.html_output_desc:
                    st.download_button("📥 Download HTML Report", st.session_state.html_output_desc, "desc_report.html", "text/html", key='dl_btn_desc')
                else:
                    st.button("📥 Download HTML Report", disabled=True, key='placeholder_desc')

    # -----------------------------------------------
    # TAB 3: LOGISTIC REGRESSION (UNCHANGED)
    # -----------------------------------------------
    with main_tab3:
        st.subheader("3. Logistic Regression Analysis")
        
        # English Description of Logistic Regression
        st.markdown("""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 20px;">
                <p>
                <b>Binary Logistic Regression</b> is used to model the probability of a binary outcome (e.g., 0 or 1, disease presence/absence) based on one or more predictor variables.
                </p>
                <ul>
                    <li><b>Univariate Analysis (Crude OR):</b> Calculates the association of each individual variable with the outcome, providing the Crude Odds Ratio (OR).</li>
                    <li><b>Multivariate Analysis (Adjusted OR):</b> Includes potential confounding variables (screened at P-value < 0.20) into a single model to determine independent predictors, providing the Adjusted Odds Ratio (aOR).</li>
                </ul>
                <p style='font-size:0.9em;'><i>You can view and edit the data in the 📄 Raw Data tab.</i></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Analysis Configuration")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Find default outcome
            def_idx = 0
            for i, c in enumerate(all_cols):
                if 'outcome' in c.lower() or 'died' in c.lower():
                    def_idx = i; break
            
            target = st.selectbox("Select Outcome (Y):", all_cols, index=def_idx, key='logit_target')
            
        with col2:
            # Check Perfect Separation
            risky_vars = check_perfect_separation(df, target)
            exclude_cols = []
            
            if risky_vars:
                st.warning(f"⚠️ Risk of Perfect Separation: {', '.join(risky_vars)}")
                exclude_cols = st.multiselect("Exclude Variables:", all_cols, default=risky_vars, key='logit_exclude')
            else:
                exclude_cols = st.multiselect("Exclude Variables (Optional):", all_cols, key='logit_exclude_optional')

        run_col, download_col = st.columns([1, 1])
        # if 'html_output_logit' not in st.session_state: # REMOVED: Initialized at top
        
        if run_col.button("🚀 Run Logistic Regression", type="primary"):
            if df[target].nunique() < 2:
                st.error("Error: Outcome must have at least 2 values (e.g., 0 and 1).")
                st.session_state.html_output_logit = None # Clear state on error
            else:
                with st.spinner("Calculating..."):
                    try:
                        final_df = df.drop(columns=exclude_cols, errors='ignore')
                        html = process_data_and_generate_html(final_df, target, var_meta=st.session_state.var_meta)
                        st.session_state.html_output_logit = html # Store HTML
                        st.components.v1.html(html, height=600, scrolling=True)
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")
                        st.session_state.html_output_logit = None # Clear state on error
                        
        with download_col:
            if st.session_state.html_output_logit:
                st.download_button("📥 Download Report", st.session_state.html_output_logit, "logit_report.html", "text/html", key='dl_btn_logit')
            else:
                # Placeholder button to maintain alignment
                st.button("📥 Download Report", disabled=True, key='placeholder_logit')

else:
    st.info("👈 Please load example data or upload a file to start.")
    st.markdown("""
    ### Features:
    1.  **Logistic Regression:** Univariate & Multivariate with Auto-selection.
    2.  **Diagnostic Test:** ROC Curve, AUC, Best Cut-off Table, Chi-square (with Report Export).
    3.  **Table 1:** Auto-generated Baseline Characteristics with P-values.
    """)

# ==========================================
# 3. GLOBAL CSS (Cleanup)
# ==========================================

# 🟢 NEW: Inject CSS to hide the default Streamlit footer (Keep this part)
st.markdown("""
<style>
/* FIX: ทำให้ Sidebar Background ยาวลงมาถึงด้านล่างสุด */
[data-testid="stSidebar"] > div {
    height: 100vh !important; /* บังคับให้ Sidebar background ยาวเท่าความสูงของหน้าจอ */
}

/* โค้ดสำหรับซ่อน Streamlit footer เดิม (ยังจำเป็นต้องมี) */
footer {
    visibility: hidden;
    height: 0px;
}
footer:after {
    content: none;
}
</style>
""", unsafe_allow_html=True)
