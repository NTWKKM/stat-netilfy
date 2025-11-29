import streamlit as st
import pandas as pd
import numpy as np
from logic import process_data_and_generate_html
import diag_test
import table_one

st.set_page_config(page_title="Medical Stat Tool", layout="wide")
st.title("🏥 Medical Statistical Tool")

# --- INITIALIZE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'var_meta' not in st.session_state: st.session_state.var_meta = {} 

# --- HELPERS ---
def safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

def check_perfect_separation(df, target_col):
    """ตรวจสอบว่าตัวแปรไหนแยกกลุ่มผลลัพธ์ได้สมบูรณ์เกินไป (เสี่ยง Error ใน Logistic)"""
    risky = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: return []
    except: return []
    for col in df.columns:
        if col == target_col: continue
        if df[col].nunique() < 10:
            try:
                if (pd.crosstab(df[col], y) == 0).any().any(): risky.append(col)
            except: pass
    return risky

# --- SIDEBAR ---
st.sidebar.title("MENU")
st.sidebar.header("1. Data Management")
if st.sidebar.button("📄 Load Super Example Data"):
    np.random.seed(42); n = 150
    data = {
        'ID': range(1, n+1),
        'Group': np.random.choice(['Control', 'Treatment'], n),
        'Age': np.random.normal(60, 12, n).astype(int),
        'Sex': np.random.choice([0, 1], n),
        'BMI': np.random.normal(25, 4, n).round(1),
        'Risk_Score': np.random.normal(5, 2, n).round(2),
        'Outcome': np.random.binomial(1, 1 / (1 + np.exp(-(np.random.normal(5, 2, n) - 5))))
    }
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {'Sex': {'type':'Categorical','map':{0:'F',1:'M'}}, 'Outcome': {'type':'Categorical','map':{0:'No',1:'Yes'}}}
    st.sidebar.success("Loaded!"); safe_rerun()

uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if uploaded:
    try:
        if uploaded.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded)
        else: st.session_state.df = pd.read_excel(uploaded)
    except: pass

# --- MAIN ---
if st.session_state.df is not None:
    df = st.session_state.df
    all_cols = df.columns.tolist()

    tab1, tab2, tab3 = st.tabs(["📊 Logistic Regression", "🔬 Diagnostic Test & Stats", "📋 Baseline Table 1"])

    # TAB 1: LOGISTIC
    with tab1:
        st.subheader("1. Logistic Regression")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=200)
        c1, c2 = st.columns([1, 2])
        out_idx = next((i for i, c in enumerate(all_cols) if 'outcome' in c.lower()), 0)
        target = c1.selectbox("Outcome (Y):", all_cols, index=out_idx)
        risky = check_perfect_separation(edited_df, target)
        exclude = c2.multiselect("Exclude:", all_cols, default=risky)
        if st.button("🚀 Run Logistic"):
            html = process_data_and_generate_html(edited_df.drop(columns=exclude, errors='ignore'), target, st.session_state.var_meta)
            st.components.v1.html(html, height=600, scrolling=True)

    # TAB 2: DIAGNOSTIC & STATS
    with tab2:
        st.subheader("2. Diagnostic Test & Statistics")
        # 🟢 เพิ่ม Compare Means (T-test) และปรับแท็บย่อย
        st2_1, st2_2, st2_3, st2_4 = st.tabs(["📈 ROC Curve", "🎲 Chi-Square", "⚖️ Compare Means (T-test)", "📊 Descriptive"])
        
        # 2.1 ROC
        with st2_1:
            st.markdown("##### ROC Curve Analysis")
            c1, c2 = st.columns(2)
            truth = c1.selectbox("Gold Standard:", all_cols, index=out_idx, key='roc_t')
            score = c2.selectbox("Test Score:", all_cols, key='roc_s')
            if st.button("Analyze ROC"):
                # Call analysis
                res, err, fig, coords = diag_test.analyze_roc(df, truth, score)
                if err: st.error(err)
                else:
                    st.success(f"AUC = {res['AUC']:.3f} (95% CI: {res['95% CI Lower']:.3f} - {res['95% CI Upper']:.3f})")
                    
                    # 🟢 แสดงค่าละเอียดตามที่ขอ
                    st.markdown("**Detailed Statistics:**")
                    st.json(res) 
                    
                    rep = [
                        {'type':'text', 'data':f"ROC Analysis: <b>{score}</b> vs <b>{truth}</b>"},
                        {'type':'table', 'header':'Key Statistics', 'data':pd.DataFrame([res]).T},
                        {'type':'plot', 'header':'ROC Curve', 'data':fig},
                        {'type':'table', 'header':'Coordinates', 'data':coords} # ไม่ใช้ head() ให้แสดงทั้งหมด
                    ]
                    html = diag_test.generate_report(f"ROC: {score}", rep)
                    st.components.v1.html(html, height=800, scrolling=True)
                    st.download_button("📥 Download Report", html, "roc.html", "text/html")

        # 2.2 CHI-SQUARE (Custom Table)
        with st2_2:
            st.markdown("##### Chi-Square Test")
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Row Variable:", all_cols, key='chi1')
            v2 = c2.selectbox("Column Variable:", all_cols, key='chi2')
            if st.button("Run Chi-Square"):
                # 🟢 ใช้ calculate_chi2_custom เพื่อสร้างตาราง 2x2 สวยงาม
                table_html, stats_info = diag_test.calculate_chi2_custom(df, v1, v2)
                
                full_html = diag_test.generate_report(f"Chi-square: {v1} vs {v2}", [
                    {'type':'text', 'data':stats_info},
                    {'type':'raw_html', 'header':'Contingency Table', 'data':table_html}
                ])
                st.components.v1.html(full_html, height=400, scrolling=True)
                st.download_button("📥 Download HTML Report", full_html, "chi2_report.html", "text/html")


        # 🟢 2.3 COMPARE MEANS (T-TEST/ANOVA)
        with st2_3:
            st.markdown("##### Compare Means (T-test / ANOVA)")
            c1, c2 = st.columns(2)
            num_var = c1.selectbox("Numeric Variable:", all_cols, key='tt_num')
            grp_var = c2.selectbox("Group Variable:", all_cols, key='tt_grp')
            
            if st.button("Run T-test / ANOVA"):
                desc_df, res_text = diag_test.calculate_ttest(df, num_var, grp_var)
                if desc_df is not None:
                    
                    report_elements = [
                        {'type':'text', 'data': res_text},
                        {'type':'table', 'header': f"Descriptive Statistics for {num_var}", 'data': desc_df}
                    ]
                    
                    html_report = diag_test.generate_report(f"Compare Means: {num_var} by {grp_var}", report_elements)
                    st.components.v1.html(html_report, height=500, scrolling=True)
                    st.download_button("📥 Download HTML Report", html_report, "ttest_report.html", "text/html")
                else:
                    st.error(res_text)


        # 2.4 DESCRIPTIVE
        with st2_4:
            st.markdown("##### Descriptive Statistics")
            dv = st.selectbox("Variable:", all_cols, key='desc')
            if st.button("Show Stats"):
                res = diag_test.calculate_descriptive(df, dv)
                if res is not None:
                    report_elements = [
                        {'type': 'table', 'header': f"Statistics for {dv}", 'data': res}
                    ]
                    html_report = diag_test.generate_report(f"Descriptive Statistics: {dv}", report_elements)
                    st.components.v1.html(html_report, height=500, scrolling=True)
                    st.download_button("📥 Download HTML Report", html_report, "desc_report.html", "text/html")
                else:
                    st.error(f"Cannot calculate statistics for {dv}.")

    # TAB 3: TABLE 1
    with tab3:
        st.subheader("3. Baseline Table 1")
        grp = st.selectbox("Group By:", ["None"]+all_cols)
        vars = st.multiselect("Variables:", all_cols, default=[c for c in all_cols if c!=grp])
        if st.button("Generate Table 1"):
            html = table_one.generate_table(df, vars, None if grp=="None" else grp, st.session_state.var_meta)
            st.components.v1.html(html, height=600, scrolling=True)
            st.download_button("Download", html, "table1.html")

else:
    st.info("Please load data.")
