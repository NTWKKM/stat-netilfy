import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import streamlit.components.v1 as components

# ✅ FIX #7-8: IMPORT CONFIG AND LOGGER (MINIMAL WIRING)
from config import CONFIG
from logger import get_logger, LoggerFactory

# ==========================================
# 1. CONFIG & LOADING SCREEN KILLER (Must be First)
# ==========================================
st.set_page_config(
    page_title=CONFIG.get('ui.page_title', 'Medical Stat Tool'),  # ✅ USE CONFIG
    layout=CONFIG.get('ui.layout', 'wide'),  # ✅ USE CONFIG
    menu_items={
        'Get Help': 'https://ntwkkm.github.io/pl/infos/stat_manual.html',
        'Report a bug': "https://github.com/NTWKKM/stat-netilfy/issues", 
    }
)

# Initialize logging system (once at app start)
@st.cache_resource(show_spinner=False)
def _init_logging() -> bool:
    """
    Configure the application's global logging and record that the app has started.
    """
    LoggerFactory.configure()
    get_logger(__name__).info("📱 Streamlit app started")
    return True

_init_logging()

# Get logger instance (after configuration)
logger = get_logger(__name__)

# ==========================================
# 1a. CHECK OPTIONAL DEPENDENCIES (FIX #1)
# ==========================================
@st.cache_resource(show_spinner=False)
def check_optional_deps():
    """
    Report presence of optional third-party dependencies used by the app.
    """
    deps_status = {}
    
    try:
        import firthlogist
        deps_status['firth'] = {'installed': True, 'msg': '✅ Firth regression enabled'}
    except ImportError:
        deps_status['firth'] = {'installed': False, 'msg': '⚠️ Firth regression unavailable - using Standard Logistic Regression (BFGS)'}

    get_logger(__name__).info("Optional dependencies: firth=%s", deps_status['firth']['installed'])
    return deps_status

if 'checked_deps' not in st.session_state:
    deps = check_optional_deps()
    st.session_state.checked_deps = True
    if not deps['firth']['installed']:
        st.info(deps['firth']['msg'])

# ==========================================
# 2. IMPORT MODULES
# ==========================================
try:
    from tabs import tab_data
    from tabs import tab_baseline_matching
    from tabs import tab_diag
    from tabs import tab_corr
    from tabs import tab_logit
    from tabs import tab_survival
    from tabs import tab_settings  # 🟢 NEW: Import Settings Tab
except (KeyboardInterrupt, SystemExit):
    raise
except Exception as e:
    logger.exception("Failed to import tabs")  # ✅ LOG ERROR
    st.exception(e)
    st.stop()

# --- INITIALIZE STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {}
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
    
# 🟢 NEW: Initialize matched data session state
if 'df_matched' not in st.session_state:
    st.session_state.df_matched = None
if 'is_matched' not in st.session_state:
    st.session_state.is_matched = False
if 'matched_treatment_col' not in st.session_state:
    st.session_state.matched_treatment_col = None
if 'matched_covariates' not in st.session_state:
    st.session_state.matched_covariates = []

# ==========================================
# 🟢 HELPER: SHINYSTAT SKILLS PROMO BANNER
# ==========================================
def render_skills_hero_banner(compact: bool = False):
    """Render the promotional banner for shinystat-skills on the main page."""
    if compact:
        with st.expander("🤖 **NEW! ใช้ AI วิเคราะห์สถิติการแพทย์อัตโนมัติด้วย `shinystat-skills` (Claude Code / Cursor / Cloud Sandbox)** — คลิกเพื่อดูวิธีติดตั้ง", expanded=False):
            col_a, col_b = st.columns([1.6, 1.4])
            with col_a:
                st.markdown("""
                **ShinyStat Biostatistical Skills** สังเคราะห์สมองกลสถิติจากเว็บแอปนี้ไปอยู่ใน AI Agent ของคุณ:
                - วิเคราะห์ **Table 1, Firth/Logistic, Cox PH, KM, PSM, ROC (DeLong), ICC, DCA, MICE** อัตโนมัติ
                - ออกรายงาน **Dual Interactive Plotly + 300 DPI Publication Figures (NEJM/JAMA)** และ Word/PPTX/HTML
                """)
                st.markdown("🔗 **[ดูรายละเอียดและติดดาวบน GitHub (ntwkkm/shinystat-skills) ↗](https://github.com/ntwkkm/shinystat-skills)**")
            with col_b:
                st.caption("📋 คัดลอกคำสั่งด้านล่างไปรันใน Terminal เพื่อติดตั้ง Skill:")
                st.code('npx skills add "https://github.com/ntwkkm/shinystat-skills"', language="bash")
        return

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); border: 1px solid #3b82f6; border-left: 6px solid #38bdf8; border-radius: 14px; padding: 20px 24px; margin-bottom: 14px; color: #f8fafc; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
            <span style="background: #38bdf8; color: #0f172a; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 999px; letter-spacing: 0.05em;">✨ NEW AI AGENT SKILL</span>
            <span style="color: #93c5fd; font-size: 0.85rem; font-weight: 600;">Autonomous Biostatistical Intelligence</span>
        </div>
        <h3 style="margin: 0 0 8px 0; color: #ffffff; font-size: 1.35rem; font-weight: 800;">
            🤖 เปลี่ยน AI ของคุณให้เป็นนักชีวสถิติคลินิกด้วย <code>shinystat-skills</code>
        </h3>
        <p style="margin: 0 0 14px 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.55;">
            Skill ที่สังเคราะห์จาก Logic ของเว็บแอปนี้โดยตรง สำหรับ <b>Claude Web / Claude Code / Cursor / Codex / Gemini</b> ช่วยตรวจข้อมูลและรันสถิติการแพทย์อัตโนมัติ (<b>Table 1, Logistic & Firth, Cox PH, Kaplan-Meier, PSM, ROC & DeLong, Bland-Altman, ICC, Calibration, DCA, MICE</b>) พร้อมสร้างกราฟ <b>300 DPI มาตรฐานวารสาร NEJM/JAMA</b> และออกรายงาน HTML/DOCX/PPTX ในคำสั่งเดียว
        </p>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px;">
            <span style="font-size: 0.78rem; background: rgba(56, 189, 248, 0.15); color: #7dd3fc; padding: 4px 10px; border-radius: 6px; font-weight: 600;">✓ Zero Custom Dependency</span>
            <span style="font-size: 0.78rem; background: rgba(56, 189, 248, 0.15); color: #7dd3fc; padding: 4px 10px; border-radius: 6px; font-weight: 600;">✓ Strict 0/1 & EPV Safety Invariants</span>
            <span style="font-size: 0.78rem; background: rgba(56, 189, 248, 0.15); color: #7dd3fc; padding: 4px 10px; border-radius: 6px; font-weight: 600;">✓ Dual Interactive Plotly + 300 DPI Report</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.caption("👇 **กดปุ่ม Copy มุมขวาของกล่องโค้ดนี้** เพื่อติดตั้งลงใน AI Agent ของคุณทันที:")
        st.code('npx skills add "https://github.com/ntwkkm/shinystat-skills"', language="bash")
    with c2:
        st.caption("⭐ **ดูคู่มือและซอร์สโค้ดทั้งหมด:**")
        st.link_button(
            "🌟 เปิด GitHub: ntwkkm/shinystat-skills ↗",
            "https://github.com/ntwkkm/shinystat-skills",
            use_container_width=True,
            type="primary"
        )
    st.divider()

# --- SIDEBAR ---
st.sidebar.title("MENU")

# 🟢 NEW: PROMO SECTION (STATIOMED + SHINYSTAT-SKILLS)
st.sidebar.success("""
**🚀 New Version Available StatioMed**:  
👉 [HuggingFace Space](https://ntwkkm-shinystat.hf.space)  
👉 [Posit Cloud](https://ntwkkm-shinymedstat.share.connect.posit.cloud/)
""")

# 🟢 NEW: SHINYSTAT-SKILLS SIDEBAR PROMO CARD
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 12px 14px; border-radius: 10px; border: 1px solid #38bdf8; margin-bottom: 8px;">
    <div style="margin-bottom: 4px;">
        <span style="background: #38bdf8; color: #0f172a; font-size: 0.68rem; font-weight: 800; padding: 2px 7px; border-radius: 999px;">✨ AI AGENT SKILL</span>
    </div>
    <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem; margin-bottom: 4px;">
        🤖 ShinyStat Skills
    </div>
    <div style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.4;">
        สังเคราะห์จากเว็บแอปนี้! เปลี่ยน AI Agent ให้รันสถิติคลินิก + กราฟ NEJM/JAMA 300 DPI อัตโนมัติ
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("📋 กด Copy คำสั่งติดตั้ง Skill:")
st.sidebar.code('npx skills add "https://github.com/ntwkkm/shinystat-skills"', language="bash")
st.sidebar.link_button(
    "⭐ GitHub: shinystat-skills ↗",
    "https://github.com/ntwkkm/shinystat-skills",
    use_container_width=True
)
st.sidebar.divider()
# 🔴 END PROMO SECTION

st.sidebar.header("1. Data Management")

# Example Data Generator
if st.sidebar.button("📄 Load Example Data"):
    logger.log_operation("example_data", "started", n_rows=600)  
    
    try:
        with logger.track_time("generate_example_data", log_level="debug"): 
            np.random.seed(42) # Fixed seed for reproducibility
            n = 600 
            
            # --- 1. Demographics (Baseline) ---
            age = np.random.normal(60, 12, n).astype(int).clip(30, 95)
            sex = np.random.binomial(1, 0.5, n)
            bmi = np.random.normal(25, 5, n).round(1).clip(15, 50)
            
            # --- 2. Create Confounding for PSM (Selection Bias) ---
            logit_treat = -4.5 + (0.05 * age) + (0.08 * bmi) + (0.2 * sex)
            p_treat = 1 / (1 + np.exp(-logit_treat))
            group = np.random.binomial(1, p_treat, n)
            
            # --- 3. Comorbidities ---
            logit_dm = -5 + (0.04 * age) + (0.1 * bmi)
            p_dm = 1 / (1 + np.exp(-logit_dm))
            diabetes = np.random.binomial(1, p_dm, n)
            
            logit_ht = -4 + (0.06 * age) + (0.05 * bmi)
            p_ht = 1 / (1 + np.exp(-logit_ht))
            hypertension = np.random.binomial(1, p_ht, n)

            # --- 4. Survival Outcome ---
            lambda_base = 0.002 
            linear_predictor = 0.03 * age + 0.4 * diabetes + 0.3 * hypertension - 0.6 * group
            hazard = lambda_base * np.exp(linear_predictor)
            surv_time = np.random.exponential(1/hazard, n)
            censor_time = np.random.uniform(0, 100, n)
            time_obs = np.minimum(surv_time, censor_time).round(1)
            time_obs = np.maximum(time_obs, 0.5)
            status_death = (surv_time <= censor_time).astype(int)
            
            # --- 5. Binary Outcome ---
            logit_cure = 0.5 + 1.2 * group - 0.04 * age - 0.5 * diabetes
            p_cure = 1 / (1 + np.exp(-logit_cure))
            outcome_cured = np.random.binomial(1, p_cure, n)

            # --- 6. Diagnostic Test Data ---
            gold_std = np.random.binomial(1, 0.3, n)
            rapid_score = np.where(gold_std==0, 
                                   np.random.normal(20, 10, n), 
                                   np.random.normal(50, 15, n))
            rapid_score = np.clip(rapid_score, 0, 100).round(1)
            
            rater_a = np.where(gold_std==1, 
                               np.random.binomial(1, 0.85, n), 
                               np.random.binomial(1, 0.10, n))
            
            agree_prob = 0.85
            rater_b = np.where(np.random.binomial(1, agree_prob, n)==1, 
                               rater_a, 
                               1 - rater_a)

            # --- 7. Correlation Data ---
            hba1c = np.random.normal(6.5, 1.5, n).clip(4, 14).round(1)
            glucose = (hba1c * 15) + np.random.normal(0, 15, n)
            glucose = glucose.round(0)

            # --- 8. ICC Data ---
            icc_rater1 = np.random.normal(120, 15, n).round(1)
            icc_rater2 = icc_rater1 + 5 + np.random.normal(0, 4, n)
            icc_rater2 = icc_rater2.round(1)

            # Create DataFrame
            data = {
                'ID': range(1, n+1),
                'Treatment_Group': group,  
                'Age_Years': age,
                'Sex_Male': sex,
                'BMI_kgm2': bmi,
                'Comorb_Diabetes': diabetes,
                'Comorb_Hypertension': hypertension,
                'Outcome_Cured': outcome_cured,
                'Time_Months': time_obs,
                'Status_Death': status_death,
                'Gold_Standard_Disease': gold_std,
                'Test_Score_Rapid': rapid_score, 
                'Diagnosis_Dr_A': rater_a,
                'Diagnosis_Dr_B': rater_b,
                'Lab_HbA1c': hba1c,
                'Lab_Glucose': glucose,
                'ICC_SysBP_Rater1': icc_rater1,
                'ICC_SysBP_Rater2': icc_rater2,
            }
            
            st.session_state.df = pd.DataFrame(data)
        
        # Set Metadata
        st.session_state.var_meta = {
            'Treatment_Group': {'type':'Categorical', 'map':{0:'Standard Care', 1:'New Drug'}, 'label': 'Treatment Group'},
            'Sex_Male': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}, 'label': 'Sex'},
            'Comorb_Diabetes': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}, 'label': 'Diabetes'},
            'Comorb_Hypertension': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}, 'label': 'Hypertension'},
            'Outcome_Cured': {'type':'Categorical', 'map':{0:'Not Cured', 1:'Cured'}, 'label': 'Outcome (Cured)'},
            'Status_Death': {'type':'Categorical', 'map':{0:'Censored/Alive', 1:'Dead'}, 'label': 'Status (Death)'},
            'Gold_Standard_Disease': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}, 'label': 'Gold Standard'},
            'Diagnosis_Dr_A': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}, 'label': 'Diagnosis (Dr. A)'},
            'Diagnosis_Dr_B': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}, 'label': 'Diagnosis (Dr. B)'},
            
            'Age_Years': {'type': 'Continuous', 'label': 'Age (Years)', 'map': {}},
            'BMI_kgm2': {'type': 'Continuous', 'label': 'BMI (kg/m²)', 'map': {}},
            'Time_Months': {'type': 'Continuous', 'label': 'Time (Months)', 'map': {}},
            'Test_Score_Rapid': {'type': 'Continuous', 'label': 'Rapid Test Score (0-100)', 'map': {}},
            'Lab_HbA1c': {'type': 'Continuous', 'label': 'HbA1c (%)', 'map': {}},
            'Lab_Glucose': {'type': 'Continuous', 'label': 'Fasting Glucose (mg/dL)', 'map': {}},
            'ICC_SysBP_Rater1': {'type': 'Continuous', 'label': 'Sys BP (Rater 1)', 'map': {}},
            'ICC_SysBP_Rater2': {'type': 'Continuous', 'label': 'Sys BP (Rater 2)', 'map': {}},
        }
        st.session_state.uploaded_file_name = "Example Clinical Data"
        
        logger.log_operation("example_data", "completed", 
                             rows=len(st.session_state.df),
                             columns=len(st.session_state.df.columns))
        st.sidebar.success(f"✅ Loaded {n} Clinical Records (Simulated)")
        st.rerun()
        
    except Exception as e:
        logger.log_operation("example_data", "failed", error=str(e)) 
        st.sidebar.error(f"Error loading example data: {e}")
    
# File Uploader
upl = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if upl:
    data_bytes = upl.getvalue()
    file_size_mb = len(data_bytes) / 1e6
    logger.log_operation("file_upload", "started",   # ✅ LOG START
                         filename=upl.name, 
                         size=f"{file_size_mb:.1f}MB")
    
    try:
        file_sig = (upl.name, hashlib.sha256(data_bytes).hexdigest())
        
        if st.session_state.get('uploaded_file_sig') != file_sig:
            with logger.track_time("file_parse", log_level="debug"):  # ✅ TRACK TIMING
                if upl.name.lower().endswith('.csv'):
                    new_df = pd.read_csv(io.BytesIO(data_bytes))
                else:
                    new_df = pd.read_excel(io.BytesIO(data_bytes))
            
            st.session_state.df = new_df
            st.session_state.uploaded_file_name = upl.name
            st.session_state.uploaded_file_sig = file_sig
            
            # FIX #6: PRESERVE METADATA ON UPLOAD
            current_meta = {}
            for col in new_df.columns:
                if col in st.session_state.var_meta:
                    current_meta[col] = st.session_state.var_meta[col]
                else:
                    if pd.api.types.is_numeric_dtype(new_df[col]):
                        unique_vals = new_df[col].dropna().unique()
                        unique_count = len(unique_vals)
                        
                        if unique_count < 10:
                            try:
                                decimals_count = sum(1 for v in unique_vals if not float(v).is_integer())
                            except (ValueError, TypeError):
                                decimals_count = 0
                            decimals_pct = decimals_count / len(unique_vals) if len(unique_vals) > 0 else 0
                            
                            if decimals_pct < 0.3:
                                current_meta[col] = {'type': 'Categorical', 'label': col, 'map': {}, 'confidence': 'auto'}
                            else:
                                current_meta[col] = {'type': 'Continuous', 'label': col, 'map': {}, 'confidence': 'auto'}
                        else:
                            current_meta[col] = {'type': 'Continuous', 'label': col, 'map': {}, 'confidence': 'auto'}
                    else:
                        current_meta[col] = {'type': 'Categorical', 'label': col, 'map': {}, 'confidence': 'auto'}

            st.session_state.var_meta = current_meta
            
            logger.log_operation("file_upload", "completed",  # ✅ LOG COMPLETION
                                rows=len(new_df), columns=len(new_df.columns))
            st.sidebar.success("File Uploaded and Metadata Initialized!")
            st.rerun()
        
        else:
            st.sidebar.info("File already loaded.")
            
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, ImportError, Exception) as e:
        logger.log_operation("file_upload", "failed", error=str(e))  # ✅ LOG ERROR
        st.sidebar.error(f"Error: {e}")
        st.session_state.df = None
        st.session_state.uploaded_file_name = None
        st.session_state.uploaded_file_sig = None

# 🟢 NEW: Reset/Clear Matched Data Button
if st.session_state.is_matched:
    if st.sidebar.button("🔄 Clear Matched Data", type="secondary"):
        logger.info("🔄 User cleared matched data")
        st.session_state.df_matched = None
        st.session_state.is_matched = False
        st.session_state.matched_treatment_col = None
        st.session_state.matched_covariates = []
        st.rerun()

if st.sidebar.button("⚠️ Reset All Data", type="primary"):
    logger.info("🔄 User reset all data")  # ✅ LOG RESET
    st.session_state.clear()
    st.rerun()

# Variable Settings (Metadata)
if st.session_state.df is not None:
    st.sidebar.header("2. Variable Metadata") # 🟢 Renamed to avoid confusion with new Settings Tab
    cols = st.session_state.df.columns.tolist()
    
    # Auto detect for select box labeling
    auto_detect_meta = {c: st.session_state.var_meta.get(c, {'type': 'Auto-detect', 'map': {}}).get('type', 'Auto-detect') for c in cols}
    
    s_var = st.sidebar.selectbox("Edit Var:", ["Select...", *cols])
    if s_var != "Select...":
        if s_var not in st.session_state.var_meta:
            is_numeric = pd.api.types.is_numeric_dtype(st.session_state.df[s_var]) if s_var in st.session_state.df.columns else False
            initial_type = 'Continuous' if is_numeric else 'Categorical'
            st.session_state.var_meta[s_var] = {'type': initial_type, 'label': s_var, 'map': {}}

        meta = st.session_state.var_meta.get(s_var, {})
        
        current_type = meta.get('type', 'Auto-detect')
        if current_type == 'Auto-detect':
            is_numeric = pd.api.types.is_numeric_dtype(st.session_state.df[s_var]) if s_var in st.session_state.df.columns else False
            current_type = 'Continuous' if is_numeric else 'Categorical'

        allowed_types = ['Categorical', 'Continuous']
        if current_type not in allowed_types:
            current_type = 'Categorical'

        n_type = st.sidebar.radio(
            "Type:",
            allowed_types,
            index=allowed_types.index(current_type),
        )
                                            
        st.sidebar.markdown("Labels (0=No):")
        map_txt = st.sidebar.text_area("Map", value="\n".join([f"{k}={v}" for k,v in meta.get('map',{}).items()]), height=80)
        
        if st.sidebar.button("💾 Save"):
            new_map = {}
            for line in map_txt.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    try:
                        k = k.strip()
                        try:
                            k_num = float(k)
                            k = int(k_num) if k_num.is_integer() else k_num
                        except ValueError:
                            pass
                        new_map[k] = v.strip()
                    except (TypeError, ValueError) as e:
                        st.sidebar.warning(f"Skipping invalid map line '{line}': {e}")
            
            if s_var not in st.session_state.var_meta: 
                st.session_state.var_meta[s_var] = {}
            
            st.session_state.var_meta[s_var]['type'] = n_type
            st.session_state.var_meta[s_var]['map'] = new_map
            st.session_state.var_meta[s_var].setdefault('label', s_var)
            
            logger.info("✅ Variable '%s' configured as %s", s_var, n_type)  # ✅ LOG CONFIG
            st.sidebar.success("Saved!")
            st.rerun()

# ==========================================
# MAIN AREA - TABS (7 TOTAL)
# ==========================================
if st.session_state.df is not None:
    df = st.session_state.df 

    # 🟢 Show Compact Promo Banner at top of workspace when data is loaded
    render_skills_hero_banner(compact=True)
    
    cols_to_verify = [c for c in st.session_state.var_meta if st.session_state.var_meta[c].get('confidence') == 'auto']
    if cols_to_verify:
        with st.expander("⚠️ Auto-Detected Variable Types (Please Verify)", expanded=False):
            st.info(f"The following {len(cols_to_verify)} column(s) were auto-detected. Please verify they are correct in the sidebar Settings:")
            for col in cols_to_verify[:10]:
                detected_type = st.session_state.var_meta[col]['type']
                st.caption(f"  • **{col}**: {detected_type}")
            if len(cols_to_verify) > 10:
                st.caption(f"  ... and {len(cols_to_verify) - 10} more")

    # 🟢 Display Matched Data Status
    if st.session_state.is_matched and st.session_state.df_matched is not None:
        st.info(f"""
        ✅ **Matched Dataset Active**
        - Original data: {len(df)} rows
        - Matched data: {len(st.session_state.df_matched)} rows ({len(df) - len(st.session_state.df_matched)} rows excluded)
        - Treatment: {st.session_state.matched_treatment_col}
        - Use dropdown in each tab to select **"✅ Matched Data"** for analysis
        """)
        
    # 🟢 FINAL TAB LAYOUT (Now 7 Tabs)
    t0, t1, t2, t3, t4, t5, t6 = st.tabs([
        "📁 Data Management", 
        "📋 Table 1 & Matching", 
        "🧪 Diagnostic Tests (ROC)",
        "📈 Correlation & ICC",
        "📊 Risk Factors (Logistic)",
        "⏳ Survival Analysis (KM & Cox)",
        "⚙️ Settings" # 🟢 NEW TAB
    ])

    # ---------------------------------------------------------
    # TAB 0: Data Management & Cleaning (MASTER LOGIC)
    # ---------------------------------------------------------
    with t0:
        # 1. User views/edits raw data (still has symbols like >100)
        st.session_state.df = tab_data.render(df) 
        
        # 2. Prepare Clean Data for Analysis Tabs
        custom_na = st.session_state.get('custom_na_list', [])
        # This converts >100 to 100.0, etc.
        df_clean = tab_data.get_clean_data(st.session_state.df, custom_na)

    # ---------------------------------------------------------
    # TAB 1-5: Use Cleaned Data (df_clean)
    # ---------------------------------------------------------
    with t1:
        tab_baseline_matching.render(df_clean, st.session_state.var_meta)
        
    with t2:
        tab_diag.render(df_clean, st.session_state.var_meta)
        
    with t3:
        # ✅ Fixed: Added st.session_state.var_meta argument
        tab_corr.render(df_clean, st.session_state.var_meta)
        
    with t4:
        tab_logit.render(df_clean, st.session_state.var_meta)
        
    with t5:
        tab_survival.render(df_clean, st.session_state.var_meta)

    # ---------------------------------------------------------
    # TAB 6: Settings (Global Config)
    # ---------------------------------------------------------
    with t6:
        tab_settings.render()
        
else:
    # 🟢 Show Full Hero Promo Banner on Landing Page (Before Data Load)
    render_skills_hero_banner(compact=False)

    st.info("👈 Please load example data or upload a file to start.")
    st.markdown("""
### ✨ 7-Tab Analysis Pipeline:

1. **📁 Data Management** - Upload, clean, set variable types
2. **📋 Table 1 & Matching** - Baseline characteristics + Propensity Score Matching
3. **🧪 Diagnostic Tests (ROC)** - Chi-Square, ROC, Kappa, RR/OR/NNT
4. **📈 Correlation & ICC** - Pearson, Spearman, ICC reliability
5. **📊 Risk Factors (Logistic)** - Binary logistic regression
6. **⏳ Survival Analysis** - Kaplan-Meier & Cox regression
7. **⚙️ Settings** - Configure system parameters, UI, and analysis defaults
    """)
    
# ==========================================
# GLOBAL CSS
# ==========================================

st.markdown("""
<style>
footer {
    visibility: hidden;
    height: 0px;
}
footer:after {
    content: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<hr style="margin-top: 20px; margin-bottom: 10px; border-color: var(--border-color); opacity: 0.5;">
<div style='text-align: center; font-size: 0.8em; color: var(--text-color); opacity: 0.8;'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit; font-weight:bold;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
</div>
""", unsafe_allow_html=True)
