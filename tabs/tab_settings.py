import streamlit as st
import pandas as pd
from config import CONFIG
from tabs._common import get_color_palette

def render():
    st.title("⚙️ System Configuration")
    st.info("💡 Note: Changes made here affect the current runtime session immediately. Some UI changes may require a page refresh.")

    # Create tabs matching the structure of config.py + Colors reference
    tabs = st.tabs([
        "📊 Analysis", 
        "🎨 UI & Display", 
        "📝 Logging", 
        "⚡ Performance", 
        "🛠️ Advanced", # For Validation & Debug
        "🌈 Colors"
    ])
    
    tab_analysis, tab_ui, tab_logging, tab_perf, tab_adv, tab_colors = tabs

    # ==========================================
    # 1. TAB: ANALYSIS SETTINGS
    # ==========================================
    with tab_analysis:
        st.header("Analysis Parameters")
        st.markdown("""
        🎓 **Getting Started**: Configure statistical methods and thresholds below. 
        Hover over field labels for detailed explanations. All changes apply immediately.
        """)

        # --- Logistic Regression ---
        with st.expander("🔹 Logistic Regression", expanded=True):
            st.markdown("""
            **Purpose**: Configure settings for logistic regression models (binary and multinomial outcomes).
            
            **When to use which method**:
            - **Auto**: Automatically selects the best method based on data (recommended for beginners)
            - **Firth**: More stable for small samples or rare events (~<20 cases per group)
            - **BFGS**: Faster convergence, good for large datasets
            - **Default**: Standard ML estimation (may fail with separation or rare events)
            
            **Tips for non-statisticians**:
            - Start with "Auto" method
            - Lower screening P-value = stricter variable selection (typical: 0.05-0.20)
            - Increase max iterations if model fails to converge
            - Min cases ensures multivariate model stability (rule of thumb: ≥10-20 events per variable)
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                curr_method = CONFIG.get('analysis.logit_method')
                new_method = st.selectbox(
                    "Method", 
                    ['auto', 'firth', 'bfgs', 'default'],
                    index=['auto', 'firth', 'bfgs', 'default'].index(curr_method),
                    key='an_logit_met',
                    help="Algorithm for coefficient estimation. 'Auto' recommended for most users."
                )
                if new_method != curr_method: CONFIG.update('analysis.logit_method', new_method)

                curr_p = CONFIG.get('analysis.logit_screening_p')
                new_p = st.number_input(
                    "Screening P-value", 
                    0.0, 1.0, float(curr_p), 0.01, 
                    format="%.3f", 
                    key='an_scr_p',
                    help="For univariate screening. Variables with p > threshold excluded from multivariate. Typical: 0.05-0.20"
                )
                if new_p != curr_p: CONFIG.update('analysis.logit_screening_p', new_p)

            with c2:
                curr_iter = CONFIG.get('analysis.logit_max_iter')
                new_iter = st.number_input(
                    "Max Iterations", 
                    10, 5000, int(curr_iter), 10, 
                    key='an_max_it',
                    help="Maximum iterations for algorithm convergence. Increase if model fails to converge."
                )
                if new_iter != curr_iter: CONFIG.update('analysis.logit_max_iter', new_iter)

                curr_min = CONFIG.get('analysis.logit_min_cases')
                new_min = st.number_input(
                    "Min Cases for Multivariate", 
                    1, 100, int(curr_min), 
                    key='an_min_cs',
                    help="Minimum events required for multivariate analysis. Ensures model stability (rule: 10-20 per variable)."
                )
                if new_min != curr_min: CONFIG.update('analysis.logit_min_cases', new_min)

        # --- Survival Analysis ---
        with st.expander("🔹 Survival Analysis"):
            st.markdown("""
            **Purpose**: Configure settings for time-to-event (survival) analyses.
            
            **Kaplan-Meier vs Weibull**:
            - **Kaplan-Meier**: Non-parametric, no distribution assumption (recommended for most studies)
            - **Weibull**: Parametric, assumes Weibull distribution (smaller sample sizes, better for extrapolation)
            
            **Cox Method (tie handling)**:
            - **Efron**: More accurate for tied event times (recommended)
            - **Breslow**: Computationally faster, less accurate with many ties
            
            **For beginners**: Use Kaplan-Meier + Efron for most analyses.
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                curr_surv = CONFIG.get('analysis.survival_method')
                new_surv = st.selectbox(
                    "Survival Method", 
                    ['kaplan-meier', 'weibull'], 
                    index=['kaplan-meier', 'weibull'].index(curr_surv), 
                    key='an_surv_m',
                    help="Kaplan-Meier (non-parametric) recommended for most studies."
                )
                if new_surv != curr_surv: CONFIG.update('analysis.survival_method', new_surv)
                
            with c2:
                curr_cox = CONFIG.get('analysis.cox_method')
                new_cox = st.selectbox(
                    "Cox Method (Tie Handling)", 
                    ['efron', 'breslow'], 
                    index=['efron', 'breslow'].index(curr_cox), 
                    key='an_cox_m',
                    help="Efron: more accurate with ties. Breslow: faster, less accurate."
                )
                if new_cox != curr_cox: CONFIG.update('analysis.cox_method', new_cox)

        # --- Data Detection & Missing ---
        with st.expander("🔹 Data Handling & Detection"):
            st.markdown("""
            **Purpose**: Configure how the system automatically detects variable types and handles missing data.
            
            **Variable Type Detection**:
            - Variables with ≤ threshold unique values → treated as categorical
            - Variables with > threshold unique values → treated as continuous
            - Example: "Unique Value Threshold = 10" means variables with 10 or fewer unique values are categorical
            
            **Decimal % Threshold**: If a continuous variable has > threshold% decimal values → treated as continuous (not integer-only)
            
            **Missing Data Strategy**:
            - **Complete-case**: Excludes rows with any missing values (standard for most analyses)
            - **Drop**: Similar to complete-case but with different handling details
            
            **Missing Flag Threshold**: Variables with > threshold% missing → flagged in report
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                # Variable Detection
                curr_vth = CONFIG.get('analysis.var_detect_threshold')
                new_vth = st.number_input(
                    "Unique Value Threshold (Cat vs Cont)", 
                    1, 50, int(curr_vth), 
                    key='an_var_th',
                    help="Variables with ≤ this threshold → categorical. Example: threshold=10, variable with 5 unique values = categorical."
                )
                if new_vth != curr_vth: CONFIG.update('analysis.var_detect_threshold', new_vth)
                
                curr_dec = CONFIG.get('analysis.var_detect_decimal_pct')
                new_dec = st.number_input(
                    "Decimal % Threshold", 
                    0.0, 1.0, float(curr_dec), 0.05, 
                    format="%.2f",
                    key='an_var_dec',
                    help="If % of decimal values in continuous variable > threshold → classified as continuous (e.g., 0.15 = 15%)."
                )
                if new_dec != curr_dec: CONFIG.update('analysis.var_detect_decimal_pct', new_dec)

            with c2:
                # Missing Data
                curr_mstrat = CONFIG.get('analysis.missing_strategy')
                new_mstrat = st.selectbox(
                    "Missing Data Strategy", 
                    ['complete-case', 'drop'], 
                    index=['complete-case', 'drop'].index(curr_mstrat), 
                    key='an_mis_str',
                    help="Complete-case: exclude rows with any missing values (recommended for most analyses)."
                )
                if new_mstrat != curr_mstrat: CONFIG.update('analysis.missing_strategy', new_mstrat)

                curr_mpct = CONFIG.get('analysis.missing_threshold_pct')
                new_mpct = st.number_input(
                    "Missing Flag Threshold (%)", 
                    0, 100, int(curr_mpct), 
                    key='an_mis_pct',
                    help="Variables with > threshold% missing values → flagged in report. Example: 20 = flag if >20% missing."
                )
                if new_mpct != curr_mpct: CONFIG.update('analysis.missing_threshold_pct', new_mpct)

        # --- P-value Handling ---
        with st.expander("🔹 P-value Formatting & Bounds (NEJM Standard)"):
            st.markdown("""
            **Purpose**: Configure p-value display format following New England Journal of Medicine (NEJM) standards.
            
            **NEJM Standard Formatting**:
            - P < 0.001 displayed as "<0.001" (not exact value)
            - P reported to 3 decimal places (0.001, 0.042, 0.123)
            - P > 0.99 displayed as ">0.99"
            
            **Lower Bound**: Smallest displayable p-value (typically 0.001 for NEJM)
            
            **Upper Bound**: Largest displayable p-value (typically 0.99 for NEJM)
            
            **Clip Tolerance**: Rounding tolerance for very small/large values
            
            **Recommended NEJM Settings**:
            - Lower Bound: 0.001
            - Upper Bound: 0.99
            - Clip Tolerance: 0.00001
            - Small P Format: "<0.001"
            - Large P Format: ">0.99"
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                curr_pl = CONFIG.get('analysis.pvalue_bounds_lower')
                new_pl = st.number_input(
                    "Lower Bound", 
                    0.0, 1.0, float(curr_pl), 0.001, 
                    format="%.4f",
                    key='an_pv_low',
                    help="NEJM standard: 0.001. Smallest p-value shown (smaller → show as format string)."
                )
                if new_pl != curr_pl: CONFIG.update('analysis.pvalue_bounds_lower', new_pl)

                curr_tol = CONFIG.get('analysis.pvalue_clip_tolerance')
                new_tol = st.number_input(
                    "Clip Tolerance", 
                    0.0, 0.1, float(curr_tol), 0.00001, 
                    format="%.6f", 
                    key='an_pv_tol',
                    help="Rounding tolerance for extreme p-values. NEJM standard: 0.00001"
                )
                if new_tol != curr_tol: CONFIG.update('analysis.pvalue_clip_tolerance', new_tol)

                curr_fs = CONFIG.get('analysis.pvalue_format_small')
                new_fs = st.text_input(
                    "Small P Format (e.g. <0.001)", 
                    curr_fs, 
                    key='an_pv_fs',
                    help="NEJM standard: '<0.001'. Display format for p-values below lower bound."
                )
                if new_fs != curr_fs: CONFIG.update('analysis.pvalue_format_small', new_fs)
            
            with c2:
                curr_pu = CONFIG.get('analysis.pvalue_bounds_upper')
                new_pu = st.number_input(
                    "Upper Bound", 
                    0.0, 1.0, float(curr_pu), 0.001, 
                    format="%.4f",
                    key='an_pv_up',
                    help="NEJM standard: 0.99. Largest p-value shown (larger → show as format string)."
                )
                if new_pu != curr_pu: CONFIG.update('analysis.pvalue_bounds_upper', new_pu)

                curr_fl = CONFIG.get('analysis.pvalue_format_large')
                new_fl = st.text_input(
                    "Large P Format (e.g. >0.99)", 
                    curr_fl, 
                    key='an_pv_fl',
                    help="NEJM standard: '>0.99'. Display format for p-values above upper bound."
                )
                if new_fl != curr_fl: CONFIG.update('analysis.pvalue_format_large', new_fl)


    # ==========================================
    # 2. TAB: UI & DISPLAY SETTINGS
    # ==========================================
    with tab_ui:
        st.header("UI & Display Settings")

        # --- General Page Setup ---
        with st.expander("🔹 Page Setup", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                curr_title = CONFIG.get('ui.page_title')
                new_title = st.text_input(
                    "Page Title", 
                    curr_title, 
                    key='ui_title',
                    help="Title shown in browser tab and page header."
                )
                if new_title != curr_title: CONFIG.update('ui.page_title', new_title)
                
                curr_theme = CONFIG.get('ui.theme')
                new_theme = st.selectbox(
                    "Theme", 
                    ['light', 'dark', 'auto'], 
                    index=['light', 'dark', 'auto'].index(curr_theme), 
                    key='ui_theme',
                    help="Light: bright colors. Dark: dark background. Auto: follows system preference."
                )
                if new_theme != curr_theme: CONFIG.update('ui.theme', new_theme)

            with c2:
                curr_layout = CONFIG.get('ui.layout')
                new_layout = st.selectbox(
                    "Layout", 
                    ['wide', 'centered'], 
                    index=['wide', 'centered'].index(curr_layout), 
                    key='ui_layout',
                    help="Wide: use full screen width. Centered: constrain to max width."
                )
                if new_layout != curr_layout: CONFIG.update('ui.layout', new_layout)

        # --- Sidebar ---
        with st.expander("🔹 Sidebar"):
            c1, c2 = st.columns(2)
            with c1:
                curr_sw = CONFIG.get('ui.sidebar_width')
                new_sw = st.number_input(
                    "Sidebar Width (px)", 
                    100, 500, int(curr_sw), 10, 
                    key='ui_sw',
                    help="Width of the left navigation sidebar in pixels."
                )
                if new_sw != curr_sw: CONFIG.update('ui.sidebar_width', new_sw)
            with c2:
                curr_logo = CONFIG.get('ui.show_sidebar_logo')
                new_logo = st.toggle(
                    "Show Sidebar Logo", 
                    curr_logo, 
                    key='ui_logo',
                    help="Display logo in the sidebar."
                )
                if new_logo != curr_logo: CONFIG.update('ui.show_sidebar_logo', new_logo)

        # --- Tables ---
        with st.expander("🔹 Tables"):
            c1, c2 = st.columns(2)
            with c1:
                curr_tr = CONFIG.get('ui.table_max_rows')
                new_tr = st.number_input(
                    "Max Table Rows", 
                    10, 10000, int(curr_tr), 100, 
                    key='ui_tr',
                    help="Maximum rows to display before pagination. Reduces page load time."
                )
                if new_tr != curr_tr: CONFIG.update('ui.table_max_rows', new_tr)
                
                curr_tpag = CONFIG.get('ui.table_pagination')
                new_tpag = st.toggle(
                    "Enable Pagination", 
                    curr_tpag, 
                    key='ui_tpag',
                    help="Split large tables into pages."
                )
                if new_tpag != curr_tpag: CONFIG.update('ui.table_pagination', new_tpag)

            with c2:
                curr_tdec = CONFIG.get('ui.table_decimal_places')
                new_tdec = st.number_input(
                    "Decimal Places", 
                    0, 10, int(curr_tdec), 
                    key='ui_tdec',
                    help="Number of decimal places in table numeric values. Example: 3 = 0.123"
                )
                if new_tdec != curr_tdec: CONFIG.update('ui.table_decimal_places', new_tdec)

        # --- Plots ---
        with st.expander("🔹 Plots"):
            c1, c2 = st.columns(2)
            with c1:
                curr_pw = CONFIG.get('ui.plot_width')
                new_pw = st.number_input(
                    "Plot Width", 
                    5, 50, int(curr_pw), 
                    key='ui_pw',
                    help="Width of plots in inches. Typical: 10-14 for wide displays."
                )
                if new_pw != curr_pw: CONFIG.update('ui.plot_width', new_pw)

                curr_pdpi = CONFIG.get('ui.plot_dpi')
                new_pdpi = st.number_input(
                    "Plot DPI", 
                    50, 600, int(curr_pdpi), 10, 
                    key='ui_pdpi',
                    help="Resolution in dots per inch. Higher = sharper but larger file. Typical: 100-300"
                )
                if new_pdpi != curr_pdpi: CONFIG.update('ui.plot_dpi', new_pdpi)

            with c2:
                curr_ph = CONFIG.get('ui.plot_height')
                new_ph = st.number_input(
                    "Plot Height", 
                    3, 30, int(curr_ph), 
                    key='ui_ph',
                    help="Height of plots in inches. Typical: 5-8 for readable charts."
                )
                if new_ph != curr_ph: CONFIG.update('ui.plot_height', new_ph)

                curr_pstyle = CONFIG.get('ui.plot_style')
                new_pstyle = st.text_input(
                    "Plot Style (e.g. seaborn)", 
                    curr_pstyle, 
                    key='ui_pstyle',
                    help="Matplotlib style. Options: seaborn, ggplot, bmh, etc."
                )
                if new_pstyle != curr_pstyle: CONFIG.update('ui.plot_style', new_pstyle)


    # ==========================================
    # 3. TAB: LOGGING SETTINGS
    # ==========================================
    with tab_logging:
        st.header("Logging Configuration")

        # Global Logging
        c1, c2 = st.columns(2)
        with c1:
            curr_log_en = CONFIG.get('logging.enabled')
            new_log_en = st.toggle(
                "Enable Logging System", 
                curr_log_en, 
                key='log_en',
                help="Record application events for debugging and monitoring."
            )
            if new_log_en != curr_log_en: CONFIG.update('logging.enabled', new_log_en)
        with c2:
            curr_log_lvl = CONFIG.get('logging.level')
            new_log_lvl = st.selectbox(
                "Global Log Level", 
                ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(curr_log_lvl), 
                key='log_lvl',
                help="DEBUG: most verbose, CRITICAL: least verbose."
            )
            if new_log_lvl != curr_log_lvl: CONFIG.update('logging.level', new_log_lvl)

        st.divider()

        # Detailed Logging Options
        col_file, col_console, col_st = st.columns(3)
        
        # File Logging
        with col_file:
            st.subheader("File Logging")
            curr_f_en = CONFIG.get('logging.file_enabled')
            new_f_en = st.checkbox(
                "Enable File Log", 
                curr_f_en, 
                key='log_f_en',
                help="Save logs to disk file."
            )
            if new_f_en != curr_f_en: CONFIG.update('logging.file_enabled', new_f_en)
            
            curr_f_dir = CONFIG.get('logging.log_dir')
            new_f_dir = st.text_input(
                "Log Directory", 
                curr_f_dir, 
                key='log_f_dir',
                help="Folder where log files are saved."
            )
            if new_f_dir != curr_f_dir: CONFIG.update('logging.log_dir', new_f_dir)

            curr_f_name = CONFIG.get('logging.log_file')
            new_f_name = st.text_input(
                "Log Filename", 
                curr_f_name, 
                key='log_f_name',
                help="Name of the log file (e.g., app.log)"
            )
            if new_f_name != curr_f_name: CONFIG.update('logging.log_file', new_f_name)

        # Console Logging
        with col_console:
            st.subheader("Console Logging")
            curr_c_en = CONFIG.get('logging.console_enabled')
            new_c_en = st.checkbox(
                "Enable Console Log", 
                curr_c_en, 
                key='log_c_en',
                help="Print logs to terminal/console."
            )
            if new_c_en != curr_c_en: CONFIG.update('logging.console_enabled', new_c_en)

            curr_c_lvl = CONFIG.get('logging.console_level')
            new_c_lvl = st.selectbox(
                "Console Level", 
                ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(curr_c_lvl), 
                key='log_c_lvl',
                help="Minimum severity level for console output."
            )
            if new_c_lvl != curr_c_lvl: CONFIG.update('logging.console_level', new_c_lvl)

        # Streamlit Logging
        with col_st:
            st.subheader("Streamlit Logging")
            curr_st_en = CONFIG.get('logging.streamlit_enabled')
            new_st_en = st.checkbox(
                "Enable Streamlit Log", 
                curr_st_en, 
                key='log_st_en',
                help="Log events in Streamlit-specific format."
            )
            if new_st_en != curr_st_en: CONFIG.update('logging.streamlit_enabled', new_st_en)

            curr_st_lvl = CONFIG.get('logging.streamlit_level')
            new_st_lvl = st.selectbox(
                "Streamlit Level", 
                ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(curr_st_lvl), 
                key='log_st_lvl',
                help="Minimum severity level for Streamlit output."
            )
            if new_st_lvl != curr_st_lvl: CONFIG.update('logging.streamlit_level', new_st_lvl)

        st.divider()

        # What to Log (Booleans)
        st.subheader("Event Logging Filters")
        c1, c2, c3 = st.columns(3)
        with c1:
            l_fo = CONFIG.get('logging.log_file_operations')
            n_fo = st.checkbox(
                "File Operations", 
                l_fo, 
                key='l_fo',
                help="Log file reads, writes, uploads."
            )
            if n_fo != l_fo: CONFIG.update('logging.log_file_operations', n_fo)
            
            l_do = CONFIG.get('logging.log_data_operations')
            n_do = st.checkbox(
                "Data Operations", 
                l_do, 
                key='l_do',
                help="Log data loading, filtering, transformations."
            )
            if n_do != l_do: CONFIG.update('logging.log_data_operations', n_do)

        with c2:
            l_ao = CONFIG.get('logging.log_analysis_operations')
            n_ao = st.checkbox(
                "Analysis Operations", 
                l_ao, 
                key='l_ao',
                help="Log statistical analyses, model fitting."
            )
            if n_ao != l_ao: CONFIG.update('logging.log_analysis_operations', n_ao)
            
            l_ui = CONFIG.get('logging.log_ui_events')
            n_ui = st.checkbox(
                "UI Events (Verbose)", 
                l_ui, 
                key='l_ui',
                help="Log clicks, selections, scrolling. Very verbose!"
            )
            if n_ui != l_ui: CONFIG.update('logging.log_ui_events', n_ui)

        with c3:
            l_pf = CONFIG.get('logging.log_performance')
            n_pf = st.checkbox(
                "Performance Timing", 
                l_pf, 
                key='l_pf',
                help="Log execution times and performance metrics."
            )
            if n_pf != l_pf: CONFIG.update('logging.log_performance', n_pf)


    # ==========================================
    # 4. TAB: PERFORMANCE SETTINGS
    # ==========================================
    with tab_perf:
        st.header("Performance Optimization")
        
        c1, c2 = st.columns(2)
        with c1:
            curr_cache = CONFIG.get('performance.enable_caching')
            new_cache = st.toggle(
                "Enable Caching", 
                curr_cache, 
                key='perf_cache',
                help="Cache results to speed up repeated operations."
            )
            if new_cache != curr_cache: CONFIG.update('performance.enable_caching', new_cache)
            
            curr_comp = CONFIG.get('performance.enable_compression')
            new_comp = st.toggle(
                "Enable Compression", 
                curr_comp, 
                key='perf_comp',
                help="Compress data to reduce memory usage."
            )
            if new_comp != curr_comp: CONFIG.update('performance.enable_compression', new_comp)

        with c2:
            curr_ttl = CONFIG.get('performance.cache_ttl')
            new_ttl = st.number_input(
                "Cache TTL (seconds)", 
                60, 86400, int(curr_ttl), 300, 
                key='perf_ttl',
                help="How long cached results remain valid (in seconds). Typical: 3600 (1 hour)"
            )
            if new_ttl != curr_ttl: CONFIG.update('performance.cache_ttl', new_ttl)

            curr_thr = CONFIG.get('performance.num_threads')
            new_thr = st.number_input(
                "Number of Threads", 
                1, 32, int(curr_thr), 
                key='perf_thr',
                help="CPU threads for parallel processing. Use system core count for optimal performance."
            )
            if new_thr != curr_thr: CONFIG.update('performance.num_threads', new_thr)


    # ==========================================
    # 5. TAB: ADVANCED (Validation & Debug)
    # ==========================================
    with tab_adv:
        st.header("Advanced Settings")

        col_val, col_dbg = st.columns(2)
        
        with col_val:
            st.subheader("Validation")
            
            curr_strict = CONFIG.get('validation.strict_mode')
            new_strict = st.toggle(
                "Strict Mode", 
                curr_strict, 
                help="Error instead of Warn on validation failures. Stop execution on errors.",
                key='val_strict'
            )
            if new_strict != curr_strict: CONFIG.update('validation.strict_mode', new_strict)

            curr_v_in = CONFIG.get('validation.validate_inputs')
            new_v_in = st.checkbox(
                "Validate Inputs", 
                curr_v_in, 
                key='val_in',
                help="Check data types and formats before analysis."
            )
            if new_v_in != curr_v_in: CONFIG.update('validation.validate_inputs', new_v_in)

            curr_v_out = CONFIG.get('validation.validate_outputs')
            new_v_out = st.checkbox(
                "Validate Outputs", 
                curr_v_out, 
                key='val_out',
                help="Check results for consistency and correctness."
            )
            if new_v_out != curr_v_out: CONFIG.update('validation.validate_outputs', new_v_out)
            
            curr_fix = CONFIG.get('validation.auto_fix_errors')
            new_fix = st.checkbox(
                "Auto-fix Errors", 
                curr_fix, 
                key='val_fix',
                help="Automatically correct common data issues."
            )
            if new_fix != curr_fix: CONFIG.update('validation.auto_fix_errors', new_fix)

        with col_dbg:
            st.subheader("Debugging")
            
            curr_dbg = CONFIG.get('debug.enabled')
            new_dbg = st.toggle(
                "Enable Debug Mode", 
                curr_dbg, 
                key='dbg_en',
                help="Show detailed debugging information and stack traces."
            )
            if new_dbg != curr_dbg: CONFIG.update('debug.enabled', new_dbg)

            curr_verb = CONFIG.get('debug.verbose')
            new_verb = st.checkbox(
                "Verbose Output", 
                curr_verb, 
                key='dbg_verb',
                help="Print detailed intermediate steps and variables."
            )
            if new_verb != curr_verb: CONFIG.update('debug.verbose', new_verb)

            curr_prof = CONFIG.get('debug.profile_performance')
            new_prof = st.checkbox(
                "Profile Performance", 
                curr_prof, 
                key='dbg_prof',
                help="Measure CPU and memory usage for each function."
            )
            if new_prof != curr_prof: CONFIG.update('debug.profile_performance', new_prof)

            curr_time = CONFIG.get('debug.show_timings')
            new_time = st.checkbox(
                "Show Timings", 
                curr_time, 
                key='dbg_time',
                help="Display execution time for each operation."
            )
            if new_time != curr_time: CONFIG.update('debug.show_timings', new_time)
        
        st.markdown("---")
        with st.expander("📄 Raw Configuration (JSON)"):
            st.json(CONFIG.to_dict())

    # ==========================================
    # 6. TAB: COLORS (Read-only)
    # ==========================================
    with tab_colors:
        st.header("Theme Color Palette (Read-only)")
        st.caption("Reference from `tabs/_common.py`")
        
        colors = get_color_palette()
        
        groups = {
            "Primary Colors": ['primary', 'primary_dark', 'primary_light'],
            "Status Colors": ['success', 'warning', 'danger', 'info'],
            "Neutral / Text": ['text', 'text_secondary', 'neutral', 'border', 'background', 'surface']
        }
        
        for group_name, keys in groups.items():
            st.subheader(group_name)
            cols = st.columns(len(keys))
            for i, key in enumerate(keys):
                color_code = colors.get(key, '#FFFFFF')
                with cols[i]:
                    st.color_picker(f"{key}", value=color_code, disabled=True, key=f"cp_{key}")
                    st.caption(f"`{key}`\n{color_code}")
            st.divider()
