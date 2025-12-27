import streamlit as st
import pandas as pd
from config import CONFIG
from tabs._common import get_color_palette

def render():
    st.title("⚙️ System Configuration")
    st.info("💡 **Note:** Changes made here affect the current runtime session immediately. Some UI changes may require a page refresh.")

    # Create tabs matching the structure of config.py + Colors reference
    tabs = st.tabs([
        "📊 Analysis", 
        "🎨 UI & Display", 
        "📝 Logging", 
        "⚡ Performance", 
        "🛠️ Advanced", 
        "🌈 Colors"
    ])
    
    tab_analysis, tab_ui, tab_logging, tab_perf, tab_adv, tab_colors = tabs

    # ==========================================
    # 1. TAB: ANALYSIS SETTINGS
    # ==========================================
    with tab_analysis:
        st.header("Statistical Analysis Parameters")
        st.caption("Fine-tune how the engine processes your medical data.")

        # --- Logistic Regression ---
        with st.expander("🔹 Logistic Regression (Risk Factor Analysis)", expanded=True):
            st.markdown("""
            **Context:** Used to find the relationship between predictors and a binary outcome (e.g., Dead vs Alive).
            """)
            c1, c2 = st.columns(2)
            with c1:
                curr_method = CONFIG.get('analysis.logit_method')
                new_method = st.selectbox(
                    "Calculation Method", 
                    ['auto', 'firth', 'bfgs', 'default'], 
                    index=['auto', 'firth', 'bfgs', 'default'].index(curr_method),
                    help="""
                    - **auto**: Let the system decide.
                    - **firth**: Best for small samples or rare events (prevents bias).
                    - **bfgs**: Standard fast calculation for large datasets.
                    """,
                    key='an_logit_met'
                )
                if new_method != curr_method: CONFIG.update('analysis.logit_method', new_method)

                curr_p = CONFIG.get('analysis.logit_screening_p')
                new_p = st.number_input(
                    "Univariate Screening P-value", 0.0, 1.0, float(curr_p), 0.01, format="%.2f",
                    help="Variables with a p-value lower than this in simple analysis will be automatically selected for the complex model. NEJM standard often uses 0.10 to 0.20.",
                    key='an_scr_p'
                )
                if new_p != curr_p: CONFIG.update('analysis.logit_screening_p', new_p)

            with c2:
                curr_iter = CONFIG.get('analysis.logit_max_iter')
                new_iter = st.number_input(
                    "Max Iterations", 10, 5000, int(curr_iter), 10,
                    help="How many times the computer tries to 'fit' the model. Increase if the model fails to converge.",
                    key='an_max_it'
                )
                if new_iter != curr_iter: CONFIG.update('analysis.logit_max_iter', new_iter)

                curr_min = CONFIG.get('analysis.logit_min_cases')
                new_min = st.number_input(
                    "Min Cases for Multivariate", 1, 100, int(curr_min),
                    help="Minimum number of participants needed to run complex analysis. Prevents unreliable results.",
                    key='an_min_cs'
                )
                if new_min != curr_min: CONFIG.update('analysis.logit_min_cases', new_min)

        # --- Survival Analysis ---
        with st.expander("🔹 Survival Analysis (Time-to-Event)"):
            st.markdown("**Context:** Used when analyzing the time until an event occurs (e.g., time to discharge).")
            c1, c2 = st.columns(2)
            with c1:
                curr_surv = CONFIG.get('analysis.survival_method')
                new_surv = st.selectbox(
                    "Survival Method", ['kaplan-meier', 'weibull'], 
                    index=['kaplan-meier', 'weibull'].index(curr_surv),
                    help="Kaplan-Meier is the standard 'gold rule' for clinical trials.",
                    key='an_surv_m'
                )
                if new_surv != curr_surv: CONFIG.update('analysis.survival_method', new_surv)
            with c2:
                curr_cox = CONFIG.get('analysis.cox_method')
                new_cox = st.selectbox(
                    "Cox Regression Tie-Handling", ['efron', 'breslow'], 
                    index=['efron', 'breslow'].index(curr_cox),
                    help="How to handle patients who have events at the exact same time. 'Efron' is generally more accurate.",
                    key='an_cox_m'
                )
                if new_cox != curr_cox: CONFIG.update('analysis.cox_method', new_cox)

        # --- Data Detection & Missing ---
        with st.expander("🔹 Data Handling & Missing Values"):
            c1, c2 = st.columns(2)
            with c1:
                curr_vth = CONFIG.get('analysis.var_detect_threshold')
                new_vth = st.number_input(
                    "Categorical Threshold", 1, 50, int(curr_vth),
                    help="If a column has fewer unique values than this, the system treats it as a 'Group' (Categorical) rather than a 'Measurement' (Continuous).",
                    key='an_var_th'
                )
                if new_vth != curr_vth: CONFIG.update('analysis.var_detect_threshold', new_vth)

            with c2:
                curr_mstrat = CONFIG.get('analysis.missing_strategy')
                new_mstrat = st.selectbox(
                    "Missing Data Handling", ['complete-case', 'drop'], 
                    index=['complete-case', 'drop'].index(curr_mstrat),
                    help="NEJM standard: 'Complete-case' only analyzes patients with full information available.",
                    key='an_mis_str'
                )
                if new_mstrat != curr_mstrat: CONFIG.update('analysis.missing_strategy', new_mstrat)

        # --- P-value Handling (NEJM Standards) ---
        with st.expander("🔹 P-value Formatting (NEJM Guideline)", expanded=True):
            st.markdown("""
            **NEJM Standard:** P-values >0.01 should be reported to 2 or 3 decimal places. P-values <0.001 should be reported as P<0.001.
            """)
            c1, c2 = st.columns(2)
            with c1:
                curr_fs = CONFIG.get('analysis.pvalue_format_small')
                new_fs = st.text_input(
                    "Very Small P-value format", curr_fs, 
                    help="Standard: '<0.001'. Used when the result is highly significant.",
                    key='an_pv_fs'
                )
                if new_fs != curr_fs: CONFIG.update('analysis.pvalue_format_small', new_fs)
            
            with c2:
                curr_fl = CONFIG.get('analysis.pvalue_format_large')
                new_fl = st.text_input(
                    "Very Large P-value format", curr_fl, 
                    help="Standard: '>0.99'. Used when there is almost zero difference.",
                    key='an_pv_fl'
                )
                if new_fl != curr_fl: CONFIG.update('analysis.pvalue_format_large', new_fl)


    # ==========================================
    # 2. TAB: UI & DISPLAY SETTINGS
    # ==========================================
    with tab_ui:
        st.header("UI & Display Settings")

        # --- NEJM Reporting Standards ---
        with st.expander("🔹 Reporting Standards (Decimals)", expanded=True):
            st.markdown("""
            **NEJM Guideline:** Report means and standard deviations to one more decimal place than the raw data. 
            For ratios (OR/HR/RR), **2 decimal places** are standard.
            """)
            c1, c2 = st.columns(2)
            with c1:
                curr_tdec = CONFIG.get('ui.table_decimal_places')
                # Adjusted default for NEJM (usually 2 for metrics, 3 for p-values)
                new_tdec = st.number_input(
                    "Standard Decimal Places", 0, 10, 2, 
                    help="Used for Mean, SD, Odds Ratios, and Percentages in tables.",
                    key='ui_tdec'
                )
                if new_tdec != curr_tdec: CONFIG.update('ui.table_decimal_places', new_tdec)

        # --- Sidebar & Layout ---
        with st.expander("🔹 Appearance"):
            c1, c2 = st.columns(2)
            with c1:
                curr_title = CONFIG.get('ui.page_title')
                new_title = st.text_input("Application Title", curr_title, key='ui_title')
                if new_title != curr_title: CONFIG.update('ui.page_title', new_title)
            with c2:
                curr_logo = CONFIG.get('ui.show_sidebar_logo')
                new_logo = st.toggle("Show Sidebar Logo", curr_logo, key='ui_logo')
                if new_logo != curr_logo: CONFIG.update('ui.show_sidebar_logo', new_logo)

        # --- Tables & Plots ---
        with st.expander("🔹 Tables & Visuals"):
            c1, c2 = st.columns(2)
            with c1:
                curr_tr = CONFIG.get('ui.table_max_rows')
                new_tr = st.number_input("Max Rows to Display", 10, 10000, int(curr_tr), key='ui_tr')
                if new_tr != curr_tr: CONFIG.update('ui.table_max_rows', new_tr)
            with c2:
                curr_pdpi = CONFIG.get('ui.plot_dpi')
                new_pdpi = st.number_input(
                    "Plot Resolution (DPI)", 72, 600, int(curr_pdpi), 
                    help="Higher DPI (e.g., 300) is better for print/publication but slower to render.",
                    key='ui_pdpi'
                )
                if new_pdpi != curr_pdpi: CONFIG.update('ui.plot_dpi', new_pdpi)


    # ==========================================
    # 3. TAB: LOGGING SETTINGS
    # ==========================================
    with tab_logging:
        st.header("Technical Logging")
        st.caption("Track errors and background operations for debugging.")

        c1, c2 = st.columns(2)
        with c1:
            curr_log_en = CONFIG.get('logging.enabled')
            new_log_en = st.toggle("Enable Logging", curr_log_en, help="Turn off if you want maximum performance and don't need to track errors.", key='log_en')
            if new_log_en != curr_log_en: CONFIG.update('logging.enabled', new_log_en)
        with c2:
            curr_log_lvl = CONFIG.get('logging.level')
            new_log_lvl = st.selectbox(
                "Detail Level", ['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(curr_log_lvl),
                help="'DEBUG' shows everything (slow), 'ERROR' shows only critical failures.",
                key='log_lvl'
            )
            if new_log_lvl != curr_log_lvl: CONFIG.update('logging.level', new_log_lvl)


    # ==========================================
    # 4. TAB: PERFORMANCE SETTINGS
    # ==========================================
    with tab_perf:
        st.header("Performance & Caching")
        
        c1, c2 = st.columns(2)
        with c1:
            curr_cache = CONFIG.get('performance.enable_caching')
            new_cache = st.toggle(
                "Enable Smart Caching", curr_cache, 
                help="Stores results of heavy calculations so you don't have to wait when switching tabs.",
                key='perf_cache'
            )
            if new_cache != curr_cache: CONFIG.update('performance.enable_caching', new_cache)
        with c2:
            curr_ttl = CONFIG.get('performance.cache_ttl')
            new_ttl = st.number_input("Cache Memory Duration (sec)", 60, 86400, int(curr_ttl), key='perf_ttl')
            if new_ttl != curr_ttl: CONFIG.update('performance.cache_ttl', new_ttl)


    # ==========================================
    # 5. TAB: ADVANCED
    # ==========================================
    with tab_adv:
        st.header("System & Debugging")

        col_val, col_dbg = st.columns(2)
        with col_val:
            st.subheader("Data Validation")
            curr_strict = CONFIG.get('validation.strict_mode')
            new_strict = st.toggle("Strict Mode", curr_strict, help="If on, the app will stop and show an error for any minor data inconsistency.", key='val_strict')
            if new_strict != curr_strict: CONFIG.update('validation.strict_mode', new_strict)
            
            curr_fix = CONFIG.get('validation.auto_fix_errors')
            new_fix = st.checkbox("Auto-fix Data Issues", curr_fix, help="Automatically tries to fix common data errors (e.g., removing trailing spaces).", key='val_fix')
            if new_fix != curr_fix: CONFIG.update('validation.auto_fix_errors', new_fix)

        with col_dbg:
            st.subheader("Developer Tools")
            curr_dbg = CONFIG.get('debug.enabled')
            new_dbg = st.toggle("Enable Debug Mode", curr_dbg, key='dbg_en')
            if new_dbg != curr_dbg: CONFIG.update('debug.enabled', new_dbg)

        st.divider()
        with st.expander("📄 Export Raw Config (JSON)"):
            st.json(CONFIG.to_dict())


    # ==========================================
    # 6. TAB: COLORS
    # ==========================================
    with tab_colors:
        st.header("Clinical Palette Reference")
        st.caption("These colors are used throughout the app for professional medical reporting.")
        
        colors = get_color_palette()
        groups = {
            "Brand": ['primary', 'primary_dark', 'primary_light'],
            "Signaling": ['success', 'warning', 'danger', 'info'],
            "Typography": ['text', 'text_secondary', 'neutral', 'border']
        }
        
        for group_name, keys in groups.items():
            st.subheader(group_name)
            cols = st.columns(len(keys))
            for i, key in enumerate(keys):
                color_code = colors.get(key, '#FFFFFF')
                with cols[i]:
                    st.color_picker(f"{key}", value=color_code, disabled=True, key=f"cp_{key}")
                    st.caption(f"**{key}**\n`{color_code}`")
            st.divider()
