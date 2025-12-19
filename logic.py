import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
import html
import streamlit as st  # ✅ IMPORT STREAMLIT

# ✅ FIX #7-8: IMPORT LOGGER (MINIMAL WIRING)
from logger import get_logger
from tabs._common import get_color_palette

# Get logger instance for this module
logger = get_logger(__name__)
# Get unified color palette
COLORS = get_color_palette()
# ✅ TRY IMPORT FIRTHLOGIST
try:
    from firthlogist import FirthLogisticRegression
    HAS_FIRTH = True
except ImportError:
    HAS_FIRTH = False

warnings.filterwarnings("ignore")

def clean_numeric_value(val):
    """
    Normalize a value into a numeric float suitable for analysis.
    
    Cleans common non-numeric markers (such as leading/trailing whitespace, '>', '<', and thousands separators like ',') and converts the result to a float. If the input is missing or cannot be parsed as a number, returns NaN.
    
    Parameters:
        val: The input value to normalize (may be a string, number, or missing).
    
    Returns:
        numeric_value (float): The parsed float, or `np.nan` when the value is missing or unparseable.
    """
    if pd.isna(val): 
        return np.nan
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan

def run_binary_logit(y, X, method='default'):
    """
    Perform binary logistic regression using the specified estimation method.
    
    Supports 'default' (statsmodels Logit default optimizer), 'bfgs' (statsmodels Logit with BFGS), and 'firth' (Firth's penalized likelihood when the firthlogist package is available). An intercept column is added to X automatically; if 'firth' is requested but firthlogist is not installed the function returns an error status.
    
    Parameters:
        y (array-like or pd.Series): Binary outcome aligned to the rows of X.
        X (array-like or pd.DataFrame): Predictor matrix; a constant/intercept column will be added.
        method (str): One of 'default', 'bfgs', or 'firth' selecting the estimator.
    
    Returns:
        tuple: (params, conf_int, pvalues, status)
            - params (pd.Series or None): Estimated coefficients indexed by predictor names (including the intercept), or None on failure.
            - conf_int (pd.DataFrame or None): Confidence intervals with columns [0, 1] indexed by predictor names, or None on failure.
            - pvalues (pd.Series or None): Two-sided p-values indexed by predictor names, or None on failure.
            - status (str): "OK" on success; otherwise an error message describing the failure.
    """
    try:
        # เตรียมข้อมูล (Statsmodels ต้องการ Constant เสมอ)
        X_const = sm.add_constant(X, has_constant='add')
        
        # 🟢 CASE 1: FIRTH'S LOGISTIC REGRESSION (Recommended)
        if method == 'firth':
            if not HAS_FIRTH:
                return None, None, None, "Library 'firthlogist' not installed. Please define requirements.txt or use Standard method."
            
            # firthlogist: fit_intercept=False เพราะทำเราใส่ Constant ใน X ไปแล้ว
            fl = FirthLogisticRegression(fit_intercept=False) 
            fl.fit(X_const, y)
            
            # แปลงผลลัพท์ให้ตรงกับ Format เดิม (Series/DataFrame)
            coef = np.asarray(fl.coef_).reshape(-1)
            if coef.shape[0] != len(X_const.columns):
                return None, None, None, "Firth output shape mismatch (coef_ vs design matrix)."
            params = pd.Series(coef, index=X_const.columns)
            pvalues = pd.Series(getattr(fl, "pvals_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
            ci = getattr(fl, "ci_", None)
            conf_int = (
                pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])
                if ci is not None
                else pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
            )
            
            return params, conf_int, pvalues, "OK"

        # 🔵 CASE 2: STANDARD LOGISTIC (Statsmodels)
        elif method == 'bfgs':
            model = sm.Logit(y, X_const).fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const).fit(disp=0)
            
        return model.params, model.conf_int(), model.pvalues, "OK"
        
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.exception("Logistic regression failed")  # ✅ LOG ERROR
        return None, None, None, str(e)

def get_label(col_name, var_meta):
    """
    Build an HTML label for a column using its name and optional metadata.
    
    When metadata provides a 'label' for the full column name (or for the suffix after the first underscore), the label is shown as a secondary, muted line beneath the bolded column name; otherwise only the bolded column name is returned. Both display name and secondary label are HTML-escaped.
    
    Parameters:
        col_name (str): The column name to display.
        var_meta (dict or None): Optional mapping of column keys to metadata dictionaries; a metadata entry with key 'label' supplies the secondary label.
    
    Returns:
        str: An HTML fragment containing a bolded column name and, if available, a secondary label on the next line.
    """
    # 1. ใช้ชื่อเต็มหาค่าเริ่มต้น (ไม่ตัด _ ทิ้งแล้ว)
    display_name = col_name 
    
    # 2. ค้นหาคำอธิบาย (Label) ใน Metadata
    secondary_label = ""
    if var_meta:
        # ลองหาด้วยกับชื่อเต็มก่อน
        if col_name in var_meta and 'label' in var_meta[col_name]:
            secondary_label = var_meta[col_name]['label']
        # (Optional) ลองหาด้วยกับชื่อย่อ เผื่อ config เก่าตั้งไว้
        elif '_' in col_name:
            parts = col_name.split('_', 1)
            if len(parts) > 1:
                short_name = parts[1]
                if short_name in var_meta and 'label' in var_meta[short_name]:
                    secondary_label = var_meta[short_name]['label']

    safe_name = html.escape(str(display_name))
    
    # 3. ถ้ามี Label ให้แสดงบรรทัดล่าง, ถ้าไม่มีให้แสดงแค่ชื่อตัวแปรคสูง
    if secondary_label:
        safe_label = html.escape(str(secondary_label))
        return f"<b>{safe_name}</b><br><span style='color:#666; font-size:0.9em'>{safe_label}</span>"
    else:
        return f"<b>{safe_name}</b>"

# ✅ CACHE DATA: ช่วยให้เว็บเร็ว ไม่ต้องคำนวณใหม่ทุกครั้งที่กดปุ่มอื่น
# 🟢 NOTE: ต้องเพิ่ม method ลงใน argument เพื่อให้ cache แยกงานตาม method ที่เลือก
@st.cache_data(show_spinner=False)
def analyze_outcome(outcome_name, df, var_meta=None, method='auto'):
    """
    Analyze a binary outcome against all other columns in a dataframe and produce an HTML report summarizing univariate and multivariate results.
    
    Per-column descriptive statistics and univariate comparisons are computed (Chi-square for categorical, Mann-Whitney U for continuous), univariable logistic regression provides crude odds ratios, and a multivariable logistic model is fitted on screened candidate predictors to produce adjusted odds ratios when feasible.
    
    Parameters:
        outcome_name (str): Column name of the binary outcome in `df`.
        df (pandas.DataFrame): Input dataset containing `outcome_name` and candidate predictors.
        var_meta (dict, optional): Variable metadata...
        method (str, optional): Regression method to use; one of 'auto', 'firth', 'bfgs', or 'default'. 'auto' selects Firth's penalized likelihood when available, otherwise BFGS-based logistic regression. 'default' uses statsmodels' standard optimizer.
    
    Returns:
        str: An HTML fragment containing a table of variables with descriptive statistics, crude odds ratios (and p-values), and adjusted odds ratios where multivariable modelling was performed. If `outcome_name` is not found in `df`, returns an HTML alert div indicating the missing outcome.
    """
    
    # ✅ LOG ANALYSIS START
    logger.log_analysis(
        analysis_type="Logistic Regression",
        outcome=outcome_name,
        n_vars=len(df.columns) - 1,
        n_samples=len(df)
    )
    
    # ✅ FIX #3: ADD BINARY OUTCOME VALIDATION
    if outcome_name not in df.columns:
        msg = f"<div class='alert'>⚠️ Outcome '{outcome_name}' not found.</div>"
        logger.warning("Outcome column not found: %s", outcome_name)  # ✅ LOG WARNING
        return msg
    
    # NEW: Validate outcome is binary (exactly 2 unique values)
    y_raw = df[outcome_name].dropna()
    unique_outcomes = set(y_raw.unique())
    
    if len(unique_outcomes) != 2:
        msg = f"""
        <div class='alert' style='background:#ffebee; border-left:4px solid {COLORS['danger']}; padding:12px; border-radius:4px;'>
            ❌ <b>Invalid Outcome:</b> Expected binary outcome (2 unique values) but found <b>{len(unique_outcomes)}</b>.<br>
            Unique values: {sorted(unique_outcomes)}<br>
            <span style='font-size:0.9em; color:#666; margin-top:8px; display:block;'>💡 Please select a truly binary outcome variable (e.g., Yes/No, Dead/Alive, 0/1)</span>
        </div>
        """
        logger.error("Invalid outcome: %d unique values instead of 2", len(unique_outcomes))  # ✅ LOG ERROR
        return msg
    
    # NEW: Warn if outcome isn't 0/1
    if not unique_outcomes.issubset({0, 1}):
        st.warning(f"i Outcome values are {sorted(unique_outcomes)}, not {{0, 1}}. Will be converted to binary.")
        # Map to binary: first sorted value -> 0, second -> 1
        sorted_outcomes = sorted(unique_outcomes, key=str)
        outcome_map = {sorted_outcomes[0]: 0, sorted_outcomes[1]: 1}
        y = y_raw.map(outcome_map).astype(int)
    else:
        y = y_raw.astype(int)
    
    df_aligned = df.loc[y.index]
    total_n = len(y)
    
    candidates = [] 
    results_db = {} 
    sorted_cols = sorted(df.columns)

    # 🇒️ NEW: DETECT DATA QUALITY FOR AUTO-METHOD SELECTION
    has_perfect_separation = False
    small_sample = len(df) < 50
    rare_outcome = (y == 1).sum() < 20
    
    # Check for perfect separation in any predictor
    if method == 'auto':
        for col in sorted_cols:
            if col == outcome_name:
                continue
            try:
                X_num = df_aligned[col].apply(clean_numeric_value)
                if X_num.nunique() > 1:
                    tab = pd.crosstab(X_num, y)
                    if (tab == 0).any().any():  # Zero cell = separation
                        has_perfect_separation = True
                        logger.warning("🔴 Perfect separation detected in: %s", col)
                        break
            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Skipping separation check for %s: %s", col, e)
                continue
    
    # 🇒️ NEW: AUTO-SELECT METHOD BASED ON DATA QUALITY
    preferred_method = 'bfgs'  # Default fallback
    
    if method == 'auto':
        if HAS_FIRTH and (has_perfect_separation or small_sample or rare_outcome):
            preferred_method = 'firth'
            conditions = []
            if has_perfect_separation:
                conditions.append("perfect_separation")
            if small_sample:
                conditions.append(f"small_sample(n={len(df)}<50)")
            if rare_outcome:
                conditions.append(f"rare_outcome({(y==1).sum()}<20)")
            logger.info("✅ Auto-selected Firth's method | Conditions: %s", ', '.join(conditions))
        else:
            preferred_method = 'bfgs'
            logger.info("✅ Auto-selected Standard method (BFGS) | Data quality OK")
    elif method == 'firth':
        preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
    elif method == 'bfgs':
        preferred_method = 'bfgs'
    elif method == 'default':  
        preferred_method = 'default'

    # --- CALCULATION LOOP ---
    with logger.track_time("univariate_analysis", log_level="debug"):  # ✅ TRACK TIMING
        for col in sorted_cols:
            if col == outcome_name:
                continue
            if df_aligned[col].isnull().all():
                continue

            res = {'var': col}
            X_raw = df_aligned[col]
            X_num = X_raw.apply(clean_numeric_value)
            
            X_neg = X_raw[y == 0]
            X_pos = X_raw[y == 1]
            
            orig_name = col.split('_', 1)[1] if len(col.split('_', 1)) > 1 else col
            
            unique_vals = X_num.dropna().unique()
            unique_count = len(unique_vals)
            
            # ✅ FIX #2: IMPROVE CATEGORICAL/CONTINUOUS DETECTION
            is_categorical = False
            is_binary = set(unique_vals).issubset({0, 1})
            
            # NEW: Better detection logic
            if is_binary:
                is_categorical = True
            elif unique_count < 10:  # 🟢 INCREASED THRESHOLD from 5 to 10
                # Check if mostly integers (likely categorical codes)
                decimals_count = sum(1 for v in unique_vals if not float(v).is_integer())
                decimals_pct = decimals_count / len(unique_vals) if unique_vals.size > 0 else 0
                
                if decimals_pct < 0.3:  # If <30% have decimals, treat as categorical
                    is_categorical = True
                # else: treat as continuous
            
            # Allow user override via metadata
            user_setting = {}
            if var_meta and (col in var_meta or orig_name in var_meta):
                key = col if col in var_meta else orig_name
                user_setting = var_meta[key]
                
                if user_setting.get('type') == 'Categorical':
                    is_categorical = True
                elif user_setting.get('type') == 'Continuous':
                    is_categorical = False
            
            if is_categorical:
                n_used = len(X_raw.dropna())
                mapper = user_setting.get('map', {})
    
                try:
                    levels = sorted(X_raw.dropna().unique(), key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))
                except (ValueError, TypeError):
                    levels = sorted(X_raw.astype(str).unique())
    
                desc_tot = [f"<span class='n-badge'>n={n_used}</span>"]
                desc_neg = [f"<span class='n-badge'>n={len(X_neg.dropna())}</span>"]
                desc_pos = [f"<span class='n-badge'>n={len(X_pos.dropna())}</span>"]
    
                # ✅ DEFINE count_val ONCE, BEFORE THE LOOP
                def count_val(series, v_str) -> int:
                    """
                    Count occurrences in a Series matching a target string after normalizing numeric-like values (e.g., converting "1.0" to "1").
                    
                    Parameters:
                        series (pandas.Series): Series whose elements will be compared as strings after normalization.
                        v_str (str): Target string to match against each normalized element.
                    
                    Returns:
                        int: Number of elements equal to `v_str` after normalization.
                    """
                    return (series.astype(str).apply(lambda x: x.replace('.0','') if x.replace('.','',1).isdigit() else x) == v_str).sum()
    
                for lvl in levels:  # ← Loop starts here
                    try:
                        if float(lvl).is_integer():
                            key = int(float(lvl))
                        else:
                            key = float(lvl)
                    except (ValueError, TypeError):
                        key = lvl
        
                    label_txt = mapper.get(key, str(lvl))
                    lvl_str = str(lvl)
                    if str(lvl).endswith('.0'): 
                        lvl_str = str(int(float(lvl)))
        
                    # ✅ Just call it - no definition here anymore!
                    c_all = count_val(X_raw, lvl_str)
                       
                    if c_all == 0:
                        c_all = (X_raw == lvl).sum()
                    
                    p_all = (c_all/n_used)*100 if n_used else 0
                    c_n = count_val(X_neg, lvl_str)
                    p_n = (c_n/len(X_neg.dropna()))*100 if len(X_neg.dropna()) else 0
                    c_p = count_val(X_pos, lvl_str)
                    p_p = (c_p/len(X_pos.dropna()))*100 if len(X_pos.dropna()) else 0
                    
                    desc_tot.append(f"{label_txt}: {c_all} ({p_all:.1f}%)")
                    desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                    desc_pos.append(f"{c_p} ({p_p:.1f}%)")
                
                res['desc_total'] = "<br>".join(desc_tot)
                res['desc_neg'] = "<br>".join(desc_neg)
                res['desc_pos'] = "<br>".join(desc_pos)
                
                try:
                    contingency = pd.crosstab(X_raw, y)
                    if contingency.size > 0:
                        _, p, _, _ = stats.chi2_contingency(contingency)
                        res['p_comp'] = p
                        res['test_name'] = "Chi-square"
                    else: 
                        res['p_comp'] = np.nan
                        res['test_name'] = "-"
                except (ValueError, np.linalg.LinAlgError):
                    res['p_comp'] = np.nan
                    res['test_name'] = "-"
                
            else:
                n_used = len(X_num.dropna())
                m_t, s_t = X_num.mean(), X_num.std()
                m_n, s_n = pd.to_numeric(X_neg, errors='coerce').mean(), pd.to_numeric(X_neg, errors='coerce').std()
                m_p, s_p = pd.to_numeric(X_pos, errors='coerce').mean(), pd.to_numeric(X_pos, errors='coerce').std()
                
                res['desc_total'] = f"<span class='n-badge'>n={n_used}</span><br>Mean: {m_t:.2f}<br>(SD {s_t:.2f})"
                res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
                res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
                
                try:
                    _, p = stats.mannwhitneyu(pd.to_numeric(X_neg, errors='coerce').dropna(), pd.to_numeric(X_pos, errors='coerce').dropna())
                    res['p_comp'] = p
                    res['test_name'] = "Mann-Whitney U"
                except (ValueError, TypeError): 
                    res['p_comp'] = np.nan
                    res['test_name'] = "-"

            # --- UNIVARIATE REGRESSION ---
            data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
            if not data_uni.empty and data_uni['x'].nunique() > 1:
                params, conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                if status == "OK" and 'x' in params:
                    coef = params['x']
                    or_val = np.exp(coef)
                    
                    if 'x' in conf.index:
                        ci_low, ci_high = np.exp(conf.loc['x'][0]), np.exp(conf.loc['x'][1])
                    else:
                        ci_low, ci_high = np.nan, np.nan 
                        
                    res['or'] = f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"
                    res['p_or'] = pvals['x']
                else: 
                    res['or'] = "-"
            else: 
                res['or'] = "-"

            results_db[col] = res
            
            p_screen = res.get('p_comp', np.nan)
            if pd.isna(p_screen): 
                p_screen = res.get('p_or', np.nan)
            if pd.notna(p_screen) and p_screen < 0.20:
                candidates.append(col)

    # --- MULTIVARIATE ANALYSIS ---
    with logger.track_time("multivariate_analysis", log_level="debug"):  # ✅ TRACK TIMING
        aor_results = {}
        cand_valid = [c for c in candidates if df_aligned[c].apply(clean_numeric_value).notna().sum() > 5]
        final_n_multi = 0

        if len(cand_valid) > 0:
            multi_df = pd.DataFrame({'y': y})
            for c in cand_valid:
                multi_df[c] = df_aligned[c].apply(clean_numeric_value)
            multi_data = multi_df.dropna()
            final_n_multi = len(multi_data)
            
            if not multi_data.empty and final_n_multi > 10:
                params, conf, pvals, status = run_binary_logit(multi_data['y'], multi_data[cand_valid], method=preferred_method)
                
                if status == "OK":
                    for var in cand_valid:
                        if var in params:
                            coef = params[var]
                            aor = np.exp(coef)
                            ci_low, ci_high = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                            ap = pvals[var]
                            aor_results[var] = {'aor': f"{aor:.2f} ({ci_low:.2f}-{ci_high:.2f})", 'ap': ap}

    # --- HTML BUILD ---
    html_rows = []
    current_sheet = ""
    
    # 🟢 1. เตรียมรายชื่อคอลัมน์ที่มีผลลัชภ์
    valid_cols_for_html = [c for c in sorted_cols if c in results_db]

    # 🟢 2. ฟังก์ชนเรียงลำดับ (Group -> Name)
    def sort_key_for_grouping(col_name) -> tuple[str, str]:
        group = col_name.split('_')[0] if '_' in col_name else "Variables"
        return (group, col_name)

    # 🟢 3. เรียงลำดับ
    grouped_cols = sorted(valid_cols_for_html, key=sort_key_for_grouping)

    # ✅ FIX #5: P-VALUE BOUNDS CHECKING
    def fmt_p(val) -> str:
        """
        Format a p-value for display, clipping small numerical errors to the valid range [0, 1] and applying display thresholds.
        
        Returns:
            str: Formatted p-value:
                - "-" if the input is missing (NaN).
                - "<0.001" if the value is less than 0.001.
                - ">0.999" if the value is greater than 0.999.
                - Otherwise the p-value rounded to three decimal places (e.g., "0.123").
        """
        if pd.isna(val): 
            return "-"
            
        # Bounds check: p-values must be in [0, 1]
        if val < -0.0001 or val > 1.0001:
            # Numerical error detected - log warning
            logger.warning("⚠️ P-value out of bounds detected: %.6f. Clipping to valid range [0, 1].", val)  # ✅ LOG WARNING
            val = max(0, min(1, val))  # Clip to [0, 1]
        else:
            # Safe clipping within tolerance
            val = max(0, min(1, val))
            
        # Format the p-value
        if val < 0.001:
            return "<0.001"
        if val > 0.999:
            return ">0.999"
            
        return f"{val:.3f}"
            
    for col in grouped_cols:
        if col == outcome_name:
            continue
        res = results_db[col]
        
        sheet = col.split('_')[0] if '_' in col else "Variables"
        if sheet != current_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='9'>{sheet}</td></tr>")
            current_sheet = sheet
            
        lbl = get_label(col, var_meta)
        or_s = res.get('or', '-')

        p_val = res.get('p_comp', np.nan)
        p_s = fmt_p(p_val)
        if pd.notna(p_val) and p_val < 0.05: p_s = f"<span class='sig-p'>{p_s}*</span>"
        
        aor_s, ap_s = "-", "-"
        if col in aor_results:
            ar = aor_results[col]
            aor_s = ar['aor']
            ap_val = ar['ap']
            ap_s = fmt_p(ap_val)
            if pd.notna(ap_val) and ap_val < 0.05: ap_s = f"<span class='sig-p'>{ap_s}*</span>"
            
        row_html = f"""
        <tr>
            <td>{lbl}</td>
            <td>{res.get('desc_total','')}</td>
            <td>{res.get('desc_neg','')}</td>
            <td>{res.get('desc_pos','')}</td>
            <td>{or_s}</td>
            <td>{res.get('test_name', '-')}</td> <td>{p_s}</td>
            <td>{aor_s}</td>
            <td>{ap_s}</td>
        </tr>"""
        html_rows.append(row_html)
    
    # Update Footer Note
    if preferred_method == 'firth':
        if method == 'auto':
            method_note = "Firth's Penalized Likelihood (Auto-detected - data quality concern)"
        else:
            method_note = "Firth's Penalized Likelihood (User Selected)"
    elif preferred_method == 'bfgs':
        if method == 'auto':
            method_note = "Standard Binary Logistic Regression (Auto-selected - data quality OK)"
        else:
            method_note = "Standard Binary Logistic Regression (MLE)"
    elif preferred_method == 'default':
        method_note = "Standard Binary Logistic Regression (Default Optimizer)"
    else:
        method_note = "Binary Logistic Regression"

    # 🟢 4. ส่วน Return HTML (เพิ่มสัญลักษณ์ †)
    logger.info("✅ Logistic regression analysis completed (n_multi=%d, method=%s)", final_n_multi, preferred_method)  # ✅ LOG COMPLETION
    
    html_table = f"""
    <div id='{outcome_name}' class='table-container'>
    <div class='outcome-title'>Outcome: {outcome_name} (Total n={total_n})</div>
    <table>
        <thead>
            <tr>
                <th>Variable</th>
                <th>Total</th>
                <th>Group 0</th>
                <th>Group 1</th>
                <th>Crude OR (95% CI)</th>
                <th>Test Used</th> <th>Crude P-value</th>
                <th>aOR (95% CI) <sup style='color:{COLORS['danger']}; font-weight:bold;'>†</sup><br><span style='font-size:0.8em; font-weight:normal'>(n={final_n_multi})</span></th>
                <th>aP-value</th>
            </tr>
        </thead>
        <tbody>{chr(10).join(html_rows)}</tbody>
    </table>
    <div class='summary-box'>
        <b>Method:</b> {method_note}. Complete Case Analysis.<br>
        <i>Univariate comparison uses Chi-square test (Categorical) or Mann-Whitney U test (Continuous).</i>
        <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 0.9em; color: #666;'>
            <sup style='color:{COLORS['danger']}; font-weight:bold;'>†</sup> <b>Note on aOR:</b> Adjusted Odds Ratios are calculated only for variables with a <b>Crude P-value < 0.20</b> 
            (Screening criteria) and sufficient data quality to prevent overfitting.
        </div>
    </div>
    </div><br>
    """
    
    return html_table

def process_data_and_generate_html(df, target_outcome, var_meta=None, method='auto'):
    """
    Primary entry point to run univariate/multivariate logistic regression analysis 
    on a DataFrame and generate a complete HTML report.

    Parameters:
        df (pandas.DataFrame): The input data.
        target_outcome (str): The column name of the binary outcome variable.
        var_meta (dict, optional): Metadata mapping for variable types and labels. Defaults to None.
        method (str, optional): The logistic regression estimation method ('auto', 'firth', 'bfgs', 'default').

    Returns:
        str: The complete HTML report string.
    """ # 🟢 MODIFIED: Restored Docstring
    
    css_style = f"""
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; }}
        .table-container {{ background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0; text-align: left; min-width: 800px; }}
        th {{ background-color: {COLORS['primary_dark']}; color: #fff; padding: 12px; position: sticky; top: 0; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .outcome-title {{ background-color: {COLORS['primary_dark']}; color: white; padding: 15px; font-weight: bold; border-radius: 8px 8px 0 0; }}
        .sig-p {{ color: {COLORS['danger']}; font-weight: bold; background-color: #ffebee; padding: 2px 4px; border-radius: 4px; }}
        .sheet-header td {{ background-color: #e8f4f8; color: {COLORS['primary']}; font-weight: bold; letter-spacing: 1px; padding: 8px 15px; }}
        .n-badge {{ font-size: 0.75em; color: #888; background: #eee; padding: 1px 4px; border-radius: 3px; }}
        .summary-box {{ padding: 15px; background: #fff; font-size: 0.9em; color: #555; }}
        .report-footer {{
            text-align: right;
            font-size: 0.75em;
            color: {COLORS['text_secondary']};
            margin-top: 20px;
            border-top: 1px dashed {COLORS['border']};
            padding-top: 10px;
        }}
        a {{ color: {COLORS['primary']}; text-decoration: none; }}
        a:hover {{ color: {COLORS['primary_dark']}; }}
    </style>
    """
    
    html = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html += "<h1>Analysis Report</h1>"
    html += analyze_outcome(target_outcome, df, var_meta, method=method)
    
    # ✅ FIX: Use string concatenation instead of f-string with backslash
    footer_html = (
        "<div class='report-footer'>"
        "&copy; 2025 <a href=\"https://github.com/NTWKKM/\" target=\"_blank\" style=\"text-decoration:none; color:inherit;\">"
        "NTWKKM n donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit"
        "</div>"
    )
    html += footer_html
    
    html += "</body></html>"
    
    return html
