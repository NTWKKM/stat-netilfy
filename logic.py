import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
import html
import streamlit as st 

from logger import get_logger
from tabs._common import get_color_palette
from forest_plot_lib import create_forest_plot

# Get logger instance for this module
logger = get_logger(__name__)
# Get unified color palette
COLORS = get_color_palette()

# ✅ TRY IMPORT FIRTHLOGIST WITH SKLEARN PATCH
try:
    from firthlogist import FirthLogisticRegression
    
    # ------------------------------------------------------------------
    # FIX: Monkeypatch for sklearn >= 1.6 where _validate_data is removed
    # ------------------------------------------------------------------
    if not hasattr(FirthLogisticRegression, "_validate_data"):
        from sklearn.utils.validation import check_X_y, check_array
        
        logger.info("🔧 Applying sklearn 1.6+ compatibility patch to FirthLogisticRegression")
        
        def _validate_data_patch(self, X, y=None, reset=True, validate_separately=False, **check_params):
            """
            Compatibility shim that validates input arrays and optionally the target to match sklearn's `_validate_data` behavior.
            
            Parameters:
                X: array-like
                    Feature matrix to validate.
                y: array-like, optional
                    Target array to validate. If omitted, only `X` is validated.
                reset: bool
                    Present for signature compatibility; has no effect on validation.
                validate_separately: bool
                    Present for signature compatibility; has no effect on validation.
                **check_params:
                    Additional keyword arguments forwarded to the underlying sklearn validation routines (e.g., `dtype`, `ensure_2d`).
            
            Returns:
                numpy.ndarray or tuple
                    The validated `X` if `y` is None, otherwise a tuple `(X_validated, y_validated)`.
            """
            if y is None:
                return check_array(X, **check_params)
            else:
                return check_X_y(X, y, **check_params)
        
        FirthLogisticRegression._validate_data = _validate_data_patch
        logger.info("✅ Patch applied successfully")
    # ------------------------------------------------------------------

    HAS_FIRTH = True
    logger.info("✅ firthlogist imported successfully")
    
except ImportError as e:
    HAS_FIRTH = False
    logger.warning(f"⚠️  firthlogist not available: {str(e)}")
except (AttributeError, TypeError) as e:
    logger.exception("❌ Error patching firthlogist")
    HAS_FIRTH = False

warnings.filterwarnings("ignore")

def clean_numeric_value(val):
    """
    Convert a value into a numeric float usable for analysis.
    
    Parameters:
        val: The input value to convert; may be numeric, a string containing numeric characters,
             or a sentinel like NaN.
    
    Returns:
        float: The numeric value parsed from `val`, or `numpy.nan` if `val` is missing or cannot be parsed.
    """
    if pd.isna(val): 
        return np.nan
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan

def _robust_sort_key(x):
    """
    Produce a sorting key that orders numeric values before strings and places missing values last.
    
    The returned tuple is used as a sort key: the first element is a priority integer (0 for numeric values, 1 for non-numeric strings, 2 for missing/NaN), and the second element is the comparable key (a float for numeric values, the string representation for non-numeric values, or an empty string for missing values).
    
    Returns:
        tuple: (priority, key) where `priority` is 0, 1, or 2 and `key` is the value to compare within that priority.
    """
    try:
        # Check if it's already a number or can be one
        if pd.isna(x): return (2, "")
        val = float(x)
        return (0, val)  # Numeric first
    except (ValueError, TypeError):
        return (1, str(x))    # Then string

def run_binary_logit(y, X, method='default'):
    """
    Fit a binary logistic regression model using the requested estimation method.
    
    Parameters:
        y (array-like): Binary outcome vector aligned with rows of X.
        X (DataFrame or array-like): Predictor matrix; a constant column will be added if missing.
        method (str): Estimation method to use. Supported values:
            - 'firth': Firth's penalized likelihood (requires firthlogist); returns an explicit error message if unavailable.
            - 'bfgs': Maximum likelihood using BFGS optimization.
            - any other value (default): Maximum likelihood with the statsmodels default optimizer.
    
    Returns:
        params (Series or ndarray or None): Estimated coefficients indexed by predictor names, or None on failure.
        conf_int (DataFrame or None): Two-column confidence interval DataFrame indexed by predictor names, or None on failure.
        pvalues (Series or ndarray or None): Two-sided p-values for coefficients, or None on failure.
        status (str): Status message; "OK" on success or an error description on failure.
    """
    try:
        X_const = sm.add_constant(X, has_constant='add')
        
        if method == 'firth':
            if not HAS_FIRTH:
                return None, None, None, "Library 'firthlogist' not installed."
            
            fl = FirthLogisticRegression(fit_intercept=False) 
            fl.fit(X_const, y)
            
            coef = np.asarray(fl.coef_).reshape(-1)
            if coef.shape[0] != len(X_const.columns):
                return None, None, None, "Firth output shape mismatch."
            params = pd.Series(coef, index=X_const.columns)
            pvalues = pd.Series(getattr(fl, "pvals_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
            ci = getattr(fl, "ci_", None)
            conf_int = (
                pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])
                if ci is not None
                else pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
            )
            return params, conf_int, pvalues, "OK"

        elif method == 'bfgs':
            model = sm.Logit(y, X_const).fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const).fit(disp=0)
            
        return model.params, model.conf_int(), model.pvalues, "OK"
        
    except Exception as e:
        logger.exception("Logistic regression failed")
        return None, None, None, str(e)

def get_label(col_name, var_meta):
    """
    Create an HTML label for a column name, optionally including a secondary human-readable label from variable metadata.
    
    If var_meta contains an entry for col_name (or for the suffix after the first underscore), the corresponding 'label' value is appended on a new line in muted styling.
    
    Parameters:
        col_name (str): Column name to display.
        var_meta (dict | None): Optional mapping of column keys to metadata dictionaries; expected to contain a 'label' key when present.
    
    Returns:
        str: HTML fragment with the column name in bold, optionally followed by a muted secondary label on a new line.
    """
    display_name = col_name 
    secondary_label = ""
    if var_meta:
        if col_name in var_meta and 'label' in var_meta[col_name]:
            secondary_label = var_meta[col_name]['label']
        elif '_' in col_name:
            parts = col_name.split('_', 1)
            if len(parts) > 1:
                short_name = parts[1]
                if short_name in var_meta and 'label' in var_meta[short_name]:
                    secondary_label = var_meta[short_name]['label']

    safe_name = html.escape(str(display_name))
    if secondary_label:
        safe_label = html.escape(str(secondary_label))
        return f"<b>{safe_name}</b><br><span style='color:#666; font-size:0.9em'>{safe_label}</span>"
    else:
        return f"<b>{safe_name}</b>"

@st.cache_data(show_spinner=False)
def analyze_outcome(outcome_name, df, var_meta=None, method='auto'):
    """
    Perform univariate and multivariate logistic regression analyses for a binary outcome and produce an HTML results table plus crude and adjusted odds ratio dictionaries.
    
    Analyzes the specified binary outcome in df, auto-detects per-variable mode (categorical vs linear) with optional overrides via var_meta, computes descriptive statistics and univariate tests (chi-square or Mann–Whitney), fits crude logistic regressions to produce unadjusted odds ratios (ORs), selects candidate predictors by screening p < 0.20, fits a multivariate logistic model to produce adjusted ORs (aORs), and formats results into an HTML table. The function chooses an estimation method automatically (including Firth correction when appropriate and available) and performs complete-case analysis for multivariate modeling.
    
    Parameters:
        outcome_name (str): Column name of the binary outcome in df.
        df (pandas.DataFrame): DataFrame containing the outcome and predictor columns.
        var_meta (dict | None): Optional metadata per variable; may include 'type' to override auto-detected mode and 'map' for categorical label mapping.
        method (str): Estimation method hint: 'auto', 'firth', 'bfgs', or 'default'. 'auto' selects an appropriate method based on data and availability.
    
    Returns:
        tuple:
            html_table (str): HTML string containing the formatted results table and summary.
            or_results (dict): Mapping of variable (or level) to crude OR info: {'or', 'ci_low', 'ci_high', 'p_value'}.
            aor_results (dict): Mapping of variable (or level) to adjusted OR info: {'aor', 'ci_low', 'ci_high', 'p_value'}.
    """
    
    logger.log_analysis(analysis_type="Logistic Regression", outcome=outcome_name, n_vars=len(df.columns) - 1, n_samples=len(df))
    
    if outcome_name not in df.columns:
        return f"<div class='alert'>⚠️ Outcome '{outcome_name}' not found.</div>", {}, {}
    
    y_raw = df[outcome_name].dropna()
    unique_outcomes = set(y_raw.unique())
    
    if len(unique_outcomes) != 2:
        return f"<div class='alert'>❌ Invalid Outcome: Expected 2 values, found {len(unique_outcomes)}.</div>", {}, {}
    
    if not unique_outcomes.issubset({0, 1}):
        sorted_outcomes = sorted(unique_outcomes, key=str)
        outcome_map = {sorted_outcomes[0]: 0, sorted_outcomes[1]: 1}
        y = y_raw.map(outcome_map).astype(int)
    else:
        y = y_raw.astype(int)
    
    df_aligned = df.loc[y.index]
    total_n = len(y)
    
    candidates = [] 
    results_db = {} 
    
    # 🟢 FIX: Ensure all columns are strings before sorting to prevent TypeError
    sorted_cols = sorted(df.columns.astype(str))

    # 🟢 TRACKING MODES & METADATA FOR MULTIVARIATE
    mode_map = {} 
    cat_levels_map = {}

    # Auto-method selection
    has_perfect_separation = False
    if method == 'auto':
        for col in sorted_cols:
            if col == outcome_name: continue
            if col not in df_aligned.columns: continue
            try:
                X_num = df_aligned[col].apply(clean_numeric_value)
                if X_num.nunique() > 1:
                    if (pd.crosstab(X_num, y) == 0).any().any():
                        has_perfect_separation = True
                        break
            except Exception:
                logger.debug("Perfect separation check failed for column %s", col)
                continue
    
    preferred_method = 'bfgs'
    if method == 'auto' and HAS_FIRTH and (has_perfect_separation or len(df)<50 or (y==1).sum()<20):
        preferred_method = 'firth'
    elif method == 'firth': preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
    elif method == 'default': preferred_method = 'default'

    def fmt_p(val):
        """
        Format a p-value (or value convertible to float) for display with clipping and special thresholds.
        
        Parameters:
            val: A numeric value or value convertible to float; may be NaN or non-numeric.
        
        Returns:
            A string representation: `"-"` for NaN or non-convertible inputs, `"<0.001"` if the value is less than 0.001, `">0.999"` if greater than 0.999, or the value formatted with three decimal places (e.g., `"0.123"`).
        """
        if pd.isna(val): return "-"
        try:
            val = float(val)
            val = max(0, min(1, val))
            if val < 0.001: return "<0.001"
            if val > 0.999: return ">0.999"
            return f"{val:.3f}"
        except (ValueError, TypeError):
            return "-"

    or_results = {}
    
    # --- UNIVARIATE ANALYSIS LOOP ---
    with logger.track_time("univariate_analysis", log_level="debug"):
        for col in sorted_cols:
            if col == outcome_name: continue
            if col not in df_aligned.columns: continue
            if df_aligned[col].isnull().all(): continue

            res = {'var': col}
            X_raw = df_aligned[col]
            X_num = X_raw.apply(clean_numeric_value)
            
            X_neg = X_raw[y == 0]
            X_pos = X_raw[y == 1]
            
            orig_name = col.split('_', 1)[1] if len(col.split('_', 1)) > 1 else col
            unique_vals = X_num.dropna().unique()
            unique_count = len(unique_vals)
            
            # --- 1. DETERMINE MODE (Auto or User Override) ---
            mode = 'linear' # Default
            is_binary = set(unique_vals).issubset({0, 1})
            
            # Auto-detection logic:
            if is_binary:
                mode = 'categorical'
            elif unique_count < 10:
                decimals_pct = sum(1 for v in unique_vals if not float(v).is_integer()) / len(unique_vals) if len(unique_vals)>0 else 0
                if decimals_pct < 0.3:
                    mode = 'categorical'
            
            # User Override via var_meta
            if var_meta:
                key = col if col in var_meta else orig_name
                if key in var_meta:
                    user_mode = var_meta[key].get('type') 
                    if user_mode:
                        t = user_mode.lower()
                        if 'cat' in t: mode = 'categorical'
                        elif 'simp' in t: mode = 'categorical' # 🟡 CHANGED: Map Simple -> Categorical
                        elif 'lin' in t or 'cont' in t: mode = 'linear'

            mode_map[col] = mode
            
            # --- 2. PREPARE LEVELS (For Categorical) ---
            levels = []
            if mode == 'categorical':
                try:
                    # 🟢 IMPROVED: Use robust_sort_key for mixed numeric/string
                    levels = sorted(X_raw.dropna().unique(), key=_robust_sort_key)
                except Exception as e:
                    logger.warning(f"Failed to sort levels for {col}: {e}")
                    levels = sorted(X_raw.astype(str).unique())
                cat_levels_map[col] = levels

            # =========================================================
            # 🟢 MODE A: CATEGORICAL (All Levels: Ref vs Lvl 1, Ref vs Lvl 2...)
            # =========================================================
            if mode == 'categorical':
                n_used = len(X_raw.dropna())
                mapper = {}
                if var_meta:
                    key = col if col in var_meta else orig_name
                    if key in var_meta: mapper = var_meta[key].get('map', {})

                desc_tot, desc_neg, desc_pos = [f"<span class='n-badge'>n={n_used}</span>"], [f"<span class='n-badge'>n={len(X_neg.dropna())}</span>"], [f"<span class='n-badge'>n={len(X_pos.dropna())}</span>"]
                
                def count_val(series, v_str):
                      """
                      Count occurrences of a given string value in a pandas Series after normalizing numeric-like entries.
                      
                      Normalizes each element by converting it to a string and, for values that appear numeric, removing a trailing ".0" so that numeric 1 and "1.0" both compare equal to "1".
                      
                      Parameters:
                          series (pandas.Series): Input series whose values will be string-normalized for comparison.
                          v_str (str): Target string value to count after normalization.
                      
                      Returns:
                          int: Number of elements equal to `v_str` after normalization.
                      """
                      return (series.astype(str).apply(lambda x: x.replace('.0','') if x.replace('.','',1).isdigit() else x) == v_str).sum()

                for lvl in levels:
                    lbl_txt = str(lvl)
                    if str(lvl).endswith('.0'): lbl_txt = str(int(float(lvl)))
                    lbl_display = mapper.get(lvl, lbl_txt)

                    c_all = count_val(X_raw, str(lvl).replace('.0','') if str(lvl).endswith('.0') else str(lvl))
                    if c_all == 0: c_all = (X_raw == lvl).sum()
                    
                    p_all = (c_all/n_used)*100 if n_used else 0
                    c_n = count_val(X_neg, str(lvl).replace('.0','') if str(lvl).endswith('.0') else str(lvl))
                    p_n = (c_n/len(X_neg.dropna()))*100 if len(X_neg.dropna()) else 0
                    c_p = count_val(X_pos, str(lvl).replace('.0','') if str(lvl).endswith('.0') else str(lvl))
                    p_p = (c_p/len(X_pos.dropna()))*100 if len(X_pos.dropna()) else 0
                    
                    desc_tot.append(f"{lbl_display}: {c_all} ({p_all:.1f}%)")
                    desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                    desc_pos.append(f"{c_p} ({p_p:.1f}%)")
                
                res['desc_total'] = "<br>".join(desc_tot)
                res['desc_neg'] = "<br>".join(desc_neg)
                res['desc_pos'] = "<br>".join(desc_pos)
                
                # Chi-Square (All Levels)
                try:
                    ct = pd.crosstab(X_raw, y)
                    _, p, _, _ = stats.chi2_contingency(ct) if ct.size > 0 else (0, np.nan, 0, 0)
                    res['p_comp'] = p  # 🟢 ALWAYS store chi-square p-value
                    res['test_name'] = "Chi-square (All Levels)"
                except (ValueError, TypeError) as e:
                    logger.debug("Chi-square test failed for %s: %s", col, e)
                    res['p_comp'], res['test_name'] = np.nan, "-"

                # Regression (Dummies): Ref vs Each Level
                if len(levels) > 1:
                    temp_df = pd.DataFrame({'y': y, 'raw': X_raw}).dropna()
                    dummy_cols = []
                    for lvl in levels[1:]:
                        d_name = f"{col}::{lvl}"
                        temp_df[d_name] = (temp_df['raw'].astype(str) == str(lvl)).astype(int)
                        dummy_cols.append(d_name)
                    
                    if dummy_cols and temp_df[dummy_cols].std().sum() > 0:
                        params, conf, pvals, status = run_binary_logit(temp_df['y'], temp_df[dummy_cols], method=preferred_method)
                        if status == "OK":
                            or_lines, p_lines = ["Ref."], ["-"]
                            for lvl in levels[1:]:
                                d_name = f"{col}::{lvl}"
                                if d_name in params:
                                    odd = np.exp(params[d_name])
                                    ci_l, ci_h = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    or_lines.append(f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})")
                                    p_lines.append(fmt_p(pv))
                                    or_results[f"{col}: {lvl} vs {levels[0]}"] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                                else: or_lines.append("-"); p_lines.append("-")
                            res['or'], res['p_or'] = "<br>".join(or_lines), "<br>".join(p_lines)
                        else: res['or'] = "-"
                    else: res['or'] = "-"
                else: res['or'] = "-"

            # =========================================================
            # 🟢 MODE B: LINEAR (Continuous / Trend)
            # =========================================================
            else:
                n_used = len(X_num.dropna())
                n_before_drop = len(X_num)
                m_t, s_t = X_num.mean(), X_num.std()
                m_n, s_n = pd.to_numeric(X_neg, errors='coerce').mean(), pd.to_numeric(X_neg, errors='coerce').std()
                m_p, s_p = pd.to_numeric(X_pos, errors='coerce').mean(), pd.to_numeric(X_pos, errors='coerce').std()
                
                res['desc_total'] = f"<span class='n-badge'>n={n_used}</span><br>Mean: {m_t:.2f}<br>(SD {s_t:.2f})"
                res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
                res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
                
                # 🟢 Log missing data for Linear mode
                if n_before_drop > n_used:
                    logger.debug(f"Linear mode {col}: Dropped {n_before_drop - n_used} rows with missing values")
                
                try:
                    _, p = stats.mannwhitneyu(pd.to_numeric(X_neg, errors='coerce').dropna(), pd.to_numeric(X_pos, errors='coerce').dropna())
                    res['p_comp'] = p
                    res['test_name'] = "Mann-Whitney U"
                except (ValueError, TypeError) as e:
                    logger.debug("Mann-Whitney test failed for %s: %s", col, e)
                    res['p_comp'], res['test_name'] = np.nan, "-"
                    
                data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
                if not data_uni.empty and data_uni['x'].nunique() > 1:
                    params, Hex_conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                    if status == "OK" and 'x' in params:
                        odd = np.exp(params['x'])
                        ci_l, ci_h = np.exp(Hex_conf.loc['x'][0]), np.exp(Hex_conf.loc['x'][1])
                        pv = pvals['x']
                        res['or'] = f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})"
                        res['p_or'] = pv
                        or_results[col] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                    else: res['or'] = "-"
                else: res['or'] = "-"

            results_db[col] = res
            
            # =========================================================
            # 🟢 FIX: VARIABLE SCREENING FOR MULTIVARIATE
            # =========================================================
            # Always use p_comp (chi-square or Mann-Whitney) for screening
            p_screen = res.get('p_comp', np.nan)
            
            # ✅ FIX: Defensive check: ensure p_screen is numeric before comparison
            if isinstance(p_screen, (int, float)) and pd.notna(p_screen):
                if p_screen < 0.20:
                    candidates.append(col)

    # --- MULTIVARIATE ANALYSIS ---
    with logger.track_time("multivariate_analysis"):
        aor_results = {}
        cand_valid = [c for c in candidates if df_aligned[c].apply(clean_numeric_value).notna().sum() > 5 or c in mode_map]
        
        final_n_multi = 0
        if len(cand_valid) > 0:
            multi_df = pd.DataFrame({'y': y})
            
            # 🟢 CONSTRUCT MULTIVARIATE MATRIX BASED ON MODE
            for c in cand_valid:
                mode = mode_map.get(c, 'linear')
                
                if mode == 'categorical':
                    levels = cat_levels_map.get(c, [])
                    raw_vals = df_aligned[c]
                    if len(levels) > 1:
                        for lvl in levels[1:]:
                            d_name = f"{c}::{lvl}"
                            multi_df[d_name] = (raw_vals.astype(str) == str(lvl)).astype(int)
                
                else: # Linear
                    multi_df[c] = df_aligned[c].apply(clean_numeric_value)

            multi_data = multi_df.dropna()
            final_n_multi = len(multi_data)
            predictors = [col for col in multi_data.columns if col != 'y']

            if not multi_data.empty and final_n_multi > 10 and len(predictors) > 0:
                params, conf, pvals, status = run_binary_logit(multi_data['y'], multi_data[predictors], method=preferred_method)
                
                if status == "OK":
                    for var in cand_valid:
                        mode = mode_map.get(var, 'linear')
                        
                        # --- Multi: Categorical ---
                        if mode == 'categorical':
                            levels = cat_levels_map.get(var, [])
                            aor_entries = []
                            for lvl in levels[1:]:
                                d_name = f"{var}::{lvl}"
                                if d_name in params:
                                    aor = np.exp(params[d_name])
                                    ci_low, ci_high = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    aor_entries.append({'lvl': lvl, 'aor': aor, 'l': ci_low, 'h': ci_high, 'p': pv})
                                    aor_results[f"{var}: {lvl} vs {levels[0]}"] = {'aor': aor, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pv}
                            results_db[var]['multi_res'] = aor_entries
                        
                        # --- Multi: Linear ---
                        else:
                            if var in params:
                                aor = np.exp(params[var])
                                ci_low, ci_high = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                                pv = pvals[var]
                                results_db[var]['multi_res'] = {'aor': aor, 'l': ci_low, 'h': ci_high, 'p': pv}
                                aor_results[var] = {'aor': aor, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pv}
    # --- HTML BUILD ---
    html_rows = []
    current_sheet = ""
    valid_cols_for_html = [c for c in sorted_cols if c in results_db]
    
    # ✅ FIX: Grouping logic to prevent sorting error
    def _get_sheet_name(col_name):
        """
        Extract the sheet name from a column by taking the substring before the first underscore.
        
        Parameters:
            col_name (str): Column name to extract the sheet from.
        
        Returns:
            sheet_name (str): The substring before the first underscore in `col_name`, or "Variables" if `col_name` contains no underscore.
        """
        return col_name.split('_')[0] if '_' in col_name else "Variables"
        
    grouped_cols = sorted(valid_cols_for_html, key=lambda x: (_get_sheet_name(x), x))
    
    for col in grouped_cols:
        if col == outcome_name: continue
        res = results_db[col]
        mode = mode_map.get(col, 'linear')
        
        sheet = _get_sheet_name(col)
        if sheet != current_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='9'>{sheet}</td></tr>")
            current_sheet = sheet
            
        lbl = get_label(col, var_meta)
        
        # 🟢 IMPROVED: Add mode badge with icon (Only 2 modes now)
        mode_badge = {
            'categorical': '📊 (All Levels vs Ref)',
            'linear': '📉 (Trend)'
        }
        if mode in mode_badge:
            lbl += f"<br><span style='font-size:0.8em; color:#888'>{mode_badge[mode]}</span>"
        
        or_s = res.get('or', '-')
        
        # P-value display
        if mode == 'categorical': 
            p_col_display = res.get('p_or', '-') # Multiline
        else:
            p_val = res.get('p_comp', np.nan) # Chi2/Mann-Whitney for single line
            p_s = fmt_p(p_val)
            # ✅ FIX: Type check before comparison
            if isinstance(p_val, (int, float)) and pd.notna(p_val) and p_val < 0.05: 
                p_s = f"<span class='sig-p'>{p_s}*</span>"
            p_col_display = p_s

        # Adjusted OR
        aor_s, ap_s = "-", "-"
        multi_res = res.get('multi_res')
        
        if multi_res:
            if isinstance(multi_res, list): # Categorical List
                aor_lines, ap_lines = ["Ref."], ["-"]
                for item in multi_res:
                    p_txt = fmt_p(item['p'])
                    # ✅ FIX: Type check before comparison
                    if isinstance(item['p'], (int, float)) and pd.notna(item['p']) and item['p'] < 0.05: 
                        p_txt = f"<span class='sig-p'>{p_txt}*</span>"
                    aor_lines.append(f"{item['aor']:.2f} ({item['l']:.2f}-{item['h']:.2f})")
                    ap_lines.append(p_txt)
                aor_s, ap_s = "<br>".join(aor_lines), "<br>".join(ap_lines)
            else: # Linear Single Dict
                aor_s = f"{multi_res['aor']:.2f} ({multi_res['l']:.2f}-{multi_res['h']:.2f})"
                ap_val = multi_res['p']
                ap_txt = fmt_p(ap_val)
                # ✅ FIX: Type check before comparison
                if isinstance(ap_val, (int, float)) and pd.notna(ap_val) and ap_val < 0.05: 
                    ap_txt = f"<span class='sig-p'>{ap_txt}*</span>"
                ap_s = ap_txt
            
        html_rows.append(f"""
        <tr>
            <td>{lbl}</td>
            <td>{res.get('desc_total','')}</td>
            <td>{res.get('desc_neg','')}</td>
            <td>{res.get('desc_pos','')}</td>
            <td>{or_s}</td>
            <td>{res.get('test_name', '-')}</td> 
            <td>{p_col_display}</td>
            <td>{aor_s}</td>
            <td>{ap_s}</td>
        </tr>""")
    
    logger.info("✅ Logistic analysis done (n_multi=%d)", final_n_multi)
    
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
        <b>Method:</b> {preferred_method.capitalize()} Logit. Complete Case Analysis.<br>
        <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 0.9em; color: #666;'>
            <sup style='color:{COLORS['danger']}; font-weight:bold;'>†</sup> <b>aOR:</b> Calculated for variables with Crude P < 0.20 (n_multi={final_n_multi}).<br>
            <b>Modes:</b> 
            📊 Categorical (All Levels vs Reference) | 
            📉 Linear (Per-unit Trend)
        </div>
    </div>
    </div><br>
    """
    
    return html_table, or_results, aor_results

def generate_forest_plot_html(or_results, aor_results, plot_title="Forest Plots: Odds Ratios"):
    """
    Builds HTML containing forest plots for univariable (crude) and multivariable (adjusted) odds ratios.
    
    Parameters:
        or_results (dict): Mapping of variable name -> metrics for univariable results. Each metrics dict is expected to include keys:
            - 'or' (float): odds ratio
            - 'ci_low' (float): lower bound of the 95% confidence interval
            - 'ci_high' (float): upper bound of the 95% confidence interval
            - 'p_value' (float): p-value for the estimate
        aor_results (dict): Mapping of variable name -> metrics for multivariable results. Each metrics dict is expected to include keys:
            - 'aor' (float): adjusted odds ratio
            - 'ci_low' (float): lower bound of the 95% confidence interval
            - 'ci_high' (float): upper bound of the 95% confidence interval
            - 'p_value' (float): p-value for the estimate
        plot_title (str): Title displayed above the plots.
    
    Returns:
        str: HTML fragment containing embedded forest plot(s) when results are present, or a short message indicating no plots are available. When plots are included, an interpretation note for OR values is appended.
    """
    html_parts = [f"<h2 style='margin-top:30px; color:{COLORS['primary']};'>{plot_title}</h2>"]
    has_plot = False

    if or_results:
        df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_results.items()])
        if not df_crude.empty:
            fig = create_forest_plot(df_crude, 'or', 'ci_low', 'ci_high', 'p_value', 'variable', "<b>Univariable: Crude OR</b>", "Odds Ratio", 1.0)
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=True))
            has_plot = True

    if aor_results:
        df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_results.items()])
        if not df_adj.empty:
            fig = create_forest_plot(df_adj, 'aor', 'ci_low', 'ci_high', 'p_value', 'variable', "<b>Multivariable: Adjusted OR</b>", "Adjusted OR", 1.0)
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            has_plot = True

    if not has_plot: 
        html_parts.append("<p style='color:#999'>No results for forest plots.</p>")
    else:
        html_parts.append(f"""
        <div style='margin-top:20px; padding:15px; background:#f8f9fa; border-left:4px solid {COLORS.get('primary', '#218084')};'>
            <b>Interpretation:</b> OR > 1 (Risk Factor), OR < 1 (Protective), CI crosses 1 (Not Sig).
        </div>
        """)
    return "".join(html_parts)

def process_data_and_generate_html(df, target_outcome, var_meta=None, method='auto'):
    """
    Assemble a complete HTML report for logistic regression analysis of a specified outcome.
    
    Parameters:
        df (pandas.DataFrame): Dataset containing predictors and the outcome column.
        target_outcome (str): Name of the outcome column to analyze.
        var_meta (dict | None): Optional metadata for variables (labels, modes, overrides).
        method (str): Estimation method hint passed to analysis functions ('auto', 'firth', 'bfgs', etc.).
    
    Returns:
        full_html (str): Complete HTML document containing the result table, forest plots, styles, and footer.
        or_res (dict): Unadjusted (crude) odds ratio results produced by the univariate analysis.
        aor_res (dict): Adjusted odds ratio results produced by the multivariate analysis.
    """
    css = f"""<style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; }}
        .table-container {{ background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0; min-width: 800px; }}
        th {{ background-color: {COLORS['primary_dark']}; color: #fff; padding: 12px; position: sticky; top: 0; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .outcome-title {{ background-color: {COLORS['primary_dark']}; color: white; padding: 15px; font-weight: bold; border-radius: 8px 8px 0 0; }}
        .sig-p {{ color: {COLORS['danger']}; font-weight: bold; background-color: #ffebee; padding: 2px 4px; border-radius: 4px; }}
        .sheet-header td {{ background-color: #e8f4f8; color: {COLORS['primary']}; font-weight: bold; }}
        .n-badge {{ font-size: 0.75em; color: #888; background: #eee; padding: 1px 4px; border-radius: 3px; }}
        .report-footer {{ text-align: right; font-size: 0.75em; color: {COLORS['text_secondary']}; margin-top: 20px; border-top: 1px dashed {COLORS['border']}; padding-top: 10px; }}
        a {{ color: {COLORS['primary']}; text-decoration: none; }}
    </style>"""
    
    html_table, or_res, aor_res = analyze_outcome(target_outcome, df, var_meta, method=method)
    plot_html = generate_forest_plot_html(or_res, aor_res)
    
    full_html = f"<!DOCTYPE html><html><head>{css}</head><body><h1>Logistic Regression Report</h1>{html_table}{plot_html}"
    full_html += "<div class='report-footer'>&copy; 2025 NTWKKM. Powered by GitHub, Gemini, Streamlit</div></body></html>"
    
    return full_html, or_res, aor_res