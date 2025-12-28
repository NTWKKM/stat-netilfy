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
            Compatibility shim that restores sklearn-style data validation expected by FirthLogisticRegression.
            
            Validates the provided input arrays using sklearn's check utilities. If `y` is None, returns the validated feature array for `X`; otherwise returns the validated `(X, y)` pair. Additional validation options may be passed via `**check_params`. The parameters `reset` and `validate_separately` are accepted to match the original signature and may be forwarded to the underlying validators when applicable.
              
            Parameters:
                X: array-like
                    Feature data to validate.
                y: array-like, optional
                    Target array to validate alongside `X`. If omitted, only `X` is validated.
                reset: bool
                    Accepted for signature compatibility; not used by this shim itself.
                validate_separately: bool
                    Accepted for signature compatibility; not used by this shim itself.
                **check_params:
                    Additional keyword arguments forwarded to sklearn.utils.validation.check_array or check_X_y.
            
            Returns:
                ndarray or (ndarray, ndarray):
                    The validated `X` array if `y` is None, otherwise a tuple `(X_validated, y_validated)`.
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

# Suppress specific convergence warnings from statsmodels
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*")

def clean_numeric_value(val):
    """
    Convert various scalar inputs into a cleaned numeric float.
    
    Parameters:
        val: A scalar value (string, number, or missing). Common non-numeric characters such as leading '>' or '<' and thousands separators (commas) are removed before conversion.
    
    Returns:
        float: The converted numeric value, or `NaN` if the input is missing or cannot be converted.
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
    Provide a sort key that places numeric-like values before non-numeric and missing values.
    
    Returns:
        tuple: A (priority, value) tuple where
            - priority 0: numeric values, with `value` as a float,
            - priority 1: non-numeric values, with `value` as the string representation,
            - priority 2: missing values (`NaN`), with `value` as an empty string.
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
    Fit a binary logistic regression using the specified estimation method.
    
    Parameters:
        y (array-like): Binary outcome vector aligned to X.
        X (DataFrame or array-like): Predictor matrix. An intercept column will be added if absent.
        method (str): Estimation method to use. Accepted values:
            - 'default': statsmodels Logit with default optimizer,
            - 'bfgs': statsmodels Logit with BFGS optimizer,
            - 'firth': Firth-penalized logistic regression (requires optional dependency).
    
    Returns:
        params (pd.Series or None): Estimated coefficients indexed by predictor names (including intercept).
        conf_int (pd.DataFrame or None): Two-column DataFrame of confidence interval bounds.
        pvalues (pd.Series or None): Two-sided p-values for coefficients.
        status (str): "OK" on success; otherwise an error message.
        stats_metrics (dict): Dictionary containing model fit statistics (McFadden R2, Nagelkerke R2).
    """
    stats_metrics = {"mcfadden": np.nan, "nagelkerke": np.nan}
    
    try:
        X_const = sm.add_constant(X, has_constant='add')
        
        if method == 'firth':
            if not HAS_FIRTH:
                return None, None, None, "Library 'firthlogist' not installed.", stats_metrics
            
            fl = FirthLogisticRegression(fit_intercept=False) 
            fl.fit(X_const, y)
            
            coef = np.asarray(fl.coef_).reshape(-1)
            if coef.shape[0] != len(X_const.columns):
                return None, None, None, "Firth output shape mismatch.", stats_metrics
            params = pd.Series(coef, index=X_const.columns)
            pvalues = pd.Series(getattr(fl, "pvals_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
            ci = getattr(fl, "ci_", None)
            conf_int = (
                pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])
                if ci is not None
                else pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
            )
            return params, conf_int, pvalues, "OK", stats_metrics

        elif method == 'bfgs':
            model = sm.Logit(y, X_const)
            result = model.fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const)
            result = model.fit(disp=0)
            
        # 🟢 Calculate R-squared metrics for Standard Logistic Regression
        try:
            llf = result.llf
            llnull = result.llnull
            nobs = result.nobs
            
            # McFadden R2
            mcfadden = 1 - (llf / llnull) if llnull != 0 else np.nan
            
            # Nagelkerke R2
            cox_snell = 1 - np.exp((2/nobs) * (llnull - llf))
            max_r2 = 1 - np.exp((2/nobs) * llnull)
            nagelkerke = cox_snell / max_r2 if max_r2 > 1e-9 else np.nan
            
            stats_metrics = {"mcfadden": mcfadden, "nagelkerke": nagelkerke}
        except (AttributeError, ZeroDivisionError, TypeError) as e:
            logger.debug("Failed to calculate R2 metrics: %s", e)
        
        return result.params, result.conf_int(), result.pvalues, "OK", stats_metrics
        
    except Exception as e:
        logger.exception("Logistic regression failed")
        return None, None, None, str(e), stats_metrics

def get_label(col_name, var_meta):
    """
    Create an HTML-formatted label for a column name, optionally including a secondary metadata label.
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
    Generate an HTML report and effect estimates from univariate and multivariate logistic analyses for a binary outcome.
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

    def count_val(series, v_str):
        return (series.astype(str).apply(lambda x: x.replace('.0','') if x.replace('.','',1).isdigit() else x) == v_str).sum()
        
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
                    levels = sorted(X_raw.dropna().unique(), key=_robust_sort_key)
                except (TypeError, ValueError) as e:
                    logger.warning("Failed to sort levels for %s: %s", col, e)
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
                        # 🟢 Call site updated: unpack 5 values (ignore stats for univariate)
                        params, conf, pvals, status, _ = run_binary_logit(temp_df['y'], temp_df[dummy_cols], method=preferred_method)
                        if status == "OK":
                            # 🟢 NEW: Add Coef list
                            or_lines, coef_lines, p_lines = ["Ref."], ["-"], ["-"]
                            for lvl in levels[1:]:
                                d_name = f"{col}::{lvl}"
                                if d_name in params:
                                    raw_coef = params[d_name] # Extract Coef
                                    odd = np.exp(raw_coef)
                                    ci_l, ci_h = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    
                                    coef_lines.append(f"{raw_coef:.3f}") # Format Coef
                                    or_lines.append(f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})")
                                    p_lines.append(fmt_p(pv))
                                    or_results[f"{col}: {lvl} vs {levels[0]}"] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv, 'coef': raw_coef}
                                else: 
                                    or_lines.append("-"); p_lines.append("-"); coef_lines.append("-")
                            
                            res['or'] = "<br>".join(or_lines)
                            res['coef'] = "<br>".join(coef_lines) # Store for HTML
                            res['p_or'] = "<br>".join(p_lines)
                        else: res['or'] = "-"; res['coef'] = "-"
                    else: res['or'] = "-"; res['coef'] = "-"
                else: res['or'] = "-"; res['coef'] = "-"

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
                    # 🟢 Call site updated: unpack 5 values
                    params, Hex_conf, pvals, status, _ = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                    if status == "OK" and 'x' in params:
                        raw_coef = params['x'] # Extract Coef
                        odd = np.exp(raw_coef)
                        ci_l, ci_h = np.exp(Hex_conf.loc['x'][0]), np.exp(Hex_conf.loc['x'][1])
                        pv = pvals['x']
                        
                        res['coef'] = f"{raw_coef:.3f}" # Store for HTML
                        res['or'] = f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})"
                        res['p_or'] = pv
                        or_results[col] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv, 'coef': raw_coef}
                    else: res['or'] = "-"; res['coef'] = "-"
                else: res['or'] = "-"; res['coef'] = "-"

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
        mv_metrics_text = ""  # For R-squared storage

        def _is_candidate_valid(col: str) -> bool:
            """
            Determine whether a column has enough valid observations to be considered for multivariable modeling.
            """
            mode = mode_map.get(col, "linear")
            series = df_aligned[col]
            if mode == "categorical":
                return series.notna().sum() > 5
            return series.apply(clean_numeric_value).notna().sum() > 5

        cand_valid = [c for c in candidates if _is_candidate_valid(c)]
        
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
                # 🟢 Call site updated: Capture mv_stats
                params, conf, pvals, status, mv_stats = run_binary_logit(multi_data['y'], multi_data[predictors], method=preferred_method)
                
                if status == "OK":
                    # ⭐ Format R-squared strings
                    r2_parts = []
                    mcf = mv_stats.get('mcfadden')
                    nag = mv_stats.get('nagelkerke')
                    if pd.notna(mcf): r2_parts.append(f"McFadden R² = {mcf:.3f}")
                    if pd.notna(nag): r2_parts.append(f"Nagelkerke R² = {nag:.3f}")
                    
                    if r2_parts:
                          mv_metrics_text = " | ".join(r2_parts)

                    for var in cand_valid:
                        mode = mode_map.get(var, 'linear')
                        
                        # --- Multi: Categorical ---
                        if mode == 'categorical':
                            levels = cat_levels_map.get(var, [])
                            aor_entries = []
                            for lvl in levels[1:]:
                                d_name = f"{var}::{lvl}"
                                if d_name in params:
                                    raw_coef = params[d_name] # Extract Coef
                                    aor = np.exp(raw_coef)
                                    ci_low, ci_high = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    aor_entries.append({'lvl': lvl, 'aor': aor, 'l': ci_low, 'h': ci_high, 'p': pv, 'coef': raw_coef})
                                    aor_results[f"{var}: {lvl} vs {levels[0]}"] = {'aor': aor, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pv, 'coef': raw_coef}
                            results_db[var]['multi_res'] = aor_entries
                        
                        # --- Multi: Linear ---
                        else:
                            if var in params:
                                raw_coef = params[var] # Extract Coef
                                aor = np.exp(raw_coef)
                                ci_low, ci_high = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                                pv = pvals[var]
                                results_db[var]['multi_res'] = {'aor': aor, 'l': ci_low, 'h': ci_high, 'p': pv, 'coef': raw_coef}
                                aor_results[var] = {'aor': aor, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pv, 'coef': raw_coef}

    # --- HTML BUILD ---
    html_rows = []
    current_sheet = ""
    valid_cols_for_html = [c for c in sorted_cols if c in results_db]
    
    # ✅ FIX: Grouping logic to prevent sorting error
    def _get_sheet_name(col_name):
        return col_name.split('_')[0] if '_' in col_name else "Variables"
        
    grouped_cols = sorted(valid_cols_for_html, key=lambda x: (_get_sheet_name(x), x))
    
    for col in grouped_cols:
        if col == outcome_name: continue
        res = results_db[col]
        mode = mode_map.get(col, 'linear')
        
        sheet = _get_sheet_name(col)
        if sheet != current_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='11'>{sheet}</td></tr>") # 🟢 Update colspan for new columns
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
        coef_s = res.get('coef', '-') # Get Univariate Coef
        
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

        # Adjusted OR & Coef
        aor_s, acoef_s, ap_s = "-", "-", "-"
        multi_res = res.get('multi_res')
        
        if multi_res:
            if isinstance(multi_res, list): # Categorical List
                aor_lines, acoef_lines, ap_lines = ["Ref."], ["-"], ["-"]
                for item in multi_res:
                    p_txt = fmt_p(item['p'])
                    # ✅ FIX: Type check before comparison
                    if isinstance(item['p'], (int, float)) and pd.notna(item['p']) and item['p'] < 0.05: 
                        p_txt = f"<span class='sig-p'>{p_txt}*</span>"
                    
                    acoef_lines.append(f"{item['coef']:.3f}")
                    aor_lines.append(f"{item['aor']:.2f} ({item['l']:.2f}-{item['h']:.2f})")
                    ap_lines.append(p_txt)
                aor_s, acoef_s, ap_s = "<br>".join(aor_lines), "<br>".join(acoef_lines), "<br>".join(ap_lines)
            else: # Linear Single Dict
                acoef_s = f"{multi_res['coef']:.3f}"
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
            <td>{coef_s}</td>
            <td>{or_s}</td>
            <td>{res.get('test_name', '-')}</td> 
            <td>{p_col_display}</td>
            <td>{acoef_s}</td>
            <td>{aor_s}</td>
            <td>{ap_s}</td>
        </tr>""")
    
    logger.info("✅ Logistic analysis done (n_multi=%d)", final_n_multi)
    
    # ⭐ ADD METRICS TO HTML
    model_fit_html = ""
    if mv_metrics_text:
        model_fit_html = f"<div style='margin-top: 8px; padding-top: 8px; border-top: 1px dashed #ccc; color: {COLORS['primary_dark']};'><b>Model Fit:</b> {mv_metrics_text}</div>"

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
                <th>Crude Coef.</th> <th>Crude OR (95% CI)</th>
                <th>Test Used</th> <th>Crude P-value</th>
                <th>Adj. Coef.</th> <th>aOR (95% CI) <sup style='color:{COLORS['danger']}; font-weight:bold;'>†</sup><br><span style='font-size:0.8em; font-weight:normal'>(n={final_n_multi})</span></th>
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
            {model_fit_html}
        </div>
    </div>
    </div><br>
    """
    
    return html_table, or_results, aor_results

def generate_forest_plot_html(or_results, aor_results, plot_title="Forest Plots: Odds Ratios"):
    """
    Builds HTML containing forest plots (and interpretation) for provided univariate and multivariate odds-ratio results.
    """
    html_parts = [f"<h2 style='margin-top:30px; color:{COLORS['primary']};'>{plot_title}</h2>"]
    has_plot = False

    if or_results:
        df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_results.items()])
        if not df_crude.empty:
            fig = create_forest_plot(df_crude, 'or', 'ci_low', 'ci_high', 'variable', 'p_value', "<b>Univariable: Crude OR</b>", "Odds Ratio", 1.0)
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=True))
            has_plot = True

    if aor_results:
        df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_results.items()])
        if not df_adj.empty:
            fig = create_forest_plot(df_adj, 'aor', 'ci_low', 'ci_high', 'variable', 'p_value', "<b>Multivariable: Adjusted OR</b>", "Adjusted OR", 1.0)
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
    Builds a standalone HTML document containing logistic regression results and corresponding forest plots for a specified binary outcome.
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
    full_html += """<div class='report-footer'>&copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM n donate</a> | Powered by GitHub, Gemini, Streamlit</div>"""
    
    return full_html, or_res, aor_res
