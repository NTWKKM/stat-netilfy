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

# ✅ TRY IMPORT FIRTHLOGIST
try:
    from firthlogist import FirthLogisticRegression
    
    # ------------------------------------------------------------------
    # FIX: Monkeypatch for sklearn >= 1.6 where _validate_data is removed
    # ------------------------------------------------------------------
    if not hasattr(FirthLogisticRegression, "_validate_data"):
        from sklearn.utils.validation import check_X_y, check_array
        
        logger.info("🔧 Applying sklearn 1.6+ compatibility patch to FirthLogisticRegression")
        
        def _validate_data_patch(self, X, y=None, reset=True, validate_separately=False, **check_params):
            """Shim to restore _validate_data for firthlogist compatibility with sklearn 1.6+"""
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
    """Normalize a value into a numeric float suitable for analysis."""
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
    🟢 IMPROVED: Robust sorting key for mixed numeric/string levels.
    Returns (sort_priority, value) tuple:
    - (0, numeric_value) for numeric values
    - (1, string_value) for non-numeric values
    """
    try:
        return (0, float(x))  # Numeric first
    except (ValueError, TypeError):
        return (1, str(x))    # Then string

def run_binary_logit(y, X, method='default'):
    """Perform binary logistic regression using the specified estimation method."""
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
    """Build an HTML label for a column."""
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
    Analyze outcome with support for 3 OR modes:
    - 'Categorical': All levels vs Reference (Ref vs 1, Ref vs 2...)
    - 'Simple': Risk vs Reference (binary comparison, single line)
    - 'Linear': Continuous/Trend (per-unit increase)
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
    sorted_cols = sorted(df.columns)

    # 🟢 TRACKING MODES & METADATA FOR MULTIVARIATE
    mode_map = {} 
    cat_levels_map = {}

    # Auto-method selection
    has_perfect_separation = False
    if method == 'auto':
        for col in sorted_cols:
            if col == outcome_name: continue
            try:
                X_num = df_aligned[col].apply(clean_numeric_value)
                if X_num.nunique() > 1:
                    if (pd.crosstab(X_num, y) == 0).any().any():
                        has_perfect_separation = True
                        break
            except: continue
    
    preferred_method = 'bfgs'
    if method == 'auto' and HAS_FIRTH and (has_perfect_separation or len(df)<50 or (y==1).sum()<20):
        preferred_method = 'firth'
    elif method == 'firth': preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
    elif method == 'default': preferred_method = 'default'

    def fmt_p(val):
        if pd.isna(val): return "-"
        val = max(0, min(1, val))
        if val < 0.001: return "<0.001"
        if val > 0.999: return ">0.999"
        return f"{val:.3f}"

    or_results = {}
    
    # --- UNIVARIATE ANALYSIS LOOP ---
    with logger.track_time("univariate_analysis", log_level="debug"):
        for col in sorted_cols:
            if col == outcome_name: continue
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
            # - Binary (0/1) → Categorical (for 2-way comparison)
            # - Few discrete levels (< 10) → Categorical  
            # - Many/continuous levels → Linear
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
                    user_mode = var_meta[key].get('type') # Expect 'Categorical', 'Linear', 'Simple'
                    if user_mode:
                        t = user_mode.lower()
                        if 'cat' in t: mode = 'categorical'
                        elif 'simp' in t: mode = 'simple'
                        elif 'lin' in t or 'cont' in t: mode = 'linear'

            mode_map[col] = mode
            
            # --- 2. PREPARE LEVELS (Shared for Cat/Simple) ---
            levels = []
            if mode in ['categorical', 'simple']:
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
                    res['p_comp'] = p
                    res['test_name'] = "Chi-square (All Levels)"
                except: res['p_comp'], res['test_name'] = np.nan, "-"

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
            # 🟢 MODE B: SIMPLE (Risk vs Ref / Single Line)
            # =========================================================
            elif mode == 'simple':
                # 🟢 IMPROVED: Allow custom reference level via var_meta
                user_ref = None
                if var_meta:
                    key = col if col in var_meta else orig_name
                    if key in var_meta:
                        user_ref = var_meta[key].get('ref_level')
                
                # Use custom ref if specified, otherwise first level
                ref_val = user_ref if user_ref is not None else levels[0]
                
                # 🟢 Validate reference level exists
                if ref_val not in levels:
                    logger.warning(f"Reference level '{ref_val}' not in levels {levels} for {col}. Using first level.")
                    ref_val = levels[0]
                
                X_bin = (X_raw.astype(str) != str(ref_val)).astype(int)
                X_bin[X_raw.isna()] = np.nan 
                
                n_used = len(X_bin.dropna())
                n_before_drop = len(X_bin)
                
                desc_tot, desc_neg, desc_pos = [f"<span class='n-badge'>n={n_used}</span>"], [f"<span class='n-badge'>n={len(X_neg.dropna())}</span>"], [f"<span class='n-badge'>n={len(X_pos.dropna())}</span>"]
                
                # Show breakdown with Ref marked
                for lvl in levels:
                    lbl_txt = str(lvl)
                    if str(lvl).endswith('.0'): lbl_txt = str(int(float(lvl)))
                    c_all = (X_raw.astype(str) == str(lvl)).sum()
                    if c_all == 0: c_all = (X_raw == lvl).sum()
                    
                    p_all = (c_all/n_used)*100 if n_used else 0
                    c_n = (X_neg.astype(str) == str(lvl)).sum()
                    c_p = (X_pos.astype(str) == str(lvl)).sum()
                    
                    marker = " (Ref)" if lvl == ref_val else ""
                    desc_tot.append(f"{lbl_txt}{marker}: {c_all} ({p_all:.1f}%)")
                    desc_neg.append(f"{c_n}")
                    desc_pos.append(f"{c_p}")
                
                res['desc_total'] = "<br>".join(desc_tot)
                res['desc_neg'] = "<br>".join(desc_neg)
                res['desc_pos'] = "<br>".join(desc_pos)
                
                # 🟢 IMPROVED: Log missing data for Simple mode
                if n_before_drop > n_used:
                    logger.debug(f"Simple mode {col}: Dropped {n_before_drop - n_used} rows with missing values")
                
                # Chi-Square (Binary)
                try:
                    ct = pd.crosstab(X_bin, y)
                    _, p, _, _ = stats.chi2_contingency(ct) if ct.size > 0 else (0, np.nan, 0, 0)
                    res['p_comp'] = p
                    res['test_name'] = "Chi-square (Binary)"
                except: res['p_comp'], res['test_name'] = np.nan, "-"
                
                # Regression (Binary): Risk vs Ref
                data_uni = pd.DataFrame({'y': y, 'x': X_bin}).dropna()
                if not data_uni.empty and data_uni['x'].nunique() > 1:
                    params, conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                    if status == "OK" and 'x' in params:
                        odd = np.exp(params['x'])
                        ci_l, ci_h = np.exp(conf.loc['x'][0]), np.exp(conf.loc['x'][1])
                        pv = pvals['x']
                        
                        label_ref = str(ref_val).replace('.0','')
                        # Label Risk as "Others" if multiple non-Ref levels, else specific level
                        non_ref_levels = [str(l).replace('.0','') for l in levels if l != ref_val]
                        label_risk = "Others" if len(non_ref_levels) > 1 else non_ref_levels[0] if non_ref_levels else "Risk"
                        
                        res['or'] = f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})"
                        res['p_or'] = pv
                        or_results[f"{col} ({label_risk} vs {label_ref})"] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                    else: res['or'] = "-"
                else: res['or'] = "-"

            # =========================================================
            # 🟢 MODE C: LINEAR (Continuous / Trend)
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
                except: res['p_comp'], res['test_name'] = np.nan, "-"

                data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
                if not data_uni.empty and data_uni['x'].nunique() > 1:
                    params, conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                    if status == "OK" and 'x' in params:
                        odd = np.exp(params['x'])
                        ci_l, ci_h = np.exp(conf.loc['x'][0]), np.exp(conf.loc['x'][1])
                        pv = pvals['x']
                        res['or'] = f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})"
                        res['p_or'] = pv
                        or_results[col] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                    else: res['or'] = "-"
                else: res['or'] = "-"

            results_db[col] = res
            
            # Screening P-value
            p_screen = res.get('p_comp', np.nan)
            if pd.isna(p_screen): 
                pv_chk = res.get('p_or', np.nan)
                if isinstance(pv_chk, (int, float)): p_screen = pv_chk
            
            if pd.notna(p_screen) and p_screen < 0.20:
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
                
                elif mode == 'simple':
                    levels = cat_levels_map.get(c, [])
                    if levels:
                        # 🟢 IMPROVED: Use custom ref level if specified
                        user_ref = None
                        if var_meta:
                            key = c if c in var_meta else c.split('_', 1)[1] if '_' in c else c
                            if key in var_meta:
                                user_ref = var_meta[key].get('ref_level')
                        
                        ref_val = user_ref if user_ref is not None else levels[0]
                        if ref_val not in levels:
                            ref_val = levels[0]
                        
                        # Binary: Not Ref = 1
                        multi_df[c] = (df_aligned[c].astype(str) != str(ref_val)).astype(int)
                    else:
                        multi_df[c] = df_aligned[c].apply(clean_numeric_value) # Fallback
                        
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
                                    l, h = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    aor_entries.append({'lvl': lvl, 'aor': aor, 'l': l, 'h': h, 'p': pv})
                                    aor_results[f"{var}: {lvl} vs {levels[0]}"] = {'aor': aor, 'ci_low': l, 'ci_high': h, 'p_value': pv}
                            results_db[var]['multi_res'] = aor_entries
                        
                        # --- Multi: Simple ---
                        elif mode == 'simple':
                             if var in params:
                                aor = np.exp(params[var])
                                l, h = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                                pv = pvals[var]
                                results_db[var]['multi_res'] = {'aor': aor, 'l': l, 'h': h, 'p': pv}
                                
                                levels = cat_levels_map.get(var, [])
                                user_ref = None
                                if var_meta:
                                    key = var if var in var_meta else var.split('_', 1)[1] if '_' in var else var
                                    if key in var_meta:
                                        user_ref = var_meta[key].get('ref_level')
                                
                                ref_val = user_ref if user_ref is not None else levels[0] if levels else None
                                if ref_val and ref_val not in levels:
                                    ref_val = levels[0] if levels else None
                                
                                label_ref = str(ref_val).replace('.0','') if ref_val else "Ref"
                                non_ref_levels = [str(l).replace('.0','') for l in (levels or []) if l != ref_val]
                                label_risk = "Others" if len(non_ref_levels) > 1 else non_ref_levels[0] if non_ref_levels else "Risk"
                                aor_results[f"{var} ({label_risk} vs {label_ref})"] = {'aor': aor, 'ci_low': l, 'ci_high': h, 'p_value': pv}
                        
                        # --- Multi: Linear ---
                        else:
                            if var in params:
                                aor = np.exp(params[var])
                                l, h = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                                pv = pvals[var]
                                results_db[var]['multi_res'] = {'aor': aor, 'l': l, 'h': h, 'p': pv}
                                aor_results[var] = {'aor': aor, 'ci_low': l, 'ci_high': h, 'p_value': pv}

    # --- HTML BUILD ---
    html_rows = []
    current_sheet = ""
    valid_cols_for_html = [c for c in sorted_cols if c in results_db]
    grouped_cols = sorted(valid_cols_for_html, key=lambda x: (x.split('_')[0] if '_' in x else "Variables", x))
    
    for col in grouped_cols:
        if col == outcome_name: continue
        res = results_db[col]
        mode = mode_map.get(col, 'linear')
        
        sheet = col.split('_')[0] if '_' in col else "Variables"
        if sheet != current_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='9'>{sheet}</td></tr>")
            current_sheet = sheet
            
        lbl = get_label(col, var_meta)
        
        # 🟢 IMPROVED: Add mode badge with icon
        mode_badge = {
            'categorical': '📊 (All Levels vs Ref)',
            'simple': '📈 (Risk vs Ref)',
            'linear': '📉 (Trend)'
        }
        if mode in mode_badge:
            lbl += f"<br><span style='font-size:0.8em; color:#888'>{mode_badge[mode]}</span>"
        
        or_s = res.get('or', '-')
        
        # P-value display
        if mode == 'categorical': p_col_display = res.get('p_or', '-') # Multiline
        else:
            p_val = res.get('p_comp', np.nan) # Chi2/Mann-Whitney for single line
            if mode == 'simple': p_val = res.get('p_or', np.nan) # Use OR p-value for simple
            p_s = fmt_p(p_val)
            if pd.notna(p_val) and p_val < 0.05: p_s = f"<span class='sig-p'>{p_s}*</span>"
            p_col_display = p_s

        # Adjusted OR
        aor_s, ap_s = "-", "-"
        multi_res = res.get('multi_res')
        
        if multi_res:
            if isinstance(multi_res, list): # Categorical List
                aor_lines, ap_lines = ["Ref."], ["-"]
                for item in multi_res:
                    p_txt = fmt_p(item['p'])
                    if item['p'] < 0.05: p_txt = f"<span class='sig-p'>{p_txt}*</span>"
                    aor_lines.append(f"{item['aor']:.2f} ({item['l']:.2f}-{item['h']:.2f})")
                    ap_lines.append(p_txt)
                aor_s, ap_s = "<br>".join(aor_lines), "<br>".join(ap_lines)
            else: # Simple/Linear Single Dict
                aor_s = f"{multi_res['aor']:.2f} ({multi_res['l']:.2f}-{multi_res['h']:.2f})"
                ap_val = multi_res['p']
                ap_txt = fmt_p(ap_val)
                if pd.notna(ap_val) and ap_val < 0.05: ap_txt = f"<span class='sig-p'>{ap_txt}*</span>"
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
            📈 Simple (Risk vs Reference) | 
            📉 Linear (Per-unit Trend)
        </div>
    </div>
    </div><br>
    """
    
    return html_table, or_results, aor_results

def generate_forest_plot_html(or_results, aor_results, plot_title="Forest Plots: Odds Ratios"):
    """Generate forest plot HTML."""
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
    """Generate complete HTML report."""
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
    full_html += f"<div class='report-footer'>&copy; 2025 NTWKKM. Powered by GitHub, Gemini, Streamlit</div></body></html>"
    
    return full_html, or_res, aor_res
