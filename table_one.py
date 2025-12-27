import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import html as _html
import warnings
from tabs._common import get_color_palette

def clean_numeric(val):
    if pd.isna(val): 
        return np.nan
    s = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan

# --- Helper: Normality Check ---
def check_normality(series):
    clean = series.dropna()
    if len(clean) < 3 or len(clean) > 5000 or clean.nunique() < 3: 
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _stat, p_sw = stats.shapiro(clean)
        return p_sw > 0.05
    except Exception:
        return False

def format_p(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def get_stats_continuous(series):
    clean = series.apply(clean_numeric).dropna()
    if len(clean) == 0: return "-"
    return f"{clean.mean():.1f} \u00B1 {clean.std():.1f}"

# --- UPDATED: Return list of cats for OR matching ---
def get_stats_categorical_data(series, var_meta=None, col_name=None):
    """
    Returns specific counts and labels to be used for both Display and OR calculation alignment.
    """
    mapper = {}
    if var_meta and col_name:
        key = col_name.split('_')[1] if '_' in col_name else col_name
        if col_name in var_meta: mapper = var_meta[col_name].get('map', {})
        elif key in var_meta: mapper = var_meta[key].get('map', {})
            
    mapped_series = series.copy()
    if mapper:
        mapped_series = mapped_series.map(lambda x: mapper.get(x, mapper.get(float(x), x)) if pd.notna(x) and (x in mapper or (str(x).replace('.','',1).isdigit() and float(x) in mapper)) else x)
        
    counts = mapped_series.value_counts().sort_index()
    total = len(mapped_series.dropna())
    
    # Return mapped series for later OR calculation usage
    return counts, total, mapped_series

def get_stats_categorical_str(counts, total):
    res = []
    for cat, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        res.append(f"{_html.escape(str(cat))}: {count} ({pct:.1f}%)")
    return "<br>".join(res)

# --- Helper: Calculate OR & 95% CI from 2x2 table ---
def compute_or_ci(a, b, c, d):
    """
    Calculates OR and 95% CI from 2x2 contingency table cells.
    Applies Haldane-Anscombe correction if any cell is zero.
    
    Args:
        a, b, c, d: Cell counts from 2x2 table
        
    Returns:
        Formatted string "OR (lower-upper)" or "-" if invalid
    """
    try:
        # Haldane-Anscombe correction if ANY cell is zero
        if min(a, b, c, d) == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            
        or_val = (a * d) / (b * c)
        
        if or_val == 0 or np.isinf(or_val):
            return "-"

        # 95% CI (Natural Log Method)
        ln_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        lower = np.exp(ln_or - 1.96 * se)
        upper = np.exp(ln_or + 1.96 * se)
        
        return f"{or_val:.2f} ({lower:.2f}-{upper:.2f})"
    except Exception:
        return "-"

# --- 🟢 FIXED: Safe comparison helper for Bug #2 ---
def safe_group_compare(series, val):
    """
    Robust comparison handling float vs int vs string mismatches.
    Fixes: str(1.0) != str(1) issue.
    """
    # Try numeric comparison first if series is numeric
    if pd.api.types.is_numeric_dtype(series):
        try:
            val_float = float(val)
            return series == val_float
        except (ValueError, TypeError):
            pass # Fallback to string
            
    # Fallback to string comparison
    return series.astype(str) == str(val)

# --- 🟢 UPDATED: Calculate OR & 95% CI (One-vs-Rest) for Categorical [Legacy/Fallback] ---
def compute_or_for_row(row_series, cat_val, group_series, g1_val):
    try:
        # Complete-case mask (avoid treating NaN as "not cat" / "group0")
        mask = row_series.notna() & group_series.notna()
        row_series = row_series[mask]
        group_series = group_series[mask]
        
        # Safe string comparison for row categories
        row_bin = (row_series.astype(str) == str(cat_val))
        
        # 🟢 FIX BUG #2: Use safe numeric comparison
        group_bin = safe_group_compare(group_series, g1_val)
        
        a = (row_bin & group_bin).sum()
        b = (row_bin & ~group_bin).sum()
        c = (~row_bin & group_bin).sum()
        d = (~row_bin & ~group_bin).sum()
        
        return compute_or_ci(a, b, c, d)
    except Exception:
        return "-"

# --- 🟢 NEW: Calculate OR & 95% CI (Target vs Reference) for Categorical ---
def compute_or_vs_ref(row_series, cat_target, cat_ref, group_series, g1_val):
    """
    Calculates OR comparing 'cat_target' vs 'cat_ref'.
    Rows with other categories are ignored.
    """
    try:
        # Filter: Only rows that are Target OR Reference
        # Note: row_series and group_series must be aligned (same index)
        mask_target = (row_series.astype(str) == str(cat_target))
        mask_ref = (row_series.astype(str) == str(cat_ref))
        mask_valid = (mask_target | mask_ref) & group_series.notna()
        
        rs = row_series[mask_valid]
        gs = group_series[mask_valid]
        
        # Exposure: 1 if Target, 0 if Reference
        exposure = (rs.astype(str) == str(cat_target))
        
        # Outcome: 1 if Case (g1_val)
        # 🟢 FIX BUG #2: Use safe numeric comparison
        outcome = safe_group_compare(gs, g1_val)
        
        a = (exposure & outcome).sum()      # Exposed (Target) & Case
        b = (exposure & ~outcome).sum()     # Exposed (Target) & Control
        c = (~exposure & outcome).sum()     # Unexposed (Ref) & Case
        d = (~exposure & ~outcome).sum()    # Unexposed (Ref) & Control
        
        return compute_or_ci(a, b, c, d)
    except Exception:
        # Consider logging: logger.debug("OR calculation failed", exc_info=True)
        return "-"

# --- 🟢 UPDATED: Calculate OR & 95% CI for Continuous (Robust Logistic Regression) ---
def calculate_or_continuous_logit(df, feature_col, group_col, group1_val):
    """
    Calculates OR using Univariate Logistic Regression with Fallback Solvers.
    Includes protection against Perfect Separation.
    """
    try:
        # Prepare Data
        # Y = Target (Binary: 1=Group1, 0=Others)
        # 🟢 FIX BUG #2: Use safe numeric comparison
        y = safe_group_compare(df[group_col], group1_val).astype(int)
        
        # X = Feature (Continuous)
        X = df[feature_col].apply(clean_numeric).rename(feature_col)
        
        # Drop NaNs aligned
        mask = X.notna() & df[group_col].notna() # Check group_col na too
        y = y[mask]
        X = X[mask]
        
        # Minimum sample size check
        if len(y) < 10 or y.nunique() < 2 or X.nunique() < 2: 
            return "-" 
        
        # Add constant (intercept)
        X_const = sm.add_constant(X)
        
        # 🟢 Method 1: Try Standard Newton-Raphson
        try:
            model = sm.Logit(y, X_const)
            result = model.fit(disp=0)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            # 🟢 Method 2: Try BFGS (Better for some convergence issues)
            try:
                result = model.fit(method='bfgs', disp=0)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                return "-" # Failed all attempts

        # Extract OR and CI
        coef = result.params[feature_col]
        conf = result.conf_int().loc[feature_col]
        
        or_val = np.exp(coef)
        lower = np.exp(conf[0])
        upper = np.exp(conf[1])
        
        # 🟢 Safety Check: Filter insane values caused by Perfect Separation
        # e.g. OR > 1000 or OR < 0.001 usually means the model failed to converge properly
        if or_val > 1000 or or_val < 0.001:
            return "-"
            
        return f"{or_val:.2f} ({lower:.2f}-{upper:.2f})"
    except Exception:
        return "-"

# --- ⭐ NEW: Calculate SMD (Standardized Mean Difference) ---
def calculate_smd(df, col, group_col, g1_val, g2_val, is_cat, mapped_series=None, cats=None):
    """
    Calculate SMD for Table 1.
    - Continuous: Cohen's d (using pooled SD).
    - Categorical: Calculate SMD for each level (P1 vs P2) treating as binary.
    Returns: String (single value or <br> joined values).
    """
    try:
        # 1. Filter Data by Groups
        mask1 = safe_group_compare(df[group_col], g1_val)
        mask2 = safe_group_compare(df[group_col], g2_val)
        
        if is_cat:
            # Categorical: SMD per level
            if mapped_series is None or cats is None: return "-"
            
            s1 = mapped_series[mask1]
            s2 = mapped_series[mask2]
            n1, n2 = len(s1), len(s2)
            if n1 == 0 or n2 == 0: return "-"
            
            res_smd = []
            for cat in cats:
                # Calculate proportion for this category
                p1 = (s1.astype(str) == str(cat)).mean()
                p2 = (s2.astype(str) == str(cat)).mean()
                
                # SMD for binary (proportion)
                # Denominator: sqrt((p1(1-p1) + p2(1-p2))/2)
                var1 = p1 * (1 - p1)
                var2 = p2 * (1 - p2)
                pooled_sd = np.sqrt((var1 + var2) / 2)
                
                if pooled_sd == 0:
                    smd_val = 0.0
                else:
                    smd_val = abs(p1 - p2) / pooled_sd
                
                # Format
                val_str = f"{smd_val:.3f}"
                if smd_val >= 0.1: # Highlight Imbalanced
                    val_str = f"<b>{val_str}</b>"
                res_smd.append(val_str)
                
            return "<br>".join(res_smd)
            
        else:
            # Continuous: Standard Cohen's d
            v1 = df.loc[mask1, col].apply(clean_numeric).dropna()
            v2 = df.loc[mask2, col].apply(clean_numeric).dropna()
            
            if len(v1) == 0 or len(v2) == 0: return "-"
            
            m1, m2 = v1.mean(), v2.mean()
            s1, s2 = v1.std(), v2.std()
            
            if pd.isna(s1) or pd.isna(s2): return "-"
            
            pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
            if pooled_sd == 0: 
                smd_val = 0.0
            else:
                smd_val = abs(m1 - m2) / pooled_sd
                
            val_str = f"{smd_val:.3f}"
            if smd_val >= 0.1:
                val_str = f"<b>{val_str}</b>"
            return val_str

    except (ValueError, TypeError, ZeroDivisionError, KeyError):
        return "-"

# --- P-value Functions (Unchanged) ---
def calculate_p_continuous(data_groups):
    clean_groups = [g.apply(clean_numeric).dropna() for g in data_groups if len(g.apply(clean_numeric).dropna()) > 1]
    num_groups = len(clean_groups)
    if num_groups < 2: return np.nan, "-"
    all_normal = all(check_normality(g) for g in clean_groups)
    try:
        if all_normal:
            if num_groups == 2:
                _s, p = stats.ttest_ind(clean_groups[0], clean_groups[1], nan_policy='omit')
                test_name = "t-test"
            else:
                _s, p = stats.f_oneway(*clean_groups)
                test_name = "ANOVA"
        else:
            if num_groups == 2:
                _s, p = stats.mannwhitneyu(clean_groups[0], clean_groups[1], alternative='two-sided')
                test_name = "Mann-Whitney U"
            else:
                _s, p = stats.kruskal(*clean_groups)
                test_name = "Kruskal-Wallis"
        return p, test_name
    except Exception as e:
        return np.nan, f"Error: {e}"

def calculate_p_categorical(df, col, group_col):
    try:
        tab = pd.crosstab(df[col], df[group_col])
        if tab.size == 0: return np.nan, "-"
        is_2x2 = tab.shape == (2, 2)
        _chi2, p_chi2, _dof, ex = stats.chi2_contingency(tab)
        if is_2x2 and ex.min() < 5:
            try:
                _oddsr, p = stats.fisher_exact(tab)
                return p, "Fisher's Exact"
            except Exception:
                # Fisher's exact failed, fall back to Chi-square
                pass
        test_name = "Chi-square"
        if (ex < 5).any(): test_name = "Chi-square (Low N)"
        return p_chi2, test_name
    except Exception as e:
        return np.nan, f"Error: {e}"

# --- Main Generator ---
def generate_table(df, selected_vars, group_col, var_meta, or_style='all_levels'):
    """
    or_style: 'all_levels' (Default: Ref=1, others vs Ref) or 'simple' (One line per var)
    """
    # Get unified color palette
    COLORS = get_color_palette()
    
    if or_style not in ('all_levels', 'simple'):
        raise ValueError(f"or_style must be 'all_levels' or 'simple', got '{or_style}'")
    has_group = group_col is not None and group_col != "None"
    groups = []
    
    # Prepare Groups
    if has_group:
        mapper = {}
        if var_meta:
            key = group_col.split('_')[1] if '_' in group_col else group_col
            if group_col in var_meta: mapper = var_meta[group_col].get('map', {})
            elif key in var_meta: mapper = var_meta[key].get('map', {})
        raw_groups = df[group_col].dropna().unique().tolist()
        def _group_sort_key(v):
            s = str(v)
            try:
                return (0, float(s))
            except (ValueError, TypeError):
                return (1, s)
        raw_groups.sort(key=_group_sort_key)
        for g in raw_groups:
            label = mapper.get(g, mapper.get(float(g), str(g)) if str(g).replace('.','',1).isdigit() else str(g))
            groups.append({'val': g, 'label': str(label)})
    
    # Check if we can calculate OR (Must be exactly 2 groups)
    show_or = has_group and len(groups) == 2
    group_1_val = None
    group_2_val = None # Need for SMD
    
    if show_or:
        group_vals = [g["val"] for g in groups]
        # Auto-detect "Case" group (prefer 1 or higher value)
        # Usually g[1] is case if sorted 0,1
        group_1_val = 1 if 1 in group_vals else max(group_vals, key=_group_sort_key)
        # Identify the other group
        group_2_val = [g for g in group_vals if g != group_1_val][0]

    # CSS with unified teal colors
    css_style = f"""
    <style>
        body {{ 
            font-family: 'Segoe UI', sans-serif; 
            padding: 20px; 
            background-color: #f4f6f8; 
            margin: 0; 
            color: {COLORS['text']}; 
        }}
        .table-container {{ 
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
            padding: 20px; 
            width: 100%; 
            overflow-x: auto; 
            border: 1px solid #ddd; 
            box-sizing: border-box;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
        th {{ 
            background-color: {COLORS['primary_dark']}; 
            color: white; 
            padding: 12px 15px; 
            text-align: center; 
            border: 1px solid {COLORS['primary']}; 
        }}
        th:first-child {{ text-align: left; }}
        td {{ 
            padding: 10px 15px; 
            border: 1px solid #e0e0e0; 
            vertical-align: top; 
            color: {COLORS['text']}; 
        }}
        tr:nth-child(even) td {{ background-color: #f9f9f9; }}
        tr:hover td {{ background-color: #f1f7ff; }}
        .footer-note {{ 
            margin-top: 15px; 
            font-size: 0.85em; 
            color: {COLORS['text_secondary']}; 
            font-style: italic; 
        }}
        .report-footer {{ 
            text-align: right; 
            font-size: 0.75em; 
            color: {COLORS['text_secondary']}; 
            margin-top: 20px; 
            border-top: 1px dashed #ccc; 
            padding-top: 10px; 
        }}
        a {{ color: {COLORS['primary']}; text-decoration: none; }}
        a:hover {{ color: {COLORS['primary_dark']}; }}
    </style>
    """
    
    html = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html += "<div class='table-container'>"
    html += "<h2>Baseline Characteristics</h2>"
    
    # --- HEADER ---
    html += "<table><thead><tr>"
    html += "<th>Characteristic</th>"
    html += f"<th>Total (N={len(df)})</th>"
    if has_group:
        for g in groups:
            # 🟢 FIX BUG #2: Use safe numeric comparison
            n_g = len(df[safe_group_compare(df[group_col], g['val'])])
            html += f"<th>{_html.escape(str(g['label']))} (n={n_g})</th>"
    
    # OR Column Header
    if show_or:
        if or_style == 'simple':
            html += "<th>OR (95% CI)<br><span style='font-size:0.8em; font-weight:normal'>(Effect vs Ref)</span></th>"
        else:
            html += "<th>OR (95% CI)<br><span style='font-size:0.8em; font-weight:normal'>(All Levels vs Ref)</span></th>"
        
        # ⭐ ADD SMD HEADER
        html += "<th>SMD<br><span style='font-size:0.8em; font-weight:normal'>(>0.1 Imbalanced)</span></th>"
        
    html += "<th>P-value</th>"
    html += "<th>Test Used</th>"
    html += "</tr></thead><tbody>"
    
    # --- BODY ---
    for col in selected_vars:
        if col == group_col: continue
        
        # Meta & Labeling
        meta = {}
        key = col.split('_')[1] if '_' in col else col
        if var_meta:
            if col in var_meta: meta = var_meta[col]
            elif key in var_meta: meta = var_meta[key]
        label = meta.get('label', key)
        is_cat = meta.get('type') == 'Categorical'
        if not is_cat:
            if df[col].nunique() < 10 or df[col].dtype == object: is_cat = True 
        
        row_html = f"<tr><td><b>{_html.escape(str(label))}</b></td>"
        
        # --- DATA PREPARATION ---
        if is_cat:
            # Single source of truth for mapping: used for totals + per-group stats + OR
            counts_total, n_total, mapped_full_series = get_stats_categorical_data(df[col], var_meta, col) 
            val_total = get_stats_categorical_str(counts_total, n_total)
        else:
            val_total = get_stats_continuous(df[col])
            mapped_full_series = None
            
        row_html += f"<td style='text-align: center;'>{val_total}</td>"
        
        group_vals_list = []
        
        if has_group:
            for g in groups:
                # 🟢 FIX BUG #2: Use safe numeric comparison
                sub_df = df[safe_group_compare(df[group_col], g['val'])]
                
                # Get stats for this group
                if is_cat:
                    counts_g, n_g, _ = get_stats_categorical_data(sub_df[col], var_meta, col)
                    # Align with total counts
                    aligned_counts = {cat: counts_g.get(cat, 0) for cat in counts_total.index}
                    val_g = get_stats_categorical_str(aligned_counts, n_g)
                    row_html += f"<td style='text-align: center;'>{val_g}</td>"
                else:
                    val_g = get_stats_continuous(sub_df[col])
                    row_html += f"<td style='text-align: center;'>{val_g}</td>"
                    group_vals_list.append(sub_df[col])
            
            # 🟢 CALCULATE OR (If applicable)
            if show_or:
                or_cell_content = "-"
                if is_cat:
                    # Logic for OR Display Style
                    cats = counts_total.index.tolist()
                    if len(cats) >= 2:
                        ref_cat = cats[0] # Assume first sorted value is Reference
                        
                        if or_style == 'simple':
                            # 1. Simple Mode: One line
                            target_cat = cats[-1]
                            
                            or_res = compute_or_vs_ref(mapped_full_series, target_cat, ref_cat, df[group_col], group_1_val)
                            or_cell_content = f"{or_res}<br><span style='font-size:0.8em; color:#666'>({_html.escape(str(target_cat))} vs {_html.escape(str(ref_cat))})</span>"
                        
                        else:
                            # 2. All Levels Mode (Default): Show Ref line + All comparisons
                            lines = []
                            lines.append(f"{_html.escape(str(ref_cat))} (Ref): 1.00")
                            
                            for cat in cats[1:]:
                                or_res = compute_or_vs_ref(mapped_full_series, cat, ref_cat, df[group_col], group_1_val)
                                lines.append(f"{_html.escape(str(cat))}: {or_res}")
                                
                            or_cell_content = "<br>".join(lines)
                    else:
                        or_cell_content = "-" # Not enough categories
                else:
                    # Robust Continuous OR (Always one line per unit)
                    or_cell_content = calculate_or_continuous_logit(df, col, group_col, group_1_val)
                
                row_html += f"<td style='text-align: center; white-space: nowrap;'>{or_cell_content}</td>"

                # ⭐ CALCULATE SMD
                smd_cats = counts_total.index.tolist() if is_cat else None
                smd_val = calculate_smd(df, col, group_col, group_1_val, group_2_val, is_cat, mapped_full_series, smd_cats)
                row_html += f"<td style='text-align: center;'>{smd_val}</td>"

            # Calculate P-value
            if is_cat: 
                p_val, test_name = calculate_p_categorical(df, col, group_col)
            else: 
                p_val, test_name = calculate_p_continuous(group_vals_list)
            
            p_str = format_p(p_val)
            if isinstance(p_val, float) and p_val < 0.05:
                p_str = f"<span style='color:{COLORS['danger']}; font-weight:bold;'>{p_str}*</span>"
                
            row_html += f"<td style='text-align: center;'>{p_str}</td>"
            row_html += f"<td style='text-align: center; font-size: 0.8em; color: #666;'>{test_name}</td>"
        
        row_html += "</tr>"
        html += row_html
        
    html += "</tbody></table>"
    
    # Build footer with dynamic reference group label
    if show_or:
        # Find the label for the reference group
        ref_group_label = None
        for g in groups:
            if g['val'] == group_1_val:
                ref_group_label = g['label']
                break
        
        html += f"""<div class='footer-note'>
    <b>OR (Odds Ratio):</b> <br>
    - Categorical: { 'Simple (Single level vs Ref)' if or_style=='simple' else 'All Levels (Every level vs Ref)' }.<br>
    - Continuous: Univariate Logistic Regression (Odds change per 1 unit increase).<br>
    Reference group (exposed/case): <b>{_html.escape(str(ref_group_label))}</b> (value: {group_1_val}). Values are OR (95% CI).<br>
    <b>SMD (Standardized Mean Difference):</b> Value < 0.1 indicates good balance. Bold values indicate SMD >= 0.1.
    </div>"""
    else:
        html += """<div class='footer-note'>
    Data presented as Mean ± SD or n (%). P-values calculated using automated test selection.
    </div>"""
    
    html += "</div>"
    
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""

    return html
