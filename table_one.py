# table_one.py
import pandas as pd
import numpy as np
from scipy import stats
# 🟢 NEW: Import statsmodels for Logistic Regression (Continuous OR)
import statsmodels.api as sm 

def clean_numeric(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try: return float(s)
    except: return np.nan

# --- Helper: Normality Check ---
def check_normality(series):
    clean = series.dropna()
    if len(clean) < 3 or len(clean) > 5000 or clean.nunique() < 3: 
        return False
    try:
        stat, p_sw = stats.shapiro(clean)
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

# --- 🟢 UPDATED: Return list of cats for OR matching ---
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
    
    # Return list of (Label, Count, Pct, RawValue if possible)
    # Note: RawValue is tricky after mapping, but for OR we iterate the *labels* or *mapped series*
    # Strategy: Use the Mapped Series for One-vs-Rest Logic
    return counts, total, mapped_series

def get_stats_categorical_str(counts, total):
    res = []
    for cat, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        res.append(f"{cat}: {count} ({pct:.1f}%)")
    return "<br>".join(res)

# --- 🟢 NEW: Calculate OR & 95% CI (One-vs-Rest) for Categorical ---
def compute_or_for_row(row_series, cat_val, group_series, g1_val):
    try:
        # Complete-case mask (avoid treating NaN as "not cat" / "group0")
        mask = row_series.notna() & group_series.notna()
        row_series = row_series[mask]
        group_series = group_series[mask]
        # Construct 2x2 Table
        #            Group 1   Group 0
        # Cat Val       a         b
        # Not Cat       c         d
        
        # Cast to strings to ensure safe comparison
        row_bin = (row_series.astype(str) == str(cat_val))
        group_bin = (group_series == g1_val) # Group series is already raw/mapped values
        
        a = (row_bin & group_bin).sum()
        b = (row_bin & ~group_bin).sum()
        c = (~row_bin & group_bin).sum()
        d = (~row_bin & ~group_bin).sum()
        
        # Use Haldane-Anscombe correction if ANY cell is zero
        if min(a, b, c, d) == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            
        or_val = (a * d) / (b * c)
        
        # 95% CI (Natural Log Method)
        ln_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        lower = np.exp(ln_or - 1.96 * se)
        upper = np.exp(ln_or + 1.96 * se)
        
        return f"{or_val:.2f} ({lower:.2f}-{upper:.2f})"
    except Exception:
        return "-"

# --- 🟢 NEW: Calculate OR & 95% CI for Continuous (Logistic Regression) ---
def calculate_or_continuous_logit(df, feature_col, group_col, group1_val):
    """
    Calculates OR using Univariate Logistic Regression.
    Interpretation: OR per 1 unit increase in feature_col.
    """
    try:
        # Prepare Data
        # Y = Target (Binary: 1=Group1, 0=Others)
        y = (df[group_col] == group1_val).astype(int)
        
        # X = Feature (Continuous) - Use clean_numeric to handle strings
        X = df[feature_col].apply(clean_numeric)
        
        # Drop NaNs aligned
        mask = ~np.isnan(X) & ~np.isnan(y)
        y = y[mask]
        X = X[mask]
        
        if len(y) < 10 or y.nunique() < 2: return "-" # Not enough data
        
        # Logistic Regression using statsmodels
        # Add constant (intercept) manually as statsmodels doesn't add it by default
        X_const = sm.add_constant(X) 
        model = sm.Logit(y, X_const)
        result = model.fit(disp=0) # disp=0 to silence output
        
        # Extract OR and CI
        # params[1] is the coefficient for our variable (index 0 is constant)
        coef = result.params.iloc[1]
        conf = result.conf_int().iloc[1]
        
        or_val = np.exp(coef)
        lower = np.exp(conf[0])
        upper = np.exp(conf[1])
        
        return f"{or_val:.2f} ({lower:.2f}-{upper:.2f})"
    except Exception as e:
        # print(f"Logit Error: {e}") # Debug if needed
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
                s, p = stats.ttest_ind(clean_groups[0], clean_groups[1], nan_policy='omit')
                test_name = "t-test"
            else:
                s, p = stats.f_oneway(*clean_groups)
                test_name = "ANOVA"
        else:
            if num_groups == 2:
                s, p = stats.mannwhitneyu(clean_groups[0], clean_groups[1], alternative='two-sided')
                test_name = "Mann-Whitney U"
            else:
                s, p = stats.kruskal(*clean_groups)
                test_name = "Kruskal-Wallis"
        return p, test_name
    except: return np.nan, "Error"

def calculate_p_categorical(df, col, group_col):
    try:
        tab = pd.crosstab(df[col], df[group_col])
        if tab.size == 0: return np.nan, "-"
        is_2x2 = tab.shape == (2, 2)
        chi2, p_chi2, dof, ex = stats.chi2_contingency(tab)
        if is_2x2 and ex.min() < 5:
            oddsr, p = stats.fisher_exact(tab)
            return p, "Fisher's Exact"
        test_name = "Chi-square"
        if (ex < 5).any(): test_name = "Chi-square (Low N)"
        return p_chi2, test_name
    except: return np.nan, "Error"

# --- Main Generator ---
def generate_table(df, selected_vars, group_col, var_meta):
    has_group = group_col is not None and group_col != "None"
    groups = []
    
    # Prepare Groups
    if has_group:
        mapper = {}
        if var_meta:
            key = group_col.split('_')[1] if '_' in group_col else group_col
            if group_col in var_meta: mapper = var_meta[group_col].get('map', {})
            elif key in var_meta: mapper = var_meta[key].get('map', {})
        raw_groups = sorted(df[group_col].dropna().unique())
        for g in raw_groups:
            label = mapper.get(g, mapper.get(float(g), str(g)) if str(g).replace('.','',1).isdigit() else str(g))
            groups.append({'val': g, 'label': str(label)})
    
    # 🟢 Check if we can calculate OR (Must be exactly 2 groups)
    show_or = has_group and len(groups) == 2
    group_1_val = None
    if show_or:
        # Prefer 1 when present; otherwise fall back to the "higher" value
        group_vals = [g["val"] for g in groups]
        group_1_val = 1 if 1 in group_vals else sorted(group_vals)[-1]

    # CSS
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; margin: 0; color: #333; }
        .table-container { 
            background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
            padding: 20px; width: 100%; overflow-x: auto; border: 1px solid #ddd; box-sizing: border-box;
        }
        table { width: 100%; border-collapse: collapse; font-size: 0.95em; }
        th { background-color: #2c3e50; color: white; padding: 12px 15px; text-align: center; border: 1px solid #34495e; }
        th:first-child { text-align: left; }
        td { padding: 10px 15px; border: 1px solid #e0e0e0; vertical-align: top; color: #333; }
        tr:nth-child(even) td { background-color: #f9f9f9; }
        tr:hover td { background-color: #f1f7ff; }
        .footer-note { margin-top: 15px; font-size: 0.85em; color: #666; font-style: italic; }
        .report-footer { text-align: right; font-size: 0.75em; color: #666; margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 10px; }
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
            n_g = len(df[df[group_col] == g['val']])
            html += f"<th>{g['label']} (n={n_g})</th>"
    
    # 🟢 Add OR Column Header
    if show_or:
        html += f"<th>OR (95% CI)<br><span style='font-size:0.8em; font-weight:normal'>(vs Others / Per Unit)</span></th>"
        
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
        
        row_html = f"<tr><td><b>{label}</b></td>"
        
        # --- DATA PREPARATION ---
        # Need to handle mapping globally for the row first to ensure consistency across columns
        mapped_full_series = df[col].copy()
        col_mapper = meta.get('map', {})
        if col_mapper:
            _m = col_mapper
            mapped_full_series = mapped_full_series.map(
                lambda x, m=_m: m.get(x, m.get(float(x), x))
                if pd.notna(x) and (x in m or (str(x).replace(".", "", 1).isdigit() and float(x) in m))
                else x
            )

        if is_cat:
            counts_total, n_total, _ = get_stats_categorical_data(df[col], var_meta, col) # Pass raw for internal map
            val_total = get_stats_categorical_str(counts_total, n_total)
        else:
            val_total = get_stats_continuous(df[col])
            
        row_html += f"<td style='text-align: center;'>{val_total}</td>"
        
        group_vals_list = []
        or_strings = [] # Store OR strings for each category
        
        if has_group:
            for g in groups:
                sub_df = df[df[group_col] == g['val']]
                # Get stats for this group
                if is_cat:
                    # We need consistent index with Total
                    counts_g, n_g, _ = get_stats_categorical_data(sub_df[col], var_meta, col)
                    # Align with total counts index (categories present in total)
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
                    # Iterate through each category appearing in the dataset
                    cat_ors = []
                    for cat in counts_total.index:
                        # Calculate OR: (This Cat vs Others) x (Group 1 vs Group 0)
                        # We use mapped_full_series to ensure we match the 'cat' label
                        or_res = compute_or_for_row(mapped_full_series, cat, df[group_col], group_1_val)
                        cat_ors.append(f"{cat}: {or_res}")
                    or_cell_content = "<br>".join(cat_ors)
                else:
                    # 🟢 NEW: Continuous Variable OR Calculation
                    or_cell_content = calculate_or_continuous_logit(df, col, group_col, group_1_val)
                
                row_html += f"<td style='text-align: center; white-space: nowrap;'>{or_cell_content}</td>"

            # Calculate P-value
            if is_cat: 
                p_val, test_name = calculate_p_categorical(df, col, group_col)
            else: 
                p_val, test_name = calculate_p_continuous(group_vals_list)
            
            p_str = format_p(p_val)
            if isinstance(p_val, float) and p_val < 0.05:
                p_str = f"<span style='color:#d32f2f; font-weight:bold;'>{p_str}*</span>"
                
            row_html += f"<td style='text-align: center;'>{p_str}</td>"
            row_html += f"<td style='text-align: center; font-size: 0.8em; color: #666;'>{test_name}</td>"
        
        row_html += "</tr>"
        html += row_html
        
    html += "</tbody></table>"
    html += """<div class='footer-note'>
    <b>OR (Odds Ratio):</b> <br>
    - Categorical: One-vs-Rest method (Category X vs All Other Categories).<br>
    - Continuous: Univariate Logistic Regression (Odds change per 1 unit increase).<br>
    Reference group is the first column of the grouping variable. Values are OR (95% CI).
    </div>"""
    html += "</div>"
    
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""

    return html
