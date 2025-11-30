import pandas as pd
import numpy as np
from scipy import stats

def clean_numeric(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try: return float(s)
    except: return np.nan

# --- 🟢 NEW HELPER: Check for Normality (Shapiro-Wilk Test) ---
def check_normality(series):
    """Checks if the data is normally distributed (p-value > 0.05)."""
    clean = series.dropna()
    # Shapiro-Wilk needs at least 3 unique values
    if len(clean) < 3 or len(clean) > 5000 or clean.nunique() < 3: 
        return False # Assume non-normal or too few samples for reliable test
    
    try:
        stat, p_sw = stats.shapiro(clean)
        return p_sw > 0.05 # Returns True if Normal (p > 0.05)
    except Exception:
        return False

def format_p(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def get_stats_continuous(series):
    clean = series.apply(clean_numeric).dropna()
    if len(clean) == 0: return "-"
    # Simplification: only display Mean +/- SD
    return f"{clean.mean():.1f} \u00B1 {clean.std():.1f}"

def get_stats_categorical(series, var_meta=None, col_name=None):
    mapper = {}
    if var_meta and col_name:
        key = col_name.split('_')[1] if '_' in col_name else col_name
        if col_name in var_meta: mapper = var_meta[col_name].get('map', {})
        elif key in var_meta: mapper = var_meta[key].get('map', {})
            
    mapped_series = series.copy()
    if mapper:
        # Improved mapping robustness
        mapped_series = mapped_series.map(lambda x: mapper.get(x, mapper.get(float(x), x)) if pd.notna(x) and (x in mapper or (str(x).replace('.','',1).isdigit() and float(x) in mapper)) else x)
        
    counts = mapped_series.value_counts().sort_index()
    total = len(mapped_series.dropna())
    res = []
    for cat, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        res.append(f"{cat}: {count} ({pct:.1f}%)")
    return "<br>".join(res)

# --- 🟢 NEW/UPDATED: Calculate P-value for Continuous (with Normality check) ---
def calculate_p_continuous(data_groups):
    # data_groups is a list of series (one for each group)
    clean_groups = [g.apply(clean_numeric).dropna() for g in data_groups if len(g.apply(clean_numeric).dropna()) > 1]
    num_groups = len(clean_groups)
    
    if num_groups < 2: return np.nan, "-"

    # Check for overall normality across all available groups
    all_normal = all(check_normality(g) for g in clean_groups)

    try:
        if all_normal:
            # Parametric Tests (t-test / ANOVA)
            if num_groups == 2:
                # Independent Samples t-test
                s, p = stats.ttest_ind(clean_groups[0], clean_groups[1], nan_policy='omit')
                test_name = "t-test"
            else:
                # ANOVA (F-test)
                s, p = stats.f_oneway(*clean_groups)
                test_name = "ANOVA"
        else:
            # Non-Parametric Tests (Mann-Whitney U / Kruskal-Wallis)
            if num_groups == 2:
                # Mann-Whitney U test (Non-parametric for 2 groups)
                s, p = stats.mannwhitneyu(clean_groups[0], clean_groups[1], alternative='two-sided')
                test_name = "Mann-Whitney U"
            else:
                # Kruskal-Wallis H-test (Non-parametric for >2 groups)
                s, p = stats.kruskal(*clean_groups)
                test_name = "Kruskal-Wallis"
        
        return p, test_name

    except Exception: 
        return np.nan, "Error"

# --- 🟢 NEW/UPDATED: Calculate P-value for Categorical (with Fisher's check) ---
def calculate_p_categorical(df, col, group_col):
    try:
        tab = pd.crosstab(df[col], df[group_col])
        if tab.size == 0: return np.nan, "-"
        
        is_2x2 = tab.shape == (2, 2)
        
        # Calculate Chi2 and Expected Counts for the check
        chi2, p_chi2, dof, ex = stats.chi2_contingency(tab)

        # Rule: Use Fisher's Exact if 2x2 AND Expected Count is low (min expected < 5)
        if is_2x2 and ex.min() < 5:
            # Fisher's Exact Test
            oddsr, p = stats.fisher_exact(tab)
            test_name = "Fisher's Exact"
            return p, test_name
        
        # Otherwise (larger than 2x2 or 2x2 with adequate N), use Chi-square
        p = p_chi2
        test_name = "Chi-square"

        # Add warning for low cell count in Chi2 (> 2x2)
        if (ex < 5).any():
             test_name = "Chi-square (Low N)" 

        return p, test_name

    except Exception: 
        return np.nan, "Error"

def generate_table(df, selected_vars, group_col, var_meta):
    has_group = group_col is not None and group_col != "None"
    groups = []
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
        
    # --- CSS Styling (unchanged) ---
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; margin: 0; color: #333; }
        .table-container { 
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
            padding: 20px;
            width: 100%; 
            overflow-x: auto; 
            border: 1px solid #ddd;
            box-sizing: border-box;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: 'Segoe UI', sans-serif; 
            font-size: 0.95em;
        }
        th {
            background-color: #2c3e50; 
            color: white; 
            padding: 12px 15px; 
            text-align: center;
            font-weight: 600;
            border: 1px solid #34495e;
        }
        th:first-child { text-align: left; }
        td { 
            padding: 10px 15px; 
            border: 1px solid #e0e0e0;
            vertical-align: top;
            color: #333; 
        }
        tr:nth-child(even) td { background-color: #f9f9f9; }
        tr:hover td { background-color: #f1f7ff; }
        .footer-note { margin-top: 15px; font-size: 0.85em; color: #666; font-style: italic; }
        .report-footer {
            text-align: right;
            font-size: 0.75em;
            color: var(--text-color);
            margin-top: 20px;
            border-top: 1px dashed var(--border-color);
            padding-top: 10px;
        }
    </style>
    """
    
    html = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html += "<div class='table-container'>"
    html += "<h2>Baseline Characteristics</h2>"
    
    # --- 🟢 HEADER GENERATION (Added 'Test Used') ---
    html += "<table><thead><tr>"
    html += "<th>Characteristic</th>"
    html += f"<th>Total (N={len(df)})</th>"
    
    if has_group:
        for g in groups:
            n_g = len(df[df[group_col] == g['val']])
            html += f"<th>{g['label']} (n={n_g})</th>"
        html += "<th>P-value</th>"
        html += "<th>Test Used</th>" # <--- NEW COLUMN HEADER
    html += "</tr></thead><tbody>"
    
    # --- 🟢 BODY GENERATION (Updated Logic) ---
    for col in selected_vars:
        if col == group_col: continue
        meta = {}
        key = col.split('_')[1] if '_' in col else col
        if var_meta:
            if col in var_meta: meta = var_meta[col]
            elif key in var_meta: meta = var_meta[key]
        label = meta.get('label', key)
        is_cat = meta.get('type') == 'Categorical'
        if not is_cat:
            n_unique = df[col].nunique()
            # Heuristic check for categorization (if metadata missing)
            if n_unique < 10 or df[col].dtype == object: is_cat = True 
        
        row_html = f"<tr><td><b>{label}</b></td>"
        
        # Total Column
        if is_cat:
            val_total = get_stats_categorical(df[col], var_meta, col)
            row_html += f"<td style='text-align: center;'>{val_total}</td>"
        else:
            val_total = get_stats_continuous(df[col])
            row_html += f"<td style='text-align: center;'>{val_total}</td>"
            
        p_val = np.nan
        test_name = "-"

        if has_group:
            group_vals_list = []
            
            # Group Columns & Collect Data for P-value Calculation
            for g in groups:
                sub_df = df[df[group_col] == g['val']]
                if is_cat:
                    val_g = get_stats_categorical(sub_df[col], var_meta, col)
                    row_html += f"<td style='text-align: center;'>{val_g}</td>"
                else:
                    val_g = get_stats_continuous(sub_df[col])
                    row_html += f"<td style='text-align: center;'>{val_g}</td>"
                    group_vals_list.append(sub_df[col])

            # Calculate P-value and Test Name
            if is_cat: 
                p_val, test_name = calculate_p_categorical(df, col, group_col)
            else: 
                p_val, test_name = calculate_p_continuous(group_vals_list)
            
            # Format P-value
            p_str = format_p(p_val)
            
            # Highlight P < 0.05
            if isinstance(p_val, float) and p_val < 0.05:
                p_str = f"<span style='color:#d32f2f; font-weight:bold;'>{p_str}*</span>"
                
            # Add P-value and Test Used columns
            row_html += f"<td style='text-align: center;'>{p_str}</td>"
            row_html += f"<td style='text-align: center; font-size: 0.8em; color: #666;'>{test_name}</td>" # <--- NEW CELL
        
        row_html += "</tr>"
        html += row_html
        
    html += "</tbody></table>"
    html += "<div class='footer-note'>Data presented as Mean \u00B1 SD (Numeric, assuming normal) or n (%). P-values calculated using automated selection: t-test/ANOVA (Normal), Mann-Whitney U/Kruskal-Wallis (Non-normal), or Chi-square/Fisher's Exact (Categorical).</div>"
    html += "</div></body></html>"
    
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""

    return html
