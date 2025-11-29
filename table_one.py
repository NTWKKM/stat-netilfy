import pandas as pd
import numpy as np
from scipy import stats

def clean_numeric(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try: return float(s)
    except: return np.nan

def format_p(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def get_stats_continuous(series):
    clean = series.apply(clean_numeric).dropna()
    if len(clean) == 0: return "-"
    return f"{clean.mean():.1f} ± {clean.std():.1f}"

def get_stats_categorical(series, var_meta=None, col_name=None):
    mapper = {}
    if var_meta and col_name:
        key = col_name.split('_')[1] if '_' in col_name else col_name
        if col_name in var_meta: mapper = var_meta[col_name].get('map', {})
        elif key in var_meta: mapper = var_meta[key].get('map', {})
            
    mapped_series = series.copy()
    if mapper:
        mapped_series = mapped_series.map(lambda x: mapper.get(x, mapper.get(float(x), x)) if pd.notna(x) and (x in mapper or float(x) in mapper if str(x).replace('.','',1).isdigit() else False) else x)
    
    counts = mapped_series.value_counts().sort_index()
    total = len(mapped_series.dropna())
    res = []
    for cat, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        res.append(f"{cat}: {count} ({pct:.1f}%)")
    return "<br>".join(res)

def calculate_p_continuous(data_groups):
    clean_groups = [g.dropna() for g in data_groups if len(g.dropna()) > 0]
    if len(clean_groups) < 2: return np.nan
    try:
        if len(clean_groups) == 2: s, p = stats.ttest_ind(clean_groups[0], clean_groups[1], nan_policy='omit')
        else: s, p = stats.f_oneway(*clean_groups)
        return p
    except: return np.nan

def calculate_p_categorical(df, col, group_col):
    try:
        tab = pd.crosstab(df[col], df[group_col])
        if tab.size == 0: return np.nan
        chi2, p, dof, ex = stats.chi2_contingency(tab)
        return p
    except: return np.nan

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
    
    # --- 🟢 CSS Styling (เหมือน logic.py) ---
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
            color: #333; /* บังคับสีดำเสมอ */
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
    
    html += "<table><thead><tr>"
    html += "<th>Characteristic</th>"
    html += f"<th>Total (N={len(df)})</th>"
    
    if has_group:
        for g in groups:
            n_g = len(df[df[group_col] == g['val']])
            html += f"<th>{g['label']} (n={n_g})</th>"
        html += "<th>P-value</th>"
    html += "</tr></thead><tbody>"
    
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
            if n_unique < 10 or df[col].dtype == object: is_cat = True
        
        row_html = f"<tr><td><b>{label}</b></td>"
        if is_cat:
            val_total = get_stats_categorical(df[col], var_meta, col)
            row_html += f"<td style='text-align: center;'>{val_total}</td>"
        else:
            val_total = get_stats_continuous(df[col])
            row_html += f"<td style='text-align: center;'>{val_total}</td>"
            
        p_val = np.nan
        if has_group:
            group_vals_list = []
            for g in groups:
                sub_df = df[df[group_col] == g['val']]
                if is_cat:
                    val_g = get_stats_categorical(sub_df[col], var_meta, col)
                    row_html += f"<td style='text-align: center;'>{val_g}</td>"
                else:
                    val_g = get_stats_continuous(sub_df[col])
                    row_html += f"<td style='text-align: center;'>{val_g}</td>"
                    group_vals_list.append(sub_df[col].apply(clean_numeric))
            if is_cat: p_val = calculate_p_categorical(df, col, group_col)
            else: p_val = calculate_p_continuous(group_vals_list)
            
            p_str = format_p(p_val)
            # Highlight P < 0.05
            if isinstance(p_val, float) and p_val < 0.05:
                p_str = f"<span style='color:#d32f2f; font-weight:bold;'>{p_str}*</span>"
                
            row_html += f"<td style='text-align: center;'>{p_str}</td>"
        row_html += "</tr>"
        html += row_html
        
    html += "</tbody></table>"
    html += "<div class='footer-note'>Data presented as Mean ± SD for continuous variables, and n (%) for categorical variables. P-values calculated using T-test/ANOVA or Chi-square/Fisher's exact test.</div>"
    html += "</div></body></html>"
    
# 🟢 NEW: เพิ่ม Footer ของ Report
    html += """
    <div class="report-footer">
      &copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit
    </div>
    """

    return html
