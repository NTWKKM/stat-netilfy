# table_one.py
import pandas as pd
import numpy as np
from scipy import stats

def clean_numeric(val):
    """แปลงค่าเป็นตัวเลข (เหมือน logic.py)"""
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except:
        return np.nan

def format_p(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def get_stats_continuous(series):
    """คืนค่า Mean ± SD"""
    clean = series.apply(clean_numeric).dropna()
    if len(clean) == 0: return "-"
    return f"{clean.mean():.1f} ± {clean.std():.1f}"

def get_stats_categorical(series, var_meta=None, col_name=None):
    """คืนค่า n (%) ของแต่ละกลุ่ม"""
    # Map ค่า (ถ้ามีใน meta)
    mapper = {}
    if var_meta and col_name:
        # หา meta ที่ตรงกับชื่อ col (ตัด prefix ถ้ามี)
        key = col_name.split('_')[1] if '_' in col_name else col_name
        # ลองหาทั้งชื่อเต็มและชื่อย่อ
        if col_name in var_meta: mapper = var_meta[col_name].get('map', {})
        elif key in var_meta: mapper = var_meta[key].get('map', {})
            
    # แปลงค่าตาม Map
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
    """ANOVA หรือ T-test"""
    # data_groups = list of arrays (clean numeric)
    clean_groups = [g.dropna() for g in data_groups if len(g.dropna()) > 0]
    
    if len(clean_groups) < 2: return np.nan
    
    try:
        if len(clean_groups) == 2:
            s, p = stats.ttest_ind(clean_groups[0], clean_groups[1], nan_policy='omit')
        else:
            s, p = stats.f_oneway(*clean_groups)
        return p
    except:
        return np.nan

def calculate_p_categorical(df, col, group_col):
    """Chi-square"""
    try:
        tab = pd.crosstab(df[col], df[group_col])
        if tab.size == 0: return np.nan
        chi2, p, dof, ex = stats.chi2_contingency(tab)
        return p
    except:
        return np.nan

def generate_table(df, selected_vars, group_col, var_meta):
    """สร้าง HTML Table 1"""
    
    # เตรียมข้อมูล Group
    has_group = group_col is not None and group_col != "None"
    groups = []
    if has_group:
        # แปลง Group เป็น String/Label ให้สวยงาม
        mapper = {}
        if var_meta:
            key = group_col.split('_')[1] if '_' in group_col else group_col
            if group_col in var_meta: mapper = var_meta[group_col].get('map', {})
            elif key in var_meta: mapper = var_meta[key].get('map', {})
        
        raw_groups = sorted(df[group_col].dropna().unique())
        for g in raw_groups:
            label = mapper.get(g, mapper.get(float(g), str(g)) if str(g).replace('.','',1).isdigit() else str(g))
            groups.append({'val': g, 'label': str(label)})
    
    # เริ่มสร้าง HTML
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif;">
        <thead>
            <tr style="background-color: #2c3e50; color: white;">
                <th style="padding: 10px; text-align: left;">Characteristic</th>
                <th style="padding: 10px; text-align: center;">Total (N={})</th>
    """.format(len(df))
    
    if has_group:
        for g in groups:
            n_g = len(df[df[group_col] == g['val']])
            html += f'<th style="padding: 10px; text-align: center;">{g["label"]} (n={n_g})</th>'
        html += '<th style="padding: 10px; text-align: center;">P-value</th>'
    
    html += "</tr></thead><tbody>"
    
    # Loop ตัวแปร
    for col in selected_vars:
        if col == group_col: continue
        
        # Determine Type (ใช้ logic เดิม หรือ meta)
        meta = {}
        key = col.split('_')[1] if '_' in col else col
        if var_meta:
            if col in var_meta: meta = var_meta[col]
            elif key in var_meta: meta = var_meta[key]
            
        label = meta.get('label', key)
        is_cat = meta.get('type') == 'Categorical'
        
        # Auto-detect fallback
        clean_vals = df[col].apply(clean_numeric)
        if not is_cat:
            n_unique = df[col].nunique()
            if n_unique < 10 or df[col].dtype == object:
                is_cat = True
        
        # Row Content
        row_html = f"<tr style='border-bottom: 1px solid #eee;'><td style='padding: 8px;'><b>{label}</b></td>"
        
        # 1. Total Column
        if is_cat:
            val_total = get_stats_categorical(df[col], var_meta, col)
            row_html += f"<td style='padding: 8px; text-align: center;'>{val_total}</td>"
        else:
            val_total = get_stats_continuous(df[col])
            row_html += f"<td style='padding: 8px; text-align: center;'>{val_total}</td>"
            
        # 2. Group Columns & P-value
        p_val = np.nan
        if has_group:
            group_vals_list = [] # เก็บค่าแยกกลุ่มเพื่อคำนวณ P
            
            for g in groups:
                sub_df = df[df[group_col] == g['val']]
                
                if is_cat:
                    val_g = get_stats_categorical(sub_df[col], var_meta, col)
                    row_html += f"<td style='padding: 8px; text-align: center;'>{val_g}</td>"
                else:
                    val_g = get_stats_continuous(sub_df[col])
                    row_html += f"<td style='padding: 8px; text-align: center;'>{val_g}</td>"
                    # เตรียมข้อมูล Continuous สำหรับ P-test
                    group_vals_list.append(sub_df[col].apply(clean_numeric))
            
            # Calculate P
            if is_cat:
                p_val = calculate_p_categorical(df, col, group_col)
            else:
                p_val = calculate_p_continuous(group_vals_list)
                
            row_html += f"<td style='padding: 8px; text-align: center;'><b>{format_p(p_val)}</b></td>"
            
        row_html += "</tr>"
        html += row_html
        
    html += "</tbody></table>"
    html += "<div style='margin-top:10px; font-size:0.9em; color:#666;'>Data presented as Mean ± SD for continuous variables, and n (%) for categorical variables.</div>"
    
    return html
