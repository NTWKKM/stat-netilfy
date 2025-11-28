import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings

# ปิด Warning
warnings.filterwarnings("ignore")

# ==========================================
# 1. Helper Functions
# ==========================================

def clean_numeric_value(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except:
        return np.nan

def run_binary_logit(y, X, method='default'):
    try:
        X_const = sm.add_constant(X, has_constant='add')
        if method == 'bfgs':
            model = sm.Logit(y, X_const).fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const).fit(disp=0)
        return model.params, model.conf_int(), model.pvalues, "OK"
    except Exception as e:
        return None, None, None, str(e)

def get_label(col_name, var_meta):
    # รองรับทั้งชื่อที่มี Prefix (Demographic_age) และไม่มี (age)
    parts = col_name.split('_', 1)
    orig_name = parts[1] if len(parts) > 1 else col_name
    
    label = orig_name
    if var_meta:
        meta = var_meta.get(orig_name, {})
        label = meta.get('label', orig_name)
        
    return f"<b>{orig_name}</b><br><span style='color:#666; font-size:0.9em'>{label}</span>"

def analyze_outcome(outcome_name, df, var_meta):
    # เช็คว่ามีคอลัมน์นี้ไหม
    if outcome_name not in df.columns:
        return f"<div class='alert'>⚠️ Error: Outcome variable '{outcome_name}' not found.</div>"
    
    # เตรียมข้อมูล Outcome
    y = df[outcome_name].dropna()
    # แปลงเป็น int (0/1) ให้ชัวร์
    try:
        y = y.astype(int)
    except:
        return f"<div class='alert'>⚠️ Error: Outcome '{outcome_name}' must be numeric (0/1).</div>"

    df_aligned = df.loc[y.index]
    total_n = len(y)
    
    if y.nunique() < 2:
        return f"<div class='alert'>⚠️ Outcome '{outcome_name}' needs at least 2 groups (0 and 1). Found: {y.unique()}</div>"

    candidates = [] 
    results_db = {} 
    # เรียงคอลัมน์
    sorted_cols = sorted(df.columns)

    # --- UNIVARIATE LOOP ---
    for col in sorted_cols:
        if col == outcome_name: continue
        
        # ข้ามถ้าข้อมูลว่างหมด
        if df_aligned[col].isnull().all(): continue

        res = {'var': col}
        X_raw = df_aligned[col]
        # พยายามแปลงเป็นตัวเลข
        X_num = X_raw.apply(clean_numeric_value)
        
        X_neg = X_raw[y == 0]
        X_pos = X_raw[y == 1]
        
        # เดาชื่อเดิม (เผื่อมี prefix)
        orig_name = col.split('_', 1)[1] if len(col.split('_', 1)) > 1 else col
        
        unique_vals = X_num.dropna().unique()
        unique_count = len(unique_vals)
        
        # Logic แยก Categorical/Continuous
        # ถ้ามีค่าน้อยกว่า 5 แบบ หรือเป็น 0/1 -> มองเป็น Categorical
        is_binary_pred = set(unique_vals).issubset({0, 1})
        
        if is_binary_pred or unique_count < 5:
            # === CATEGORICAL ===
            n_used = len(X_raw.dropna())
            
            # เรียง Level ให้สวยงาม
            try: levels = sorted(X_raw.dropna().unique(), key=lambda x: float(x))
            except: levels = sorted(X_raw.astype(str).unique())
            
            desc_tot = [f"<span class='n-badge'>n={n_used}</span>"]
            desc_neg = [f"<span class='n-badge'>n={len(X_neg.dropna())}</span>"]
            desc_pos = [f"<span class='n-badge'>n={len(X_pos.dropna())}</span>"]
            
            for lvl in levels:
                lvl_str = str(lvl)
                # Count
                c_all = (X_raw.astype(str) == lvl_str).sum()
                p_all = (c_all/n_used)*100 if n_used else 0
                
                c_n = (X_neg.astype(str) == lvl_str).sum()
                p_n = (c_n/len(X_neg.dropna()))*100 if len(X_neg.dropna()) else 0
                
                c_p = (X_pos.astype(str) == lvl_str).sum()
                p_p = (c_p/len(X_pos.dropna()))*100 if len(X_pos.dropna()) else 0
                
                desc_tot.append(f"{lvl}: {c_all} ({p_all:.1f}%)")
                desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                desc_pos.append(f"{c_p} ({p_p:.1f}%)")
            
            res['desc_total'] = "<br>".join(desc_tot)
            res['desc_neg'] = "<br>".join(desc_neg)
            res['desc_pos'] = "<br>".join(desc_pos)
            
            # Chi-square
            try:
                contingency = pd.crosstab(X_raw, y)
                if contingency.size > 0:
                    chi2, p, dof, ex = stats.chi2_contingency(contingency)
                    res['p_comp'] = p
            except: res['p_comp'] = np.nan
            
        else:
            # === CONTINUOUS ===
            n_used = len(X_num.dropna())
            m_t, s_t = X_num.mean(), X_num.std()
            
            # แปลงเป็น numeric ให้ชัวร์สำหรับกลุ่มย่อย
            xn_clean = pd.to_numeric(X_neg, errors='coerce').dropna()
            xp_clean = pd.to_numeric(X_pos, errors='coerce').dropna()
            
            m_n, s_n = xn_clean.mean(), xn_clean.std()
            m_p, s_p = xp_clean.mean(), xp_clean.std()
            
            res['desc_total'] = f"<span class='n-badge'>n={n_used}</span><br>Mean: {m_t:.2f}<br>(SD {s_t:.2f})"
            res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})" if not pd.isna(m_n) else "-"
            res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})" if not pd.isna(m_p) else "-"
            
            # Mann-Whitney
            try:
                if len(xn_clean) > 0 and len(xp_clean) > 0:
                    u, p = stats.mannwhitneyu(xn_clean, xp_clean)
                    res['p_comp'] = p
            except: res['p_comp'] = np.nan

        # --- UNIVARIATE REGRESSION ---
        data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
        if not data_uni.empty and data_uni['x'].nunique() > 1:
            params, conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']])
            if status == "OK" and 'x' in params:
                coef = params['x']
                or_val = np.exp(coef)
                ci_low, ci_high = np.exp(conf.loc['x'][0]), np.exp(conf.loc['x'][1])
                res['or'] = f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"
                res['p_or'] = pvals['x']
            else: res['or'] = "-"
        else: res['or'] = "-"

        results_db[col] = res
        
        # Screen P < 0.20
        p_screen = res.get('p_comp', np.nan)
        if pd.isna(p_screen): p_screen = res.get('p_or', np.nan)
        if pd.notna(p_screen) and p_screen < 0.20:
            candidates.append(col)

    # --- MULTIVARIATE ---
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
            params, conf, pvals, status = run_binary_logit(multi_data['y'], multi_data[cand_valid], method='bfgs')
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
    for col in sorted_cols:
        if col == outcome_name or col not in results_db: continue
        res = results_db[col]
        
        lbl = get_label(col, var_meta)
        or_s = res.get('or', '-')
        
        def fmt_p(val):
            if pd.isna(val): return "-"
            if val < 0.001: return "<0.001"
            return f"{val:.3f}"

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
            <td>{p_s}</td>
            <td>{aor_s}</td>
            <td>{ap_s}</td>
        </tr>"""
        html_rows.append(row_html)
    
    return f"""
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
                <th>P-value</th>
                <th>aOR (95% CI)<br><span style='font-size:0.8em; font-weight:normal'>(Complete Case n={final_n_multi})</span></th>
                <th>aP-value</th>
            </tr>
        </thead>
        <tbody>{"".join(html_rows)}</tbody>
    </table>
    <div class='summary-box'>
        <b>Multivariate:</b> Included {len(cand_valid)} variables (p<0.20). Analysis used Complete Cases.<br>
    </div>
    </div><br>
    """

# --- Main Entry Point ---
def process_data_and_generate_html(df, target_outcome=None, var_meta=None):
    # CSS Styles (เหมือนเดิม ใส่เพื่อให้ตารางสวย)
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; }
        .table-container { background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); overflow-x: auto; }
        table { width: 100%; border-collapse: separate; border-spacing: 0; text-align: left; min-width: 800px; }
        th { background-color: #34495e; color: #fff; padding: 12px; position: sticky; top: 0; }
        td { padding: 12px; border-bottom: 1px solid #eee; vertical-align: top; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .outcome-title { background-color: #2c3e50; color: white; padding: 15px; font-weight: bold; border-radius: 8px 8px 0 0; }
        .sig-p { color: #d32f2f; font-weight: bold; background-color: #ffebee; padding: 2px 4px; border-radius: 4px; }
        .n-badge { font-size: 0.75em; color: #888; background: #eee; padding: 1px 4px; border-radius: 3px; }
        .summary-box { padding: 15px; background: #fff; font-size: 0.9em; color: #555; }
    </style>
    """
    
    html_content = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html_content += "<h1>Analysis Report</h1>"
    
    # ถ้าไม่ได้ระบุ outcome ให้หาอันแรกที่ดูใช่
    if not target_outcome:
        cols = [c for c in df.columns if 'outcome' in c.lower() or 'died' in c.lower()]
        target_outcome = cols[0] if cols else df.columns[-1]

    # รันวิเคราะห์แค่ Outcome ที่เลือก
    html_content += analyze_outcome(target_outcome, df, var_meta)
    
    html_content += "</body></html>"
    return html_content
