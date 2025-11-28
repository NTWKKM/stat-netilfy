import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings

# ปิด Warning
warnings.filterwarnings("ignore")

# ==========================================
# 1. Helper Functions (เครื่องมือช่วยคำนวณ)
# ==========================================

def clean_numeric_value(val):
    """ฟังก์ชันล้างข้อมูล: ตัด > < , ออกแล้วแปลงเป็นตัวเลข"""
    if pd.isna(val): return np.nan
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except:
        return np.nan

def run_binary_logit(y, X, method='default'):
    """ฟังก์ชันคำนวณ Binary Logistic Regression"""
    try:
        # Add constant (Intercept)
        X_const = sm.add_constant(X, has_constant='add')
        
        # ใช้ BFGS เพื่อความทนทานในการหาคำตอบ (Robust Optimization)
        if method == 'bfgs':
            model = sm.Logit(y, X_const).fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const).fit(disp=0)
            
        return model.params, model.conf_int(), model.pvalues, "OK"
    except Exception as e:
        return None, None, None, str(e)

def get_label(col_name, var_meta):
    """ดึง Label จาก Metadata (ถ้ามี)"""
    parts = col_name.split('_', 1)
    orig_name = parts[1] if len(parts) > 1 else col_name
    
    label = orig_name
    if var_meta:
        meta = var_meta.get(orig_name, {})
        label = meta.get('label', orig_name)
        
    return f"<b>{orig_name}</b><br><span style='color:#666; font-size:0.9em'>{label}</span>"

def analyze_outcome(outcome_name, df, var_meta):
    """ฟังก์ชันวิเคราะห์สถิติสำหรับ 1 Outcome"""
    if outcome_name not in df.columns:
        return f"<div class='alert'>⚠️ Outcome variable '{outcome_name}' not found in data.</div>"
    
    # Prepare Outcome (Drop NaN in Y)
    y = df[outcome_name].dropna()
    df_aligned = df.loc[y.index]
    y = y.astype(int)
    total_n = len(y)
    
    # ตรวจสอบว่า Outcome มี 2 กลุ่มจริงหรือไม่ (0, 1)
    if y.nunique() < 2:
        return f"<div class='alert'>⚠️ Outcome '{outcome_name}' has less than 2 groups. Cannot analyze.</div>"

    candidates = [] 
    results_db = {} 
    # เรียงลำดับตัวแปรตามชื่อ Sheet (Prefix)
    sorted_cols = sorted(df.columns, key=lambda x: x.split('_')[0])

    # --- UNIVARIATE ANALYSIS LOOP ---
    for col in sorted_cols:
        if col == outcome_name: continue
        if 'Outcomes_' in col and col != outcome_name: pass 
        
        # ข้ามถ้าข้อมูลว่างทั้งหมด
        if df_aligned[col].isnull().all(): continue

        res = {'var': col}
        X_raw = df_aligned[col]
        X_num = X_raw.apply(clean_numeric_value)
        
        X_neg = X_raw[y == 0]
        X_pos = X_raw[y == 1]
        
        orig_name = col.split('_', 1)[1] if len(col.split('_', 1)) > 1 else col
        unique_vals = X_num.dropna().unique()
        unique_count = len(unique_vals)
        
        # Check type
        is_binary_pred = set(unique_vals).issubset({0, 1})
        is_in_codebook = False
        if var_meta:
            is_in_codebook = (orig_name in var_meta) and not (orig_name in ['sbp', 'dbp']) 
        
        # -----------------------------------
        # 1. Descriptive & Comparative Stats
        # -----------------------------------
        if is_binary_pred or (is_in_codebook and unique_count < 10) or unique_count < 5:
            # === CATEGORICAL DATA ===
            n_used = len(X_raw.dropna())
            mapper = {}
            if var_meta:
                mapper = var_meta.get(orig_name, {}).get('map', {})
                
            try: levels = sorted(X_raw.dropna().unique(), key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))
            except: levels = sorted(X_raw.astype(str).unique())
            
            # Badge N
            desc_tot = [f"<span class='n-badge'>n={n_used}</span>"]
            desc_neg = [f"<span class='n-badge'>n={len(X_neg.dropna())}</span>"]
            desc_pos = [f"<span class='n-badge'>n={len(X_pos.dropna())}</span>"]
            
            for lvl in levels:
                try: key = float(lvl); key = int(key) if key.is_integer() else key
                except: key = lvl
                label_txt = mapper.get(key, str(lvl))
                lvl_str = str(lvl)
                if str(lvl).endswith('.0'): lvl_str = str(int(float(lvl)))
                
                # Count Logic
                def count_val(series, val_str):
                    return (series.astype(str).apply(lambda x: x.replace('.0','') if x.replace('.','',1).isdigit() else x) == val_str).sum()

                c_all = count_val(X_raw, lvl_str)
                # Fallback for strict match
                if c_all == 0: c_all = (X_raw == lvl).sum()
                
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
            
            # Chi-square / Fisher
            try:
                contingency = pd.crosstab(X_raw, y)
                if contingency.size > 0:
                    chi2, p, dof, ex = stats.chi2_contingency(contingency)
                    res['p_comp'] = p
                else: res['p_comp'] = np.nan
            except: res['p_comp'] = np.nan
            
        else:
            # === CONTINUOUS DATA ===
            n_used = len(X_num.dropna())
            m_t, s_t = X_num.mean(), X_num.std()
            m_n, s_n = pd.to_numeric(X_neg, errors='coerce').mean(), pd.to_numeric(X_neg, errors='coerce').std()
            m_p, s_p = pd.to_numeric(X_pos, errors='coerce').mean(), pd.to_numeric(X_pos, errors='coerce').std()
            
            res['desc_total'] = f"<span class='n-badge'>n={n_used}</span><br>Mean: {m_t:.2f}<br>(SD {s_t:.2f})"
            res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
            res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
            
            # Mann-Whitney U
            try:
                u, p = stats.mannwhitneyu(pd.to_numeric(X_neg, errors='coerce').dropna(), pd.to_numeric(X_pos, errors='coerce').dropna())
                res['p_comp'] = p
            except: res['p_comp'] = np.nan

        # -----------------------------------
        # 2. Univariate Regression (Crude OR)
        # -----------------------------------
        data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
        if not data_uni.empty and data_uni['x'].nunique() > 1:
            params, conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']])
            
            if status == "OK" and 'x' in params:
                coef = params['x']
                or_val = np.exp(coef)
                ci_low = np.exp(conf.loc['x'][0])
                ci_high = np.exp(conf.loc['x'][1])
                res['or'] = f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"
                res['p_or'] = pvals['x']
            else:
                res['or'] = "-"
        else:
            res['or'] = "-"

        results_db[col] = res
        
        # Screening for Multivariate (P < 0.20)
        p_screen = res.get('p_comp', np.nan)
        if pd.isna(p_screen): p_screen = res.get('p_or', np.nan)
        
        if pd.notna(p_screen) and p_screen < 0.20:
            candidates.append(col)

    # -----------------------------------
    # 3. MULTIVARIATE ANALYSIS
    # -----------------------------------
    aor_results = {}
    
    # Filter candidates with enough data
    cand_valid = []
    for c in candidates:
        if df_aligned[c].apply(clean_numeric_value).notna().sum() > 10:
             cand_valid.append(c)

    final_n_multi = 0

    if len(cand_valid) > 0:
        # Prepare Multivariate Data
        multi_df = pd.DataFrame({'y': y})
        for c in cand_valid:
            multi_df[c] = df_aligned[c].apply(clean_numeric_value)
        
        # Complete Case Analysis (Drop Missing)
        multi_data = multi_df.dropna()
        final_n_multi = len(multi_data)
        
        # Run if we have enough cases
        if not multi_data.empty and final_n_multi > 10:
            # ใช้ BFGS ตรงนี้สำคัญมาก
            params, conf, pvals, status = run_binary_logit(multi_data['y'], multi_data[cand_valid], method='bfgs')
            
            if status == "OK":
                for var in cand_valid:
                    if var in params:
                        coef = params[var]
                        aor = np.exp(coef)
                        ci_low = np.exp(conf.loc[var][0])
                        ci_high = np.exp(conf.loc[var][1])
                        ap = pvals[var]
                        aor_results[var] = {'aor': f"{aor:.2f} ({ci_low:.2f}-{ci_high:.2f})", 'ap': ap}

    # -----------------------------------
    # 4. Generate Table HTML
    # -----------------------------------
    html_rows = []
    current_sheet = ""
    for col in sorted_cols:
        if col == outcome_name or col not in results_db: continue
        res = results_db[col]
        sheet = col.split('_')[0]
        if sheet != current_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='8'>{sheet}</td></tr>")
            current_sheet = sheet
            
        lbl = get_label(col, var_meta)
        or_s = res.get('or', '-')
        
        def fmt_p(val):
            if pd.isna(val): return "-"
            if val < 0.001: return "<0.001"
            return f"{val:.3f}"

        p_val = res.get('p_comp', np.nan)
        p_s = fmt_p(p_val)
        if pd.notna(p_val) and p_val < 0.05: p_s = f"<span class='sig-p'>{p_s}*</span>"
        
        if col in aor_results:
            ar = aor_results[col]
            aor_s = ar['aor']
            ap_val = ar['ap']
            ap_s = fmt_p(ap_val)
            if pd.notna(ap_val) and ap_val < 0.05: ap_s = f"<span class='sig-p'>{ap_s}*</span>"
        else:
            aor_s = "-"
            ap_s = "-"
            
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
    
    table_html = f"""
    <div id='{outcome_name}' class='table-container'>
    <div class='outcome-title'>Outcome: {outcome_name} (Total n={total_n})</div>
    <table>
        <thead>
            <tr>
                <th>Variable</th>
                <th>Total (n={len(y)})</th>
                <th>Group Outcome (-)</th>
                <th>Group Outcome (+)</th>
                <th>OR (95% CI)</th>
                <th>P-value</th>
                <th>aOR (95% CI)<br><span style='font-size:0.8em; font-weight:normal'>(Complete Case n={final_n_multi})</span></th>
                <th>aP-value</th>
            </tr>
        </thead>
        <tbody>
            {"".join(html_rows)}
        </tbody>
    </table>
    <div class='summary-box'>
        <b>Method:</b> Binary Logistic Regression (BFGS Optimization). <br>
        <b>Multivariate:</b> Included {len(cand_valid)} variables (p<0.20). Analysis used Complete Cases (n={final_n_multi}).<br>
        <i>* P-value < 0.05 is considered statistically significant.</i>
    </div>
    </div>
    <br>
    """
    return table_html

# ==========================================
# 5. Main Processing Function (เรียกใช้จาก App)
# ==========================================

def process_data_and_generate_html(df, var_meta=None):
    """
    Main entry point for the Streamlit app.
    Args:
        df: Pandas DataFrame containing all data
        var_meta: Dictionary containing codebook metadata (optional)
    Returns:
        String: Full HTML report
    """
    
    # --- 1. Data Preprocessing ---
    # ทำสำเนาเพื่อไม่ให้กระทบ DataFrame ต้นฉบับ
    df = df.copy()
    
    # 🛠️ STEP: แยก BP (109/76) -> SBP (109) และ DBP (76)
    if 'Physical exam_bp' in df.columns:
        def split_bp(val):
            try:
                if pd.isna(val): return np.nan, np.nan
                parts = str(val).split('/')
                if len(parts) == 2:
                    return float(parts[0].strip()), float(parts[1].strip())
                return np.nan, np.nan
            except:
                return np.nan, np.nan

        bp_data = df['Physical exam_bp'].apply(split_bp).tolist()
        df['Physical exam_sbp'] = [x[0] for x in bp_data]
        df['Physical exam_dbp'] = [x[1] for x in bp_data]
        # ไม่ drop คอลัมน์เดิมก็ได้ เผื่อ user อยากเห็น แต่ในที่นี้ drop เพื่อความสะอาดของตาราง
        df = df.drop(columns=['Physical exam_bp'], errors='ignore')
        
        # Add meta for new columns if var_meta exists
        if var_meta:
            var_meta['sbp'] = {'label': 'Systolic Blood Pressure', 'map': {}}
            var_meta['dbp'] = {'label': 'Diastolic Blood Pressure', 'map': {}}

    # --- 2. Define Outcomes to Analyze ---
    # ถ้าอยากให้ User เลือก Outcome ได้ ให้รับเป็น Parameter เพิ่ม
    # ในที่นี้ใช้ Default ตามโจทย์เดิม
    outcomes_to_run = ['Outcomes_newrvbad', 'Outcomes_newrvgood', 'Outcomes_sumrv', 'Outcomes_sumoutcome']
    
    # --- 3. CSS Style ---
    css_style = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; padding-bottom: 80px; background-color: #f4f6f8; color: #333; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; font-weight: 300; letter-spacing: 1px; }
        
        /* Table & Container */
        .table-container { 
            background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
            margin-bottom: 50px; position: relative; scroll-margin-top: 20px; 
        }
        table { width: 100%; border-collapse: separate; border-spacing: 0; text-align: left; }
        
        /* Double Sticky Headers */
        .outcome-title { 
            background-color: #2c3e50; color: white; padding: 15px 20px; 
            font-size: 1.2em; font-weight: 500; border-radius: 8px 8px 0 0; 
            position: sticky; top: 0; z-index: 102; box-shadow: 0 2px 4px rgba(0,0,0,0.2); 
        }
        th { 
            background-color: #34495e; color: #fff; font-weight: 600; 
            padding: 12px 15px; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.5px; 
            position: sticky; top: 55px; z-index: 101; box-shadow: 0 2px 2px -1px rgba(0, 0, 0, 0.4); 
        }

        td { padding: 12px 15px; border-bottom: 1px solid #eee; vertical-align: top; font-size: 0.95em; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f7ff; transition: background-color 0.2s; }
        
        .sheet-header td { background-color: #e8f4f8; color: #2980b9; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9em; padding: 10px 20px; border-left: 5px solid #3498db; }
        .n-badge { font-size: 0.75em; background-color: #eee; padding: 2px 6px; border-radius: 4px; color: #555; display: inline-block; margin-bottom: 4px; }
        .sig-p { color: #d32f2f; font-weight: bold; background-color: #ffebee; padding: 2px 5px; border-radius: 4px; }
        .summary-box { padding: 15px 20px; background-color: #fff; border-top: 1px solid #eee; font-size: 0.9em; color: #555; border-radius: 0 0 8px 8px; }
        
        /* Quick Nav */
        .quick-nav { 
            position: fixed; bottom: 20px; right: 20px; 
            background: rgba(255, 255, 255, 0.95); padding: 10px 15px; border-radius: 30px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.2); display: flex; gap: 8px; z-index: 1000; 
            backdrop-filter: blur(5px); border: 1px solid #eee; 
        }
        .quick-nav a { 
            text-decoration: none; color: #555; font-weight: 500; font-size: 0.85em; 
            padding: 8px 16px; border-radius: 20px; background: #f4f6f8; 
            transition: all 0.2s; white-space: nowrap; 
        }
        .quick-nav a:hover { background: #3498db; color: white; transform: translateY(-2px); }
        .quick-nav .top-btn { background: #2c3e50; color: white; }

        @media screen and (max-width: 768px) {
            body { padding: 10px; padding-bottom: 100px; }
            .table-container { overflow-x: auto; -webkit-overflow-scrolling: touch; }
            table { min-width: 900px; }
            .outcome-title { position: sticky; left: 0; width: 100%; top: 0; }
            th { top: 55px; }
            th, td { padding: 10px 8px; font-size: 0.85em; }
            .quick-nav { left: 50%; transform: translateX(-50%); bottom: 15px; width: 90%; flex-wrap: wrap; justify-content: center; }
            .quick-nav a { flex: 1; text-align: center; }
        }
    </style>
    """

    # --- 4. Build HTML ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Research Analysis Results</title>
        {css_style}
    </head>
    <body>
        <div id="top"></div>
        <h1>Research Analysis Report</h1>
        <div style="text-align:center; color:#555; margin-bottom:40px;">
            Generated by <b>Auto Stat Tool</b> (Running locally in your browser)
        </div>
    """
    
    # Run Analysis Loop
    found_any = False
    nav_links = ""
    
    for out in outcomes_to_run:
        if out in df.columns:
            html_content += analyze_outcome(out, df, var_meta)
            
            # Create Nav Link
            short_name = out.replace("Outcomes_", "")
            nav_links += f"<a href='#{out}'>{short_name}</a>"
            found_any = True
        else:
            # กรณีไม่เจอคอลัมน์ Outcome ในไฟล์ที่อัพโหลด
            pass

    if not found_any:
        html_content += "<h3 style='text-align:center; color:red'>❌ No matching outcome columns found. Please check your file headers.</h3>"

    # Navigation Footer
    html_content += f"""
        <div class="quick-nav">
            {nav_links}
            <a href="#top" class="top-btn">↑ Top</a>
        </div>
    </body>
    </html>
    """
    
    return html_content
