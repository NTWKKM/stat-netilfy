import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
import streamlit as st  # ✅ IMPORT STREAMLIT

# ✅ TRY IMPORT FIRTHLOGIST
try:
    from firthlogist import FirthLogisticRegression
    HAS_FIRTH = True
except ImportError:
    HAS_FIRTH = False

warnings.filterwarnings("ignore")

def clean_numeric_value(val):
    """
    Normalize a value into a numeric float suitable for analysis.
    
    Cleans common non-numeric markers (such as leading/trailing whitespace, '>', '<', and thousands separators like ',') and converts the result to a float. If the input is missing or cannot be parsed as a number, returns NaN.
    
    Parameters:
        val: The input value to normalize (may be a string, number, or missing).
    
    Returns:
        numeric_value (float): The parsed float, or `np.nan` when the value is missing or unparseable.
    """
    if pd.isna(val): return np.nan
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except:
        return np.nan

def run_binary_logit(y, X, method='default'):
    """
    Execute a binary logistic regression using the selected estimation method.
    
    Supports three methods: 'default' (statsmodels Logit with default optimizer), 'bfgs' (statsmodels Logit using BFGS), and 'firth' (Firth's penalized likelihood when available).
    
    Parameters:
        y (array-like or pd.Series): Binary outcome vector aligned to rows of X.
        X (array-like or pd.DataFrame): Predictors matrix; an intercept column will be added automatically.
        method (str): One of 'default', 'bfgs', or 'firth'. If 'firth' is requested but the firthlogist library is unavailable, the function returns an error message.
    
    Returns:
        tuple: (params, conf_int, pvalues, status)
            - params (pd.Series or None): Estimated coefficients indexed by predictor names (including the intercept) or None on failure.
            - conf_int (pd.DataFrame or None): Confidence intervals with columns [0, 1] indexed by predictor names, or None on failure.
            - pvalues (pd.Series or None): Two-sided p-values indexed by predictor names, or None on failure.
            - status (str): "OK" on success; otherwise an error message (for example when firthlogist is not installed or another exception occurred).
    """
    try:
        # เตรียมข้อมูล (Statsmodels ต้องการ Constant เสมอ)
        X_const = sm.add_constant(X, has_constant='add')
        
        # 🟢 CASE 1: FIRTH'S LOGISTIC REGRESSION (Recommended)
        if method == 'firth':
            if not HAS_FIRTH:
                # ถ้า User เลือก Firth แต่ไม่มี Library ให้ Return Error
                return None, None, None, "Library 'firthlogist' not installed. Please define requirements.txt or use Standard method."
            
            # firthlogist: fit_intercept=False เพราะเราใส่ Constant ใน X ไปแล้ว
            fl = FirthLogisticRegression(fit_intercept=False) 
            fl.fit(X_const, y)
            
            # แปลงผลลัพธ์ให้ตรงกับ Format เดิม (Series/DataFrame)
            params = pd.Series(fl.coef_, index=X_const.columns)
            pvalues = pd.Series(getattr(fl, "pvals_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
            ci = getattr(fl, "ci_", None)
            conf_int = (
                pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])
                if ci is not None
                else pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
            )
            
            return params, conf_int, pvalues, "OK"

        # 🔵 CASE 2: STANDARD LOGISTIC (Statsmodels)
        elif method == 'bfgs':
            model = sm.Logit(y, X_const).fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const).fit(disp=0)
            
        return model.params, model.conf_int(), model.pvalues, "OK"
        
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        return None, None, None, str(e)

def get_label(col_name, var_meta):
    """
    Create an HTML label for a variable by deriving a display name from the column name and optional metadata.
    
    Parameters:
    	col_name (str): Column identifier; if it contains an underscore, the substring after the first underscore is used as the variable name shown.
    	var_meta (dict or None): Optional mapping from variable name to metadata dict. If metadata for the variable contains a 'label' entry, that value is used as the secondary (grey) label.
    
    Returns:
    	html_label (str): An HTML string with the variable name in bold on the first line and a secondary grey label on the second line.
    """
    parts = col_name.split('_', 1)
    orig_name = parts[1] if len(parts) > 1 else col_name
    
    label = orig_name 
    # [แก้ไข] ย่อหน้าบรรทัด 105 และ 106 ให้ถูกต้อง
    if var_meta and orig_name in var_meta:
        if 'label' in var_meta[orig_name]:          # <--- ย่อหน้าเข้ามา 8 spaces
            label = var_meta[orig_name]['label']    # <--- ย่อหน้าเข้ามา 12 spaces
             
    return f"<b>{orig_name}</b><br><span style='color:#666; font-size:0.9em'>{label}</span>"

# ✅ CACHE DATA: ช่วยให้เว็บเร็วขึ้น ไม่ต้องคำนวณใหม่ทุกครั้งที่กดปุ่มอื่น
# 🟢 NOTE: ต้องเพิ่ม method ลงใน argument เพื่อให้ cache แยกกันตาม method ที่เลือก
@st.cache_data(show_spinner=False)
def analyze_outcome(outcome_name, df, var_meta=None, method='auto'):
    """
    Analyze a binary outcome against all other columns in a dataframe and produce an HTML report summarizing univariate and multivariate results.
    
    Per-column descriptive statistics and univariate comparisons are computed (Chi-square for categorical, Mann–Whitney U for continuous), univariable logistic regression provides crude odds ratios, and a multivariable logistic model is fitted on screened candidate predictors to produce adjusted odds ratios when feasible.
    
    Parameters:
        outcome_name (str): Column name of the binary outcome in `df`.
        df (pandas.DataFrame): Input dataset containing `outcome_name` and candidate predictors.
        var_meta (dict, optional): Variable metadata that can supply display labels, mapping for categorical values, or force a variable `type` ('Categorical' or 'Continuous'). Keys may be full column names or the original variable name portion after the first underscore.
        method (str, optional): Regression method to use; one of 'auto', 'firth', or 'bfgs'. 'auto' selects Firth's penalized likelihood when available, otherwise BFGS-based logistic regression.
    
    Returns:
        str: An HTML fragment containing a table of variables with descriptive statistics, crude odds ratios (and p-values), and adjusted odds ratios where multivariable modelling was performed. If `outcome_name` is not found in `df`, returns an HTML alert div indicating the missing outcome.
    """
    if outcome_name not in df.columns:
        return f"<div class='alert'>⚠️ Outcome '{outcome_name}' not found.</div>"
    
    y = df[outcome_name].dropna().astype(int)
    df_aligned = df.loc[y.index]
    total_n = len(y)
    
    candidates = [] 
    results_db = {} 
    sorted_cols = sorted(df.columns)

    # 🟢 Logic การเลือก Method ตามที่ User สั่ง
    preferred_method = 'bfgs' # Default fallback
    
    if method == 'auto':
        preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
    elif method == 'firth':
        preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
    elif method == 'bfgs':
        preferred_method = 'bfgs'

    for col in sorted_cols:
        if col == outcome_name: continue
        if df_aligned[col].isnull().all(): continue

        res = {'var': col}
        X_raw = df_aligned[col]
        X_num = X_raw.apply(clean_numeric_value)
        
        X_neg = X_raw[y == 0]
        X_pos = X_raw[y == 1]
        
        # ชื่อตัวแปร
        orig_name = col.split('_', 1)[1] if len(col.split('_', 1)) > 1 else col
        
        # --- TYPE DETECTION ---
        unique_vals = X_num.dropna().unique()
        unique_count = len(unique_vals)
        
        is_categorical = False
        is_binary = set(unique_vals).issubset({0, 1})
        if is_binary or unique_count < 5:
            is_categorical = True
            
        # User Override
        user_setting = {}
        if var_meta and (col in var_meta or orig_name in var_meta):
            key = col if col in var_meta else orig_name
            user_setting = var_meta[key]
            
            if user_setting.get('type') == 'Categorical':
                is_categorical = True
            elif user_setting.get('type') == 'Continuous':
                is_categorical = False
        
        # --- DESCRIPTIVE ANALYSIS ---
        if is_categorical:
            n_used = len(X_raw.dropna())
            mapper = user_setting.get('map', {})
            
            try: levels = sorted(X_raw.dropna().unique(), key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))
            except: levels = sorted(X_raw.astype(str).unique())
            
            desc_tot = [f"<span class='n-badge'>n={n_used}</span>"]
            desc_neg = [f"<span class='n-badge'>n={len(X_neg.dropna())}</span>"]
            desc_pos = [f"<span class='n-badge'>n={len(X_pos.dropna())}</span>"]
            
            for lvl in levels:
                try: 
                    if float(lvl).is_integer(): key = int(float(lvl))
                    else: key = float(lvl)
                except: key = lvl
                
                label_txt = mapper.get(key, str(lvl))
                lvl_str = str(lvl)
                if str(lvl).endswith('.0'): lvl_str = str(int(float(lvl)))
                
                def count_val(series, v_str):
                     """
                     Count how many elements in a pandas Series equal a given string after normalizing numeric-like values.
                     
                     This converts each element to string; if the string represents a number (allowing one decimal point) a trailing ".0" is removed (e.g., "1.0" -> "1") before comparing to v_str. The comparison is string equality performed after this normalization.
                     
                     Parameters:
                         series (pandas.Series): Series whose values will be normalized and compared.
                         v_str (str): Target string to match against each normalized series element.
                     
                     Returns:
                         int: Number of elements equal to v_str after normalization.
                     """
                     return (series.astype(str).apply(lambda x: x.replace('.0','') if x.replace('.','',1).isdigit() else x) == v_str).sum()

                c_all = count_val(X_raw, lvl_str)
                if c_all == 0:
                    c_all = (X_raw == lvl).sum()
                
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
            
            try:
                contingency = pd.crosstab(X_raw, y)
                if contingency.size > 0:
                    chi2, p, dof, ex = stats.chi2_contingency(contingency)
                    res['p_comp'] = p
                    res['test_name'] = "Chi-square"
                else: 
                    res['p_comp'] = np.nan
                    res['test_name'] = "-"
            except (ValueError, np.linalg.LinAlgError):
                res['p_comp'] = np.nan
                res['test_name'] = "-"
            
        else:
            n_used = len(X_num.dropna())
            m_t, s_t = X_num.mean(), X_num.std()
            m_n, s_n = pd.to_numeric(X_neg, errors='coerce').mean(), pd.to_numeric(X_neg, errors='coerce').std()
            m_p, s_p = pd.to_numeric(X_pos, errors='coerce').mean(), pd.to_numeric(X_pos, errors='coerce').std()
            
            res['desc_total'] = f"<span class='n-badge'>n={n_used}</span><br>Mean: {m_t:.2f}<br>(SD {s_t:.2f})"
            res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
            res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
            
            try:
                u, p = stats.mannwhitneyu(pd.to_numeric(X_neg, errors='coerce').dropna(), pd.to_numeric(X_pos, errors='coerce').dropna())
                res['p_comp'] = p
                res['test_name'] = "Mann-Whitney U"
            except (ValueError, TypeError): 
                res['p_comp'] = np.nan
                res['test_name'] = "-"

        # --- UNIVARIATE REGRESSION (Crude OR) ---
        # 🟢 ใช้ Method ที่ User เลือก
        data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
        if not data_uni.empty and data_uni['x'].nunique() > 1:
            params, conf, pvals, status = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
            if status == "OK" and 'x' in params:
                coef = params['x']
                or_val = np.exp(coef)
                
                if 'x' in conf.index:
                    ci_low, ci_high = np.exp(conf.loc['x'][0]), np.exp(conf.loc['x'][1])
                else:
                    ci_low, ci_high = np.nan, np.nan 
                    
                res['or'] = f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"
                res['p_or'] = pvals['x']
            else: res['or'] = "-"
        else: res['or'] = "-"

        results_db[col] = res
        
        # Screening P < 0.20
        p_screen = res.get('p_comp', np.nan)
        if pd.isna(p_screen): p_screen = res.get('p_or', np.nan)
        if pd.notna(p_screen) and p_screen < 0.20:
            candidates.append(col)

    # --- MULTIVARIATE ANALYSIS ---
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
            # 🟢 ใช้ Method ที่ User เลือก สำหรับ Multivariate
            params, conf, pvals, status = run_binary_logit(multi_data['y'], multi_data[cand_valid], method=preferred_method)
            
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
    current_sheet = ""
    for col in sorted_cols:
        if col == outcome_name or col not in results_db: continue
        res = results_db[col]
        
        sheet = col.split('_')[0] if '_' in col else "Variables"
        if sheet != current_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='9'>{sheet}</td></tr>")
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
            <td>{res.get('test_name', '-')}</td> <td>{p_s}</td>
            <td>{aor_s}</td>
            <td>{ap_s}</td>
        </tr>"""
        html_rows.append(row_html)
    
    # Update Footer Note
    if preferred_method == 'firth':
        method_note = "Firth's Penalized Likelihood (User Selected)"
    elif preferred_method == 'bfgs':
        method_note = "Standard Binary Logistic Regression (MLE)"
    else:
        method_note = "Binary Logistic Regression"

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
                <th>Test Used</th> <th>Crude P-value</th>
                <th>aOR (95% CI)<br><span style='font-size:0.8em; font-weight:normal'>(n={final_n_multi})</span></th>
                <th>aP-value</th>
            </tr>
        </thead>
        <tbody>{"".join(html_rows)}</tbody>
    </table>
    <div class='summary-box'>
        <b>Method:</b> {method_note}. Complete Case Analysis.<br>
        <i>Univariate comparison uses Chi-square test (Categorical) or Mann-Whitney U test (Continuous).</i>
    </div>
    </div><br>
    """

# 🟢 UPDATE: เพิ่ม method='auto' ใน parameter
def process_data_and_generate_html(df, target_outcome, var_meta=None, method='auto'):
    """
    Builds a complete HTML analysis report for a binary outcome from the provided DataFrame.
    
    Parameters:
    	df (pandas.DataFrame): Source data containing the outcome and predictor columns.
    	target_outcome (str): Column name of the binary outcome to analyze.
    	var_meta (dict | None): Optional variable metadata mapping used to override labels or force variable types.
    	method (str): Regression method to use for modeling; one of 'auto', 'firth', 'bfgs', or 'default'. 'auto' selects a suitable method based on availability.
    
    Returns:
    	html (str): A complete HTML document (string) containing the analysis table, method notes, and footer.
    """
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
        .sheet-header td { background-color: #e8f4f8; color: #2980b9; font-weight: bold; letter-spacing: 1px; padding: 8px 15px; }
        .n-badge { font-size: 0.75em; color: #888; background: #eee; padding: 1px 4px; border-radius: 3px; }
        .summary-box { padding: 15px; background: #fff; font-size: 0.9em; color: #555; }
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
    html += "<h1>Analysis Report</h1>"
    # 🟢 ส่งต่อ method ไปยัง analyze_outcome
    html += analyze_outcome(target_outcome, df, var_meta, method=method)
    
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div>"""
    
    html += "</body></html>"
    
    return html
