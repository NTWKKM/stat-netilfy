import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score # 🟢 Import เพิ่ม
import matplotlib.pyplot as plt
import io, base64
import streamlit as st
import html as _html

# ... (ฟังก์ชัน calculate_descriptive และ calculate_chi2 ของเดิม เก็บไว้เหมือนเดิม) ...

@st.cache_data(show_spinner=False) 
def calculate_descriptive(df, col):
    """คำนวณสถิติพื้นฐาน"""
    if col not in df.columns: return "Column not found"
    data = df[col].dropna()
    try:
        num_data = pd.to_numeric(data, errors='raise')
        is_numeric = True
    except:
        is_numeric = False
        
    if is_numeric:
        desc = num_data.describe()
        return pd.DataFrame({
            "Statistic": ["Count", "Mean", "SD", "Median", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
            "Value": [
                f"{desc['count']:.0f}", f"{desc['mean']:.4f}", f"{desc['std']:.4f}",
                f"{desc['50%']:.4f}", f"{desc['min']:.4f}", f"{desc['max']:.4f}",
                f"{desc['25%']:.4f}", f"{desc['75%']:.4f}"
            ]
        })
    else:
        counts = data.value_counts()
        percent = data.value_counts(normalize=True) * 100
        return pd.DataFrame({
            "Category": counts.index, "Count": counts.values, "Percentage (%)": percent.values
        }).sort_values("Count", ascending=False)

@st.cache_data(show_spinner=False)
def calculate_chi2(df, col1, col2, method='Pearson (Standard)', v1_pos=None, v2_pos=None):
    """(SYNCED) คำนวณ Chi-square/Fisher"""
    if col1 not in df.columns or col2 not in df.columns: 
        return None, None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    
    # 1. Crosstab
    tab_chi2 = pd.crosstab(data[col1], data[col2])
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    tab_row_pct = pd.crosstab(data[col1], data[col2], normalize='index', margins=True, margins_name="Total") * 100
    
    # --- REORDERING LOGIC ---
    all_col_labels = tab_raw.columns.tolist() 
    all_row_labels = tab_raw.index.tolist()
    base_col_labels = [col for col in all_col_labels if col != 'Total']
    base_row_labels = [row for row in all_row_labels if row != 'Total']

    def get_original_label(label_str, df_labels):
        for lbl in df_labels:
            if str(lbl) == label_str: return lbl
        return label_str 

    # Reorder Cols
    final_col_order_base = base_col_labels[:]
    if v2_pos is not None: 
        v2_pos_original = get_original_label(v2_pos, base_col_labels)
        if v2_pos_original in final_col_order_base:
            final_col_order_base.remove(v2_pos_original)
            final_col_order_base.insert(0, v2_pos_original)
    else:
        def custom_sort(label):
            try: return float(label)
            except (ValueError, TypeError): return str(label)
        final_col_order_base.sort(key=custom_sort, reverse=True)
    final_col_order = final_col_order_base + ['Total'] 

    # Reorder Rows
    final_row_order_base = base_row_labels[:]
    if v1_pos is not None: 
        v1_pos_original = get_original_label(v1_pos, base_row_labels)
        if v1_pos_original in final_row_order_base:
            final_row_order_base.remove(v1_pos_original)
            final_row_order_base.insert(0, v1_pos_original)
    else:
        def custom_sort(label):
            try:
                # numeric labels first, then non‑numeric; both sortable
                return (0, float(label))
            except (ValueError, TypeError):
                return (1, str(label))
        final_row_order_base.sort(key=custom_sort, reverse=True)
    final_row_order = final_row_order_base + ['Total']

    # Reindex
    tab_raw = tab_raw.reindex(index=final_row_order, columns=final_col_order)
    tab_row_pct = tab_row_pct.reindex(index=final_row_order, columns=final_col_order)
    tab_chi2 = tab_chi2.reindex(index=final_row_order_base, columns=final_col_order_base)
    
    col_names = final_col_order 
    index_names = final_row_order

    display_data = []
    for row_name in index_names:
        row_data = []
        for col_name in col_names:
            count = tab_raw.loc[row_name, col_name]
            if col_name == 'Total': pct = 100.0
            else: pct = tab_row_pct.loc[row_name, col_name]
            cell_content = f"{count} ({pct:.1f}%)"
            row_data.append(cell_content)
        display_data.append(row_data)
    
    display_tab = pd.DataFrame(display_data, columns=col_names, index=index_names)
    display_tab.index.name = col1
    
    # Stats
    try:
        is_2x2 = (tab_chi2.shape == (2, 2))
        
        if "Fisher" in method:
            if not is_2x2: return display_tab, None, "Error: Fisher's Exact Test requires a 2x2 table.", None
            odds_ratio, p_value = stats.fisher_exact(tab_chi2)
            method_name = "Fisher's Exact Test"
            msg = f"{method_name}: P-value={p_value:.4f}, OR={odds_ratio:.4f}"
            stats_res = {"Test": method_name, "Statistic (OR)": odds_ratio, "P-value": p_value, "Degrees of Freedom": "-", "N": len(data)}
        else:
            use_correction = True if "Yates" in method else False
            chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=use_correction)
            method_name = "Chi-Square"
            if is_2x2: method_name += " (with Yates')" if use_correction else " (Pearson)"
            msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
            stats_res = {"Test": method_name, "Statistic": chi2, "P-value": p, "Degrees of Freedom": dof, "N": len(data)}
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg += " ⚠️ Warning: Expected count < 5. Consider using Fisher's Exact Test."
        
        risk_df = None
        if is_2x2:
            try:
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                row_labels = tab_chi2.index.tolist(); col_labels = tab_chi2.columns.tolist()
                label_exp = str(row_labels[0]); label_unexp = str(row_labels[1]); label_event = str(col_labels[0])
                risk_exp = a/(a+b) if (a+b)>0 else 0; risk_unexp = c/(c+d) if (c+d)>0 else 0
                rr = risk_exp/risk_unexp if risk_unexp>0 else np.nan
                rd = risk_exp - risk_unexp; nnt = abs(1/rd) if rd!=0 else np.inf
                odd_ratio, _ = stats.fisher_exact(tab_chi2)
                
                risk_data = [
                    {"Statistic": f"Risk in {label_exp} (R1)", "Value": f"{risk_exp:.4f}", "Interpretation": f"Risk of '{label_event}' in group {label_exp}"},
                    {"Statistic": f"Risk in {label_unexp} (R0)", "Value": f"{risk_unexp:.4f}", "Interpretation": f"Baseline Risk of '{label_event}' in group {label_unexp}"},
                    {"Statistic": "Risk Ratio (RR)", "Value": f"{rr:.4f}", "Interpretation": f"Risk in {label_exp} is {rr:.2f} times that of {label_unexp}"},
                    {"Statistic": "Risk Difference (RD)", "Value": f"{rd:.4f}", "Interpretation": f"Absolute difference (R1 - R0)"},
                    {"Statistic": "Number Needed to Treat (NNT)", "Value": f"{nnt:.1f}", "Interpretation": "Patients to treat to prevent/cause 1 outcome"},
                    {"Statistic": "Odds Ratio (OR)", "Value": f"{odd_ratio:.4f}", "Interpretation": "Odds of Event (Exp vs Unexp)"}
                ]
                risk_df = pd.DataFrame(risk_data)
            except: pass

        return display_tab, stats_res, msg, risk_df

    except Exception as e:
        return display_tab, None, str(e), None

# 🟢 NEW: ฟังก์ชันคำนวณ Kappa
@st.cache_data(show_spinner=False)
def calculate_kappa(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns:
        return None, "Columns not found", None

    data = df[[col1, col2]].dropna()
    if data.empty: return None, "No data after dropping NAs", None

    # แปลงเป็น String เพื่อให้ชัวร์ว่าเป็น Categorical
    y1 = data[col1].astype(str)
    y2 = data[col2].astype(str)

    # คำนวณ Kappa
    try:
        kappa = cohen_kappa_score(y1, y2)
        
        # แปลผล (Landis & Koch, 1977)
        if kappa < 0: interp = "Poor agreement"
        elif kappa <= 0.20: interp = "Slight agreement"
        elif kappa <= 0.40: interp = "Fair agreement"
        elif kappa <= 0.60: interp = "Moderate agreement"
        elif kappa <= 0.80: interp = "Substantial agreement"
        else: interp = "Perfect/Almost perfect agreement"

        res_df = pd.DataFrame({
            "Statistic": ["Cohen's Kappa", "N (Pairs)", "Interpretation"],
            "Value": [f"{kappa:.4f}", f"{len(data)}", interp]
        })
        
        # Confusion Matrix
        conf_matrix = pd.crosstab(y1, y2, rownames=[f"{col1} (Obs 1)"], colnames=[f"{col2} (Obs 2)"])
        
    except ValueError as e:
        return None, str(e), None
    else:
        return res_df, None, conf_matrix

# --- ROC Functions (เหมือนเดิม) ---
def auc_ci_hanley_mcneil(auc, n1, n2):
    q1 = auc / (2 - auc); q2 = 2 * (auc**2) / (1 + auc)
    se_auc = np.sqrt(((auc * (1 - auc)) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1 * n2))
    return auc - 1.96 * se_auc, auc + 1.96 * se_auc, se_auc

def auc_ci_delong(y_true, y_scores):
    y_true = np.array(y_true); y_scores = np.array(y_scores)
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]; y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]; fps = 1 + threshold_idxs - tps
    n_pos = tps[-1]; n_neg = fps[-1]
    if n_pos == 0 or n_neg == 0: return np.nan, np.nan, np.nan
    auc = roc_auc_score(y_true, y_scores)
    pos_scores = y_scores[y_true == 1]; neg_scores = y_scores[y_true == 0]
    v10 = []; v01 = []
    for p in pos_scores: v10.append( (np.sum(p > neg_scores) + 0.5*np.sum(p == neg_scores)) / n_neg )
    for n in neg_scores: v01.append( (np.sum(pos_scores > n) + 0.5*np.sum(pos_scores == n)) / n_pos )
    s10 = np.var(v10, ddof=1); s01 = np.var(v01, ddof=1)
    se_auc = np.sqrt((s10 / n_pos) + (s01 / n_neg))
    return auc - 1.96*se_auc, auc + 1.96*se_auc, se_auc

@st.cache_data(show_spinner=False)
def analyze_roc(df, truth_col, score_col, method='delong', pos_label_user=None):
    data = df[[truth_col, score_col]].dropna()
    y_true_raw = data[truth_col]
    y_score = pd.to_numeric(data[score_col], errors='coerce').dropna()
    y_true_raw = y_true_raw.loc[y_score.index]
    
    if y_true_raw.nunique() != 2 or pos_label_user is None:
        return None, "Error: Binary outcome required.", None, None

    y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return None, "Error: Need both classes after dropping NA scores.", None, None
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    
    if method == 'delong': ci_lower, ci_upper, se = auc_ci_delong(y_true, y_score.values); m_name = "DeLong"
    else: ci_lower, ci_upper, se = auc_ci_hanley_mcneil(auc_val, n1, n0); m_name = "Hanley"
    
    p_val_auc = (
        stats.norm.sf(abs((auc_val - 0.5) / se)) * 2
        if (se is not None and np.isfinite(se) and se > 0)
        else np.nan
    )
    j_scores = tpr - fpr; best_idx = np.argmax(j_scores)
    
    stats_res = {
        "AUC": auc_val, "SE": se, "95% CI Lower": max(0, ci_lower), "95% CI Upper": min(1, ci_upper),
        "Method": m_name, "P-value": p_val_auc, "Youden J": j_scores[best_idx],
        "Best Cut-off": thresholds[best_idx], "Sensitivity": tpr[best_idx], "Specificity": 1-fpr[best_idx],
        "N(+)": n1, "N(-)": n0, "Positive Label": pos_label_user
    }
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={auc_val:.3f}'); ax.plot([0,1],[0,1],'k--')
    ax.plot(1-(1-fpr[best_idx]), tpr[best_idx], 'ro')
    ax.set_xlabel('1-Specificity'); ax.set_ylabel('Sensitivity'); ax.legend()
    
    coords_df = pd.DataFrame({'Threshold': thresholds, 'Sens': tpr, 'Spec': 1-fpr}).round(4)
    return stats_res, None, fig, coords_df

# 🟢 NEW: ฟังก์ชันคำนวณ ICC (Intraclass Correlation Coefficient)
@st.cache_data(show_spinner=False)
def calculate_icc(df, cols):
    """
    คำนวณ ICC(2,1) และ ICC(3,1) โดยใช้ Two-way ANOVA Formula
    Ref: Shrout & Fleiss (1979), Koo & Li (2016)
    """
    # 1. Prepare Data
    if len(cols) < 2: return None, "Please select at least 2 variables (raters/methods).", None
    data = df[cols].dropna()
    n, k = data.shape # n=subjects, k=raters
     
    if n < 2: return None, "Insufficient data (need at least 2 rows).", None
    if k < 2: return None, "Insufficient raters (need at least 2 columns).", None
    
    # 2. ANOVA Calculations (Manual Calculation using Numpy for Speed & No Dependency)
    # Grand Mean
    grand_mean = data.values.mean()
    
    # Sum of Squares
    SStotal = ((data.values - grand_mean)**2).sum()
    
    # Between-subjects (Rows)
    row_means = data.mean(axis=1)
    SSrow = k * ((row_means - grand_mean)**2).sum()
    
    # Between-raters (Cols)
    col_means = data.mean(axis=0)
    SScol = n * ((col_means - grand_mean)**2).sum()
    
    # Residual (Error)
    SSres = SStotal - SSrow - SScol
    
    # Degrees of Freedom
    df_row = n - 1
    df_col = k - 1
    df_res = df_row * df_col
    
    # Mean Squares
    MSrow = SSrow / df_row
    MScol = SScol / df_col
    MSres = SSres / df_res

     # Guard against zero denominators (no variance)
    denom_icc3 = MSrow + (k - 1) * MSres
    denom_icc2 = MSrow + (k - 1) * MSres + (k / n) * (MScol - MSres)
    if denom_icc3 == 0 or denom_icc2 == 0:
        return None, "Insufficient variance to compute ICC (denominator = 0).", None
    # 3. Calculate ICCs
    # ICC(3,1) Consistency: Fixed raters, Single measure
    # Formula: (MSR - MSE) / (MSR + (k-1)MSE)
    icc3_1 = (MSrow - MSres) / denom_icc3
    
    # ICC(2,1) Absolute Agreement: Random raters, Single measure
    # Formula: (MSR - MSE) / (MSR + (k-1)MSE + (k/n)(MSC - MSE))
    icc2_1 = (MSrow - MSres) / denom_icc2
    
    # Interpretation (Koo & Li, 2016)
    def interpret_icc(v):
        # Treat NaN/inf as undefined rather than "Excellent"
        if not np.isfinite(v): return "Undefined"
        if v < 0.5: return "Poor"
        elif v < 0.75: return "Moderate"
        elif v < 0.9: return "Good"
        else: return "Excellent"

    res_df = pd.DataFrame({
        "Model": ["ICC(2,1) - Absolute Agreement", "ICC(3,1) - Consistency"],
        "Description": [
            "Use when raters are random & agreement matters (e.g. 2 different machines)", 
            "Use when raters are fixed & consistency matters (e.g. ranking consistency)"
        ],
        "ICC Value": [icc2_1, icc3_1],
        "Interpretation": [interpret_icc(icc2_1), interpret_icc(icc3_1)]
    })
    
    # Format Value
    res_df["ICC Value"] = res_df["ICC Value"].map('{:.4f}'.format)
    
    # ANOVA Table (Optional, for debugging or detailed report)
    anova_df = pd.DataFrame({
        "Source": ["Between Subjects (Rows)", "Between Raters (Cols)", "Residual (Error)"],
        "SS": [SSrow, SScol, SSres],
        "df": [df_row, df_col, df_res],
        "MS": [MSrow, MScol, MSres]
    })
    
    return res_df, None, anova_df
    
def generate_report(title, elements):
    # (คงเดิม - เพราะมีการปิด plt.close(data) ในนี้อยู่แล้ว)
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; margin: 0; color: #333; }
        .report-container { background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); padding: 20px; width: 100%; box-sizing: border-box; margin-bottom: 20px; }
        h2 { color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        h4 { color: #34495e; margin-top: 25px; margin-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 0.9em; }
        th, td { padding: 10px 15px; border: 1px solid #e0e0e0; vertical-align: middle; text-align: center; } 
        th { background-color: #f0f2f6; font-weight: 600; }
        tr:nth-child(even) td { background-color: #f9f9f9; }
        .report-footer { text-align: right; font-size: 0.75em; color: #666; margin-top: 20px; border-top: 1px dashed #ddd; padding-top: 10px; }
        .th-exposure { text-align: left; background-color: #e8ecf1; } 
        .th-outcome { background-color: #e8ecf1; }
        .td-label { text-align: left; font-weight: bold; background-color: #fcfcfc; }
    </style>
    """
    html = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html += f"<div class='report-container'><h2>{title}</h2>"
    
    for element in elements:
        element_type = element['type']
        data = element['data']
        header = element.get('header', '')
        if header: html += f"<h4>{header}</h4>"
        
        if element_type == 'text': html += f"<p>{data}</p>"
        elif element_type == 'table': 
            idx = not ('Interpretation' in data.columns)
            html += data.to_html(index=idx, classes='report-table')
        elif element_type == 'contingency_table':
            col_labels = data.columns.tolist() 
            row_labels = data.index.tolist()   
            exp_name = data.index.name         
            out_name = element.get('outcome_col', 'Outcome')
            html_tab = "<table>"
            html_tab += "<thead><tr>"
            html_tab += "<th style='background-color: white; border: none;'></th>" 
            html_tab += f"<th colspan='{len(col_labels)}' class='th-outcome'>{out_name}</th>"
            html_tab += "</tr>"
            html_tab += "<tr>"
            html_tab += f"<th class='th-exposure'>{exp_name}</th>"
            for label in col_labels:
                html_tab += f"<th>{_html.escape(str(label))}</th>"
            html_tab += "</tr></thead>"
            html_tab += "<tbody>"
            for idx_label in row_labels:
                html_tab += "<tr>"
                html_tab += f"<td class='td-label'>{_html.escape(str(idx_label))}</td>"
                for col_label in col_labels:
                    val = data.loc[idx_label, col_label]
                    html_tab += f"<td>{_html.escape(str(val))}</td>"
                html_tab += "</tr>"
            html_tab += "</tbody></table>"
            html += html_tab
        elif element_type == 'plot':
            buf = io.BytesIO()
            if isinstance(data, plt.Figure):
                data.savefig(buf, format='png', bbox_inches='tight'); plt.close(data)
                uri = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{uri}" style="max-width: 100%;"/>'
            buf.close()
            
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""
    return html
