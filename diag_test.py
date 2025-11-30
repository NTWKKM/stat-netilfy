import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import io, base64

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

def calculate_chi2(df, col1, col2, correction=True):
    """
    (เหมือน correlation.py) คำนวณ Chi-square พร้อมตาราง 2 ชั้น และ Risk Interpretation
    """
    if col1 not in df.columns or col2 not in df.columns: 
        return None, None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    
    # 1. Contingency Table
    tab_chi2 = pd.crosstab(data[col1], data[col2])
    
    # 2. Display Table
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    tab_row_pct = pd.crosstab(data[col1], data[col2], normalize='index', margins=True, margins_name="Total") * 100
    tab_total_pct = pd.crosstab(data[col1], data[col2], normalize='all', margins=True, margins_name="Total") * 100
    
    col_names = tab_raw.columns.tolist() 
    index_names = tab_raw.index.tolist()
    display_data = []
    
    for row_name in index_names:
        row_data = []
        for col_name in col_names:
            count = tab_raw.loc[row_name, col_name]
            total_pct = tab_total_pct.loc[row_name, col_name]
            if col_name == 'Total' and row_name == 'Total':
                cell_content = f"{count} / ({total_pct:.1f}%)"
            elif col_name == 'Total' or row_name == 'Total':
                cell_content = f"{count} / ({total_pct:.1f}%)"
            else:
                row_pct = tab_row_pct.loc[row_name, col_name]
                cell_content = f"{count} ({row_pct:.1f}%) / ({total_pct:.1f}%)"
            row_data.append(cell_content)
    
    display_tab = pd.DataFrame(display_data, columns=[col1] + col_names).set_index(col1)

    # 3. Stats
    try:
        chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=correction)
        method_name = "Chi-Square"
        if tab_chi2.shape == (2, 2):
            method_name += " (with Yates' Correction)" if correction else " (Pearson Uncorrected)"
        msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
        
        stats_res = {
            "Test": method_name, "Statistic": chi2, "P-value": p, "Degrees of Freedom": dof, "N": len(data)
        }
        
        # 4. Risk Measures
        risk_df = None
        if tab_chi2.shape == (2, 2):
            try:
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                
                row_labels = tab_chi2.index.tolist()
                col_labels = tab_chi2.columns.tolist()
                label_exp = str(row_labels[0])
                label_unexp = str(row_labels[1])
                label_event = str(col_labels[0])
                
                risk_exp = a / (a + b) if (a + b) > 0 else 0
                risk_unexp = c / (c + d) if (c + d) > 0 else 0
                rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
                rd = risk_exp - risk_unexp 
                nnt = abs(1/rd) if rd != 0 else np.inf
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

# --- ROC Functions (คงเดิม) ---
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

def analyze_roc(df, truth_col, score_col, method='delong', pos_label_user=None):
    data = df[[truth_col, score_col]].dropna()
    y_true_raw = data[truth_col]
    y_score = pd.to_numeric(data[score_col], errors='coerce').dropna()
    y_true_raw = y_true_raw.loc[y_score.index]
    
    if y_true_raw.nunique() != 2 or pos_label_user is None:
        return None, "Error: Binary outcome required.", None, None

    y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    n1 = sum(y_true == 1); n0 = sum(y_true == 0)
    
    if method == 'delong': ci_lower, ci_upper, se = auc_ci_delong(y_true, y_score.values); m_name = "DeLong"
    else: ci_lower, ci_upper, se = auc_ci_hanley_mcneil(auc_val, n1, n0); m_name = "Hanley"
    
    p_val_auc = stats.norm.sf(abs((auc_val - 0.5)/se))*2 if se > 0 else 0.0
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

def generate_report(title, elements):
    """
    สร้าง HTML Report พร้อมแก้ไข Bug Header
    """
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; margin: 0; color: #333; }
        .report-container { background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); padding: 20px; width: 100%; box-sizing: border-box; margin-bottom: 20px; }
        h2 { color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        h4 { color: #34495e; margin-top: 25px; margin-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 0.9em; }
        th, td { padding: 10px 15px; border: 1px solid #e0e0e0; vertical-align: top; text-align: left; }
        th { background-color: #f0f2f6; font-weight: 600; }
        tr:nth-child(even) td { background-color: #f9f9f9; }
        .report-table th, .report-table td { text-align: center; } 
        .report-table th:first-child, .report-table td:first-child { text-align: left; }
        .report-footer { text-align: right; font-size: 0.75em; color: #666; margin-top: 20px; border-top: 1px dashed #ddd; padding-top: 10px; }
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
            # 🟢 FIX: header=True
            df_html = data.to_html(index=True, classes='report-table', header=True)
            col_names = data.columns.tolist()
            idx_name = data.index.name if data.index.name else "Variable 1"
            out_name = element.get('outcome_col', 'Outcome')
            h1 = f"<tr><th rowspan='2' class='report-table' style='text-align: left;'>{idx_name}</th><th colspan='{len(col_names)}' class='report-table'>{out_name}</th></tr>"
            h2 = "<tr>" + "".join([f"<th class='report-table'>{c}</th>" for c in col_names]) + "</tr>"
            try:
                table_start = df_html.split('<thead>')[0]
                table_end = df_html.split('</thead>')[1]
                custom_header = f"<thead>{h1}{h2}</thead>"
                html += table_start + custom_header + table_end
            except:
                html += df_html

        elif element_type == 'plot':
            buf = io.BytesIO()
            if isinstance(data, plt.Figure):
                data.savefig(buf, format='png', bbox_inches='tight'); plt.close(data)
                uri = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{uri}" style="max-width: 100%;"/>'
            buf.close()
            
    html += "<div class='report-footer'>&copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit</div></div></body></html>"
    return html
