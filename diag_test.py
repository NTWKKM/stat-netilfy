# diag_test.py
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import io, base64 # เพิ่ม io และ base64 สำหรับการฝังรูปใน HTML

def calculate_descriptive(df, col):
    """คำนวณสถิติพื้นฐาน"""
    if col not in df.columns: return "Column not found"
    
    data = df[col].dropna()
    try:
        # พยายามแปลงเป็นตัวเลข
        num_data = pd.to_numeric(data, errors='raise')
        is_numeric = True
    except:
        is_numeric = False
        
    if is_numeric:
        desc = num_data.describe()
        return pd.DataFrame({
            "Statistic": ["Count", "Mean", "SD", "Median", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
            "Value": [
                f"{desc['count']:.0f}",
                f"{desc['mean']:.4f}",
                f"{desc['std']:.4f}",
                f"{desc['50%']:.4f}",
                f"{desc['min']:.4f}",
                f"{desc['max']:.4f}",
                f"{desc['25%']:.4f}",
                f"{desc['75%']:.4f}"
            ]
        })
    else:
        # Categorical
        counts = data.value_counts()
        percent = data.value_counts(normalize=True) * 100
        return pd.DataFrame({
            "Category": counts.index,
            "Count": counts.values,
            "Percentage (%)": percent.values
        }).sort_values("Count", ascending=False)

def calculate_chi2(df, col1, col2, correction=False):
    """คำนวณ Chi-square พร้อมทั้ง RR, ARR, NNT และสร้างตารางแสดงผลที่มี Count และ Percent"""
    if col1 not in df.columns or col2 not in df.columns: 
        return None, {"error": "Columns not found"}
    
    data = df[[col1, col2]].dropna()
    N_total = len(data)
    
    # Contingency Table (Frequency Count)
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    
    # --- Chi-square Calculation ---
    results = {}
    
    # ต้องแยก Chi-square ออกจาก margins
    tab_for_chi2 = pd.crosstab(data[col1], data[col2]) 
    
    try:
        # Pass correction parameter to chi2_contingency
        chi2, p, dof, ex = stats.chi2_contingency(tab_for_chi2, correction=correction)

        # Determine the name of the test used for the message
        test_name = "Chi-square (Pearson)"
        if correction:
            test_name = "Chi-square (Yates' corrected)"

        # Update chi2_msg to include the test name
        results['chi2_msg'] = f"{test_name} statistic: {chi2:.4f}, p-value: {p:.4f}, df: {dof}"
        results['p_value'] = p
        
    except Exception as e:
        
        results['chi2_msg'] = f"Chi-square calculation error: {str(e)}"
        
    # --- RR / ARR / NNT Calculation (Only for 2x2 table) ---
    if tab_for_chi2.shape == (2, 2):
        # Assumption: Row 0 = Exposed, Row 1 = Unexposed, Col 0 = Event (Positive)
        tab_arr = tab_for_chi2.values 
        a, b = tab_arr[0, 0], tab_arr[0, 1] 
        c, d = tab_arr[1, 0], tab_arr[1, 1]
        
        N_exp = a + b 
        N_unexp = c + d 
        
        # Calculate Risk
        R_exp = a / N_exp if N_exp > 0 else 0 # Risk in Exposed (R1)
        R_unexp = c / N_unexp if N_unexp > 0 else 0 # Risk in Unexposed (R0)
        
        if N_exp > 0 and N_unexp > 0:
            # RR
            RR = R_exp / R_unexp if R_unexp > 0 else np.inf
            
            # Risk Difference (RD) / Absolute Risk Reduction (ARR)
            RD = R_exp - R_unexp
            
            # NNT = 1 / |RD|
            NNT = 1 / abs(RD) if RD != 0 and abs(RD) <= 1 else np.inf
            
            # Odds Ratio (for reference)
            OR = (a * d) / (b * c) if b != 0 and c != 0 else np.inf
            
            results['Is_2x2'] = True
            results['RR'] = RR
            results['RD'] = RD
            results['NNT'] = NNT
            results['OR'] = OR
            results['R_exp'] = R_exp
            results['R_unexp'] = R_unexp
            results['R_exp_label'] = tab_for_chi2.index[0]
            results['R_unexp_label'] = tab_for_chi2.index[1]
            results['Event_label'] = tab_for_chi2.columns[0]
        else:
            results['RR'] = np.nan
            results['NNT'] = np.nan
            results['RD'] = np.nan
            results['Is_2x2'] = True
            
    # --- Formatting Display Table (Count, Row %, Total %) ---
    
    # 1. Calculate Row Percentages (Horizontal %)
    tab_row_percent = pd.crosstab(data[col1], data[col2], normalize='index', margins=True, margins_name="Total") * 100
    
    # 2. Calculate Total Percentages (Grand Total %)
    tab_total_percent = pd.crosstab(data[col1], data[col2], normalize='all', margins=True, margins_name="Total") * 100
    
    # สร้างตาราง Display Final
    col_names = tab_raw.columns.tolist() 
    index_names = tab_raw.index.tolist() 
    
    display_tab_data = []
    
    for row_name in index_names:
        row_data = [] 
        
        for col_name in col_names:
            count = tab_raw.loc[row_name, col_name]
            total_pct = tab_total_percent.loc[row_name, col_name]

            if col_name == 'Total' and row_name == 'Total':
                # Grand Total Cell: แสดงเฉพาะ Count และ Total %
                row_pct = 100.0 
                cell_content = f"{count} / ({total_pct:.1f}%)"
            elif col_name == 'Total':
                # Row Marginal Total Cell: แสดง Count และ Total %
                row_pct = 100.0
                cell_content = f"{count} ({row_pct:.1f}%) / ({total_pct:.1f}%)"
            else:
                # Normal Data Cell: แสดง Count, Row %, Total %
                row_pct = tab_row_percent.loc[row_name, col_name]
                cell_content = f"{count} ({row_pct:.1f}%) / ({total_pct:.1f}%)"
            
            row_data.append(cell_content)
            
        display_tab_data.append([row_name] + row_data) 

    # สร้าง DataFrame สำหรับแสดงผล
    display_tab = pd.DataFrame(display_tab_data, columns=[col1] + col_names)
    display_tab = display_tab.set_index(col1)
        
    return display_tab, results
    
# --- ROC & AUC FUNCTIONS ---

def auc_ci_hanley_mcneil(auc, n1, n2):
    """
    Hanley & McNeil (1982) method for AUC Variance (Parametric/Binomial assumption)
    n1: positive cases, n2: negative cases
    """
    q1 = auc / (2 - auc)
    q2 = 2 * (auc**2) / (1 + auc)
    
    se_auc = np.sqrt(((auc * (1 - auc)) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1 * n2))
    lower = auc - 1.96 * se_auc
    upper = auc + 1.96 * se_auc
    return lower, upper, se_auc

def auc_ci_delong(y_true, y_scores):
    """
    DeLong et al. (1988) method for AUC Variance (Non-parametric)
    Ref: Fast implementation logic
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Sort by score
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    n_pos = tps[-1]
    n_neg = fps[-1]
    
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan # Cannot calc
    
    auc = roc_auc_score(y_true, y_scores)
    
    # DeLong Covariance Calculation
    # Compute V10 (X) and V01 (Y)
    
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    
    # V10: For each positive, what fraction of negatives is it greater than?
    v10 = []
    for p in pos_scores:
        v10.append( (np.sum(p > neg_scores) + 0.5*np.sum(p == neg_scores)) / n_neg )
    v10 = np.array(v10)
    
    # V01: For each negative, what fraction of positives is it smaller than?
    v01 = []
    for n in neg_scores:
        v01.append( (np.sum(pos_scores > n) + 0.5*np.sum(pos_scores == n)) / n_pos )
    v01 = np.array(v01)
    
    # Variance
    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    
    var_auc = (s10 / n_pos) + (s01 / n_neg)
    se_auc = np.sqrt(var_auc)
    
    return auc - 1.96*se_auc, auc + 1.96*se_auc, se_auc


def analyze_roc(df, truth_col, score_col, method='delong', pos_label_user=None):
    """Main ROC Analysis"""
    data = df[[truth_col, score_col]].dropna()
    y_true_raw = data[truth_col]
    y_score = pd.to_numeric(data[score_col], errors='coerce').dropna()
    # Align indices
    y_true_raw = y_true_raw.loc[y_score.index]
    
    unique_vals = y_true_raw.nunique()
    if unique_vals != 2:
        return None, "Outcome must have exactly 2 classes.", None, None 

    # 🟢 START: Manual Encoding based on user input (Overriding LabelEncoder)
    if pos_label_user is None:
        return None, "Positive label (pos_label) must be specified for binary outcome.", None, None
        
    all_labels_raw = [str(x) for x in y_true_raw.unique()]
    neg_label_raw = [lab for lab in all_labels_raw if lab != pos_label_user][0]
    
    # แปลงเป็น 0/1
    y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)
    
    # Cast y_true back to pd.Series for alignment/indexing safety
    y_true = pd.Series(y_true, index=y_true_raw.index)
    # 🟢 END: Manual Encoding
        
    # 1. Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    
    n1 = sum(y_true == 1)
    n0 = sum(y_true == 0)
    
    # 2. Calculate CI
    if method == 'delong':
        ci_lower, ci_upper, se = auc_ci_delong(y_true.values, y_score.values)
        method_name = "DeLong et al."
    else:
        # Binomial Exact / Hanley McNeil
        ci_lower, ci_upper, se = auc_ci_hanley_mcneil(auc_val, n1, n0)
        method_name = "Hanley & McNeil (Parametric/Binomial)"
        
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)

    # 🟢 START: Calculate P-value for AUC (H0: AUC = 0.5)
    p_value_auc = np.nan
    try:
        if se > 0:
            # Z = (AUC - 0.5) / SE
            Z_score = (auc_val - 0.5) / se
            # Two-tailed P-value from Standard Normal Distribution
            p_value_auc = stats.norm.sf(abs(Z_score)) * 2 
        else:
            # Perfect separation (AUC=1 or 0), P-value is effectively 0
            p_value_auc = 0.0 
    except:
        p_value_auc = np.nan
    # 🟢 END: Calculate P-value for AUC
    
    # 3. Youden Index
    # J = Sensitivity + Specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    youden_j = j_scores[best_idx]
    best_thresh = thresholds[best_idx]
    best_sens = tpr[best_idx]
    best_spec = 1 - fpr[best_idx]
    
    stats_res = {
        "AUC": auc_val,
        "SE": se,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "Method": method_name,
        "P-value (H0: AUC=0.5)": p_value_auc, # 🟢 ADDED P-VALUE HERE
        "Youden Index (J)": youden_j,
        "Best Cut-off": best_thresh,
        "Sensitivity": best_sens,
        "Specificity": best_spec,
        "N (Positive)": n1,
        "N (Negative)": n0,
        "Positive Label": pos_label_user, 
        "Negative Label": neg_label_raw    
    }
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark Youden point
    ax.plot(1-best_spec, best_sens, 'ro', label=f'Best Cut-off ({best_thresh:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(f'ROC Curve: {score_col} vs {truth_col}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # 5. Create Coordinates DataFrame for detailed table (เพิ่มส่วนนี้)
    coords_df = pd.DataFrame({
        'Threshold': thresholds,
        'Sensitivity (TPR)': tpr,
        'Specificity': 1 - fpr,
        '1 - Specificity (FPR)': fpr,
        'Youden J': tpr - fpr
    }).sort_values('Threshold', ascending=False).reset_index(drop=True)
    coords_df['Threshold'] = coords_df['Threshold'].round(4)
    coords_df = coords_df.round(4)
    
    return stats_res, None, fig, coords_df # แก้ไขให้ return 4 ค่า

def generate_report(title, elements):
    """Generates a simple HTML report based on a list of elements (text, plot, table).
    ใช้สำหรับแสดงผลลัพธ์ของ ROC, Chi-Square และ Descriptive
    """
    
    # --- CSS Styling (Fixed to use Streamlit CSS variables for theme compatibility) ---
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: var(--secondary-background-color); margin: 0; color: var(--text-color); }
        .report-container { 
            background: var(--background-color); 
            border-radius: 8px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
            padding: 20px;
            width: 100%; 
            box-sizing: border-box;
            margin-bottom: 20px;
        }
        h2 { color: var(--primary-color); border-bottom: 2px solid var(--border-color); padding-bottom: 10px; }
        h4 { color: var(--text-color); margin-top: 25px; margin-bottom: 10px; }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: 'Segoe UI', sans-serif; 
            font-size: 0.9em;
        }
        th, td { 
            padding: 10px 15px; 
            border: 1px solid var(--border-color);
            vertical-align: top;
            text-align: left;
        }
        th {
            background-color: var(--primary-color); 
            color: var(--text-color-inverted);
            font-weight: 600;
        }
        tr:nth-child(even) td { background-color: var(--secondary-background-color); }
        .alert { background-color: var(--secondary-background-color); color: var(--warning-color); padding: 10px; border: 1px solid var(--border-color); border-radius: 5px; margin-bottom: 15px; }
        
        .report-table th, .report-table td { text-align: center; } 
        .report-table th:first-child, .report-table td:first-child { text-align: left; }
        
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
    html += f"<div class='report-container'><h2>{title}</h2>"
    
    for element in elements:
        element_type = element['type']
        data = element['data']
        header = element.get('header', '')
        
        if header:
            html += f"<h4>{header}</h4>"
            
        if element_type == 'text':
            html += f"<p>{data}</p>"
        elif element_type == 'table':
            # Handle standard table (e.g., Key Statistics, Descriptive)
            # FIX: ใช้ 'in data.columns' แทน data.columns.contains()
            include_index = not ('Category' in data.columns) and not ('Statistic' in data.columns)
            html += data.to_html(index=include_index, classes='report-table')
            
        # 🟢 NEW: Handle Contingency Table with Two-Level Header
        elif element_type == 'contingency_table':
            # DataFrame should have V1 as index and V2 levels + 'Total' as columns
            df_html = data.to_html(index=True, classes='report-table', header=False) # สร้าง HTML โดยไม่มี Header
            
            # 1. ดึงชื่อคอลัมน์ (0, 1, Total) และชื่อ Index (Exposure)
            col_names_raw = data.columns.tolist()
            index_name = data.index.name
            outcome_col_name = element.get('outcome_col', 'Outcome')
            
            # 2. สร้าง Header สองชั้นด้วย HTML
            
            # Row 1: Merge cells for Outcome_Disease and Total
            # (First TH is for Exposure/Row Variable)
            header_row1 = f"<tr>"
            header_row1 += f"<th rowspan='2' class='report-table' style='text-align: left;'>{index_name}</th>" # ชื่อ Exposure/V1
            # Th ที่เหลือ Merge Cells เพื่อแสดงชื่อ Outcome (รวม Total ด้วย)
            header_row1 += f"<th colspan='{len(col_names_raw)}' class='report-table'>{outcome_col_name}</th>" 
            header_row1 += f"</tr>"
            
            # Row 2: Actual Outcome Levels (0, 1, Total)
            header_row2 = f"<tr>"
            for col_name in col_names_raw:
                 # แสดงชื่อระดับ (0, 1, Total)
                 header_row2 += f"<th class='report-table'>{col_name}</th>"
            header_row2 += f"</tr>"
            
            # 3. นำ Header ที่สร้างเองไปใส่ในตาราง HTML
            # ค้นหาและแทนที่ส่วน <thead>
            table_start_tag = df_html.split('<thead>')[0]
            table_end_tag = df_html.split('</thead>')[1]
            
            custom_header = f"<thead>{header_row1}{header_row2}</thead>"
            
            html += table_start_tag + custom_header + table_end_tag


        elif element_type == 'plot':
            # Save matplotlib figure to a string buffer and convert to base64 for embedding
            buf = io.BytesIO()
            if isinstance(data, plt.Figure):
                data.savefig(buf, format='png')
                plt.close(data) # Close the figure to free memory
                data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{data_uri}" style="max-width: 100%; height: auto; display: block; margin: 15px auto;"/>'
            else:
                 html += '<p class="alert">⚠️ Plot data is not a valid Matplotlib Figure object.</p>'
        
    # 🟢 NEW: เพิ่ม Footer ของ Report
    html += """
    <div class="report-footer">
      &copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit
    </div>
    """
    
    html += "</div></body></html>"
    return html
