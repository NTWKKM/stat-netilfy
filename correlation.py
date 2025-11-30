import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import io, base64

def calculate_chi2(df, col1, col2, correction=True):
    """คำนวณ Chi-square พร้อมสร้างตาราง 2 ชั้น และบังคับเรียง 1, 0"""
    if col1 not in df.columns or col2 not in df.columns: 
        return None, None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    
    # 1. สร้างตาราง Crosstab ทั้ง 3 แบบ
    tab_chi2 = pd.crosstab(data[col1], data[col2])
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    tab_row_pct = pd.crosstab(data[col1], data[col2], normalize='index', margins=True, margins_name="Total") * 100
    
    # --- 🟢 FIX: Reorder Columns (Outcome) and Rows (Exposure) to put '1' first ---
    
    # 1. แยก Label หลักออกจาก 'Total'
    all_col_labels = tab_raw.columns.tolist() 
    all_row_labels = tab_raw.index.tolist()
    base_col_labels = [col for col in all_col_labels if col != 'Total']
    base_row_labels = [row for row in all_row_labels if row != 'Total']

    # 2. Custom Sort: เรียงตามค่าตัวเลขมากไปน้อย (เพื่อบังคับให้ 1, 0)
    def custom_sort(label):
        try:
            # พยายามแปลงเป็น float สำหรับการเรียงตัวเลข
            return float(label)
        except (ValueError, TypeError):
            # สำหรับ string หรือค่าที่ไม่ใช่ตัวเลข (ให้ใช้ค่า string นั้น)
            return str(label)

    base_col_labels.sort(key=custom_sort, reverse=True) # เช่น [1, 0]
    base_row_labels.sort(key=custom_sort, reverse=True) # เช่น [1, 0]

    # 3. กำหนดลำดับสุดท้าย (รวม 'Total')
    final_col_order = base_col_labels + ['Total'] 
    final_row_order = base_row_labels + ['Total']

    # 4. Reindex ตารางทั้งหมดตามลำดับใหม่
    tab_raw = tab_raw.reindex(index=final_row_order, columns=final_col_order)
    tab_row_pct = tab_row_pct.reindex(index=final_row_order, columns=final_col_order)
    tab_chi2 = tab_chi2.reindex(index=base_row_labels, columns=base_col_labels) # tab_chi2 ไม่มี 'Total'
    
    # 5. Update iteration lists
    col_names = final_col_order 
    index_names = final_row_order

    # --- 🟢 END REORDERING FIX ---
    
    display_data = []
    
    for row_name in index_names:
        row_data = []
        for col_name in col_names:
            count = tab_raw.loc[row_name, col_name]
            
            # แก้ KeyError: 'Total'
            if col_name == 'Total':
                pct = 100.0
            else:
                pct = tab_row_pct.loc[row_name, col_name]
                
            cell_content = f"{count} ({pct:.1f}%)"
            row_data.append(cell_content)
            
        display_data.append(row_data)
    
    display_tab = pd.DataFrame(display_data, columns=col_names, index=index_names)
    display_tab.index.name = col1
    
    # 3. คำนวณ Chi-square Stats
    try:
        chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=correction)
        
        method_name = "Chi-Square"
        if tab_chi2.shape == (2, 2):
            method_name += " (with Yates' Correction)" if correction else " (Pearson Uncorrected)"
            
        msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
        
        stats_res = {
            "Test": method_name, "Statistic": chi2, "P-value": p, "Degrees of Freedom": dof, "N": len(data)
        }
        
        # 4. Risk Calculation (ใช้ tab_chi2 ที่ reindex แล้ว)
        risk_df = None
        if tab_chi2.shape == (2, 2):
            try:
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                
                # Label ถูกดึงมาอย่างถูกลำดับแล้ว (1 ก่อน 0)
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

def calculate_correlation(df, col1, col2, method='pearson'):
    # (ส่วนนี้คงเดิม)
    if col1 not in df.columns or col2 not in df.columns: return None, "Columns not found", None
    data = df[[col1, col2]].dropna()
    try:
        v1 = pd.to_numeric(data[col1], errors='raise'); v2 = pd.to_numeric(data[col2], errors='raise')
    except: return None, f"Error: Numeric required.", None
    if method == 'pearson': corr, p = stats.pearsonr(v1, v2); name="Pearson"; desc="Linear"
    else: corr, p = stats.spearmanr(v1, v2); name="Spearman"; desc="Monotonic"
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(v1, v2, alpha=0.6, edgecolors='w', s=50)
    if method == 'pearson':
        try: m, b = np.polyfit(v1, v2, 1); ax.plot(v1, m*v1 + b, 'r--', alpha=0.8)
        except: pass
    ax.set_xlabel(col1); ax.set_ylabel(col2); ax.set_title(f"{col1} vs {col2}"); ax.grid(True, alpha=0.3)
    return {"Method": name, "Coefficient": corr, "P-value": p, "N": len(data)}, None, fig

def generate_report(title, elements):
    # (ส่วนนี้คงเดิม เพราะ Layout ถูกต้องแล้ว)
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
            
            # --- Header Row 1 ---
            html_tab += "<thead><tr>"
            html_tab += "<th style='background-color: white; border: none;'></th>" 
            html_tab += f"<th colspan='{len(col_labels)}' class='th-outcome'>{out_name}</th>"
            html_tab += "</tr>"
            
            # --- Header Row 2 ---
            html_tab += "<tr>"
            html_tab += f"<th class='th-exposure'>{exp_name}</th>"
            for label in col_labels:
                html_tab += f"<th>{label}</th>"
            html_tab += "</tr></thead>"
            
            # --- Body (Rows 3-5) ---
            html_tab += "<tbody>"
            for idx_label, row in data.iterrows():
                html_tab += "<tr>"
                html_tab += f"<td class='td-label'>{idx_label}</td>"
                
                for val in row:
                    html_tab += f"<td>{val}</td>"
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
            
    html += "<div class='report-footer'>&copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="color: inherit; text-decoration: none; font-weight: 700; border-bottom: 1px dotted var(--text-muted);">NTWKKM</a>. All Rights Reserved.| Powered by GitHub, Gemini, Streamlit</div></div></body></html>"
    return html
