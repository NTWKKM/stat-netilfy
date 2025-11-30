import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import io, base64

def calculate_chi2(df, col1, col2, correction=True):
    """คำนวณ Chi-square พร้อมสร้างตาราง Display (Count/%) และตาราง Risk Interpretation"""
    if col1 not in df.columns or col2 not in df.columns: 
        return None, None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    
    # 1. Contingency Table (Frequency Count for Chi2 calculation)
    tab_chi2 = pd.crosstab(data[col1], data[col2])
    
    # 2. สร้างตาราง Display (Format: Count (Row%) / (Total%))
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

    # 3. คำนวณ Chi-square Stats
    try:
        chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=correction)
        
        method_name = "Chi-Square"
        if tab_chi2.shape == (2, 2):
            method_name += " (with Yates' Correction)" if correction else " (Pearson Uncorrected)"
            
        msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
        
        stats_res = {
            "Test": method_name,
            "Statistic": chi2,
            "P-value": p,
            "Degrees of Freedom": dof,
            "N": len(data)
        }
        
        # 🟢 4. สร้างตาราง Risk Measures พร้อม Interpretation (เลียนแบบ old_diag_test.py)
        risk_df = None
        if tab_chi2.shape == (2, 2):
            try:
                # ดึง Label เพื่อมาทำ Interpretation
                # Assumption: Row 0=Exposed, Row 1=Unexposed | Col 0=Event, Col 1=No Event
                # หมายเหตุ: การเรียงลำดับขึ้นอยู่กับข้อมูล (เช่น 0, 1 หรือ Yes, No) 
                # ควรตรวจสอบลำดับข้อมูลจริง แต่ในที่นี้จะยึดตาม Index 0/1
                
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                
                row_labels = tab_chi2.index.tolist()
                col_labels = tab_chi2.columns.tolist()
                
                label_exp = str(row_labels[0])   # R1 Group
                label_unexp = str(row_labels[1]) # R0 Group
                label_event = str(col_labels[0]) # Event outcome
                
                # Calculations
                risk_exp = a / (a + b) if (a + b) > 0 else 0
                risk_unexp = c / (c + d) if (c + d) > 0 else 0
                
                rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
                rd = risk_exp - risk_unexp # Absolute Risk Reduction (ARR) if negative
                nnt = abs(1/rd) if rd != 0 else np.inf
                
                odd_ratio, _ = stats.fisher_exact(tab_chi2)
                
                # สร้างตาราง DataFrame
                risk_data = [
                    {
                        "Statistic": f"Risk in {label_exp} (R1)",
                        "Value": f"{risk_exp:.4f}",
                        "Interpretation": f"Risk of '{label_event}' in group {label_exp}"
                    },
                    {
                        "Statistic": f"Risk in {label_unexp} (R0)",
                        "Value": f"{risk_unexp:.4f}",
                        "Interpretation": f"Baseline Risk of '{label_event}' in group {label_unexp}"
                    },
                    {
                        "Statistic": "Risk Ratio (RR)",
                        "Value": f"{rr:.4f}",
                        "Interpretation": f"Risk in {label_exp} is {rr:.2f} times that of {label_unexp}"
                    },
                    {
                        "Statistic": "Risk Difference (RD)",
                        "Value": f"{rd:.4f}",
                        "Interpretation": f"Absolute difference (R1 - R0)"
                    },
                    {
                        "Statistic": "Number Needed to Treat (NNT)",
                        "Value": f"{nnt:.1f}",
                        "Interpretation": "Patients to treat to prevent/cause 1 outcome"
                    },
                    {
                        "Statistic": "Odds Ratio (OR)",
                        "Value": f"{odd_ratio:.4f}",
                        "Interpretation": "Odds of Event (Exp vs Unexp)"
                    }
                ]
                risk_df = pd.DataFrame(risk_data)
                
            except Exception as e:
                pass # ถ้าคำนวณไม่ได้ (เช่น หารศูนย์) ก็ปล่อย risk_df เป็น None

        return display_tab, stats_res, msg, risk_df # Return 4 ค่า

    except Exception as e:
        return display_tab, None, str(e), None

def calculate_correlation(df, col1, col2, method='pearson'):
    # ... (ส่วนนี้คงเดิม) ...
    if col1 not in df.columns or col2 not in df.columns:
        return None, "Columns not found", None

    data = df[[col1, col2]].dropna()
    try:
        v1 = pd.to_numeric(data[col1], errors='raise')
        v2 = pd.to_numeric(data[col2], errors='raise')
    except:
        return None, f"Error: Both '{col1}' and '{col2}' must be numeric variables.", None

    if method == 'pearson':
        corr, p = stats.pearsonr(v1, v2)
        name = "Pearson Correlation (r)"
        desc = "Linear Relationship"
    else:
        corr, p = stats.spearmanr(v1, v2)
        name = "Spearman Correlation (rho)"
        desc = "Monotonic Relationship"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(v1, v2, alpha=0.6, edgecolors='w', s=50)
    
    if method == 'pearson':
        try:
            m, b = np.polyfit(v1, v2, 1)
            ax.plot(v1, m*v1 + b, color='red', linestyle='--', alpha=0.8, label='Linear Fit')
            ax.legend()
        except: pass
        
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Scatter Plot: {col1} vs {col2}")
    ax.grid(True, alpha=0.3)
    
    stats_res = {
        "Method": name,
        "Type": desc,
        "Coefficient": corr,
        "P-value": p,
        "N": len(data)
    }
    return stats_res, None, fig

def generate_report(title, elements):
    # ... (ส่วนนี้เหมือนเดิม รองรับ contingency_table) ...
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
        .report-footer {
            text-align: right;
            font-size: 0.75em;
            color: #666;
            margin-top: 20px;
            border-top: 1px dashed #ddd;
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
        
        if header: html += f"<h4>{header}</h4>"
            
        if element_type == 'text':
            html += f"<p>{data}</p>"
        elif element_type == 'table':
            # ใช้ to_html ปกติสำหรับตารางทั่วไป (เช่น stats, risk_df)
            # ไม่เอา index ถ้าเป็น risk_df (ดูสะอาดกว่า)
            idx_flag = True
            if 'Statistic' in data.columns and 'Interpretation' in data.columns: idx_flag = False
            html += data.to_html(index=idx_flag, classes='report-table')
            
        # 🟢 รองรับ Contingency Table Header แบบ 2 ชั้น
        elif element_type == 'contingency_table':
            df_html = data.to_html(index=True, classes='report-table', header=False)
            col_names_raw = data.columns.tolist()
            index_name = data.index.name
            outcome_col_name = element.get('outcome_col', 'Outcome')
            
            # Row 1
            header_row1 = "<tr>"
            header_row1 += f"<th rowspan='2' class='report-table' style='text-align: left;'>{index_name}</th>"
            header_row1 += f"<th colspan='{len(col_names_raw)}' class='report-table'>{outcome_col_name}</th>" 
            header_row1 += "</tr>"
            
            # Row 2
            header_row2 = "<tr>"
            for col_name in col_names_raw:
                 header_row2 += f"<th class='report-table'>{col_name}</th>"
            header_row2 += "</tr>"
            
            table_start_tag = df_html.split('<thead>')[0]
            table_end_tag = df_html.split('</thead>')[1]
            custom_header = f"<thead>{header_row1}{header_row2}</thead>"
            html += table_start_tag + custom_header + table_end_tag

        elif element_type == 'plot':
            buf = io.BytesIO()
            if isinstance(data, plt.Figure):
                data.savefig(buf, format='png', bbox_inches='tight')
                plt.close(data)
                data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{data_uri}" style="max-width: 100%;"/>'
            buf.close()
            
    html += """
    <div class="report-footer">
      &copy; 2025 NTWKKM | Powered by GitHub, Gemini, Streamlit
    </div>
    """
    html += "</div></body></html>"
    return html
