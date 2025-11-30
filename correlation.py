import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import io, base64

def calculate_chi2(df, col1, col2, correction=True):
    """คำนวณ Chi-square พร้อม Risk Ratio สำหรับตาราง 2x2"""
    if col1 not in df.columns or col2 not in df.columns: 
        return None, None, "Columns not found"
    
    data = df[[col1, col2]].dropna()
    # สร้างตารางไขว้ (Crosstab)
    tab = pd.crosstab(data[col1], data[col2])
    
    try:
        chi2, p, dof, ex = stats.chi2_contingency(tab, correction=correction)
        
        method_name = "Chi-Square"
        if tab.shape == (2, 2):
            method_name += " (with Yates' Correction)" if correction else " (Pearson Uncorrected)"
        
        msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
        
        # เก็บค่าพื้นฐาน
        stats_res = {
            "Test": method_name,
            "Statistic": chi2,
            "P-value": p,
            "Degrees of Freedom": dof,
            "N": len(data)
        }

        # 🟢 เพิ่มการคำนวณ Risk Ratio (RR) / Odds Ratio (OR) สำหรับ 2x2
        if tab.shape == (2, 2):
            try:
                # แปลงเป็น numpy array เพื่อเข้าถึง index ง่ายๆ [[a, b], [c, d]]
                # a=Exposed+,Event+ | b=Exposed+,Event-
                # c=Exposed-,Event+ | d=Exposed-,Event-
                # หมายเหตุ: การเรียงลำดับขึ้นอยู่กับค่าของข้อมูล (เช่น 0,1 หรือ No,Yes)
                vals = tab.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                
                # Odds Ratio
                odd_ratio, p_or = stats.fisher_exact(tab)
                stats_res["Odds Ratio (OR)"] = odd_ratio
                
                # Risk Ratio (RR) = [a/(a+b)] / [c/(c+d)]
                risk_exposed = a / (a + b) if (a + b) > 0 else 0
                risk_unexposed = c / (c + d) if (c + d) > 0 else 0
                
                if risk_unexposed > 0:
                    rr = risk_exposed / risk_unexposed
                    stats_res["Risk Ratio (RR)"] = rr
                    
                    # Absolute Risk Reduction (ARR) & NNT
                    arr = risk_exposed - risk_unexposed # หรือ risk_unexposed - risk_exposed ขึ้นอยู่กับบริบท
                    stats_res["Risk Difference (RD)"] = arr
                    if arr != 0:
                        stats_res["NNT"] = abs(1 / arr)
                    else:
                        stats_res["NNT"] = np.inf
            except Exception as e:
                stats_res["Risk Calc Error"] = str(e)

        return tab, stats_res, msg
    except Exception as e:
        return tab, None, str(e)

def calculate_correlation(df, col1, col2, method='pearson'):
    # ... (ส่วนนี้เหมือนเดิม ใช้โค้ดเดิมได้เลยครับ) ...
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
    # ... (ส่วนนี้เหมือนเดิม ใช้โค้ดเดิมได้เลยครับ) ...
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
        if element.get('header'): html += f"<h4>{element['header']}</h4>"
        if element['type'] == 'text': 
            html += f"<p>{element['data']}</p>"
        elif element['type'] == 'table': 
            html += element['data'].to_html(index=True, classes='report-table')
        elif element['type'] == 'plot':
            buf = io.BytesIO()
            if isinstance(element['data'], plt.Figure):
                element['data'].savefig(buf, format='png', bbox_inches='tight')
                plt.close(element['data'])
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
