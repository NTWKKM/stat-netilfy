import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
import io
import base64

# --- Helper: Clean Data ---
def clean_survival_data(df, time_col, event_col, covariates=None):
    """
    เตรียมข้อมูลสำหรับ Survival Analysis
    - ลบแถวที่มี Missing Value ในคอลัมน์ที่เลือก
    - แปลงค่าให้เป็น Numeric
    """
    cols = [time_col, event_col]
    if covariates:
        cols += covariates
    
    # เลือกเฉพาะคอลัมน์ที่ใช้
    data = df[cols].copy()
    
    # แปลงเป็นตัวเลข (Coerce errors to NaN)
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
        
    # ลบ NaN
    data = data.dropna()
    return data

# --- 1. Kaplan-Meier & Log-Rank ---
def fit_km_logrank(df, time_col, event_col, group_col=None):
    """
    สร้างกราฟ Kaplan-Meier และคำนวณ Log-Rank Test
    """
    data = clean_survival_data(df, time_col, event_col, [group_col] if group_col else [])
    
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stats_res = {}
    
    if group_col:
        # กรณีมีกลุ่มเปรียบเทียบ (Multivariate Plot)
        groups = data[group_col].unique()
        T_list, E_list, labels = [], [], []
        
        for i, g in enumerate(groups):
            mask = data[group_col] == g
            # Fit K-M สำหรับแต่ละกลุ่ม
            kmf.fit(data.loc[mask, time_col], event_observed=data.loc[mask, event_col], label=str(g))
            kmf.plot_survival_function(ax=ax, ci_show=False) # ไม่โชว์ CI เพื่อความสะอาดหากมีหลายเส้น
            
            # เก็บข้อมูลสำหรับ Log-Rank
            T_list.append(data.loc[mask, time_col])
            E_list.append(data.loc[mask, event_col])
            labels.append(str(g))
            
            # เก็บ Median Survival
            stats_res[f"Median Survival ({g})"] = kmf.median_survival_time_
            
        # Log-Rank Test (เปรียบเทียบกลุ่มแรกกับกลุ่มอื่นๆ หรือ pairwise - ที่นี่ทำแบบรวม)
        # หมายเหตุ: lifelines logrank_test เปรียบเทียบได้ทีละ 2 กลุ่มหลักๆ 
        # หรือใช้ multivariate_logrank_test สำหรับ >2 กลุ่ม (แต่ในที่นี้ทำ simple case 2 กลุ่มก่อน)
        if len(groups) == 2:
            lr_result = logrank_test(T_list[0], T_list[1], event_observed_A=E_list[0], event_observed_B=E_list[1])
            stats_res['Log-Rank p-value'] = lr_result.p_value
            ax.set_title(f"KM Curve: {group_col} (p = {lr_result.p_value:.4f})")
        else:
             ax.set_title(f"KM Curve: {group_col}")
             
    else:
        # กรณีไม่มีกลุ่ม (Univariate Plot)
        kmf.fit(data[time_col], event_observed=data[event_col], label="All")
        kmf.plot_survival_function(ax=ax)
        stats_res["Median Survival"] = kmf.median_survival_time_
        ax.set_title("Kaplan-Meier Survival Curve")
        
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Survival Probability")
    ax.grid(True, alpha=0.3)
    
    return fig, pd.DataFrame(stats_res, index=["Value"]).T
# --- 🟢 2. Nelson-Aalen (Cumulative Hazard) ---
def fit_nelson_aalen(df, time_col, event_col, group_col=None):
    data = clean_survival_data(df, time_col, event_col, [group_col] if group_col else [])
    naf = NelsonAalenFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if group_col:
        groups = data[group_col].unique()
        for g in groups:
            mask = data[group_col] == g
            naf.fit(data.loc[mask, time_col], event_observed=data.loc[mask, event_col], label=str(g))
            naf.plot_cumulative_hazard(ax=ax)
        ax.set_title(f"Nelson-Aalen Cumulative Hazard: {group_col}")
    else:
        naf.fit(data[time_col], event_observed=data[event_col], label="All")
        naf.plot_cumulative_hazard(ax=ax)
        ax.set_title("Nelson-Aalen Cumulative Hazard Curve")
        
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Cumulative Hazard")
    ax.grid(True, alpha=0.3)
    return fig
    
# --- 3. Cox Proportional Hazards Model ---
def fit_cox_ph(df, time_col, event_col, covariates):
    """
    วิเคราะห์ Cox Regression และตรวจสอบ Assumption
    """
    data = clean_survival_data(df, time_col, event_col, covariates)
    
    cph = CoxPHFitter()
    try:
        # Fit Model
        cph.fit(data, duration_col=time_col, event_col=event_col)
        
        # ดึงผลลัพธ์ (Summary)
        summary_df = cph.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
        summary_df.columns = ['Coef', 'HR', 'Lower 95% CI', 'Upper 95% CI', 'P-value']
        
        return cph, summary_df, None
    except Exception as e:
        return None, None, str(e)

# --- 4. Generate Report (Format เดิมของ Project) ---
def generate_report_survival(title, elements):
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; }
        .report-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h2 { border-bottom: 2px solid #34495e; padding-bottom: 10px; color: #2c3e50; }
        h4 { color: #2980b9; margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .report-footer { text-align: right; font-size: 0.8em; color: #777; margin-top: 30px; border-top: 1px dashed #ccc; padding-top: 10px; }
    </style>
    """
    
    html = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html += f"<div class='report-container'><h2>{title}</h2>"
    
    for el in elements:
        if el['type'] == 'text':
            html += f"<p>{el['data']}</p>"
        elif el['type'] == 'header':
             html += f"<h4>{el['data']}</h4>"
        elif el['type'] == 'table':
            html += el['data'].to_html(classes='table')
        elif el['type'] == 'plot':
            buf = io.BytesIO()
            el['data'].savefig(buf, format='png', bbox_inches='tight')
            plt.close(el['data'])
            uri = base64.b64encode(buf.getvalue()).decode('utf-8')
            html += f'<img src="data:image/png;base64,{uri}" style="max-width:100%;"/>'
            
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""
    return html
