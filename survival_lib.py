import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
import io
import base64
import streamlit as st # 🟢 1. IMPORT STREAMLIT

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
@st.cache_data(show_spinner=False) # 🟢 2. ADD CACHE
def fit_km_logrank(df, time_col, event_col, group_col=None):
    """
    สร้างกราฟ Kaplan-Meier และคำนวณ Log-Rank Test พร้อมสรุป N/Events
    """
    data = clean_survival_data(df, time_col, event_col, [group_col] if group_col else [])
    
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stats_res = {} # จะเก็บข้อมูลสำหรับแสดงในตาราง
    
    if group_col:
        # --- กรณีมีกลุ่มเปรียบเทียบ ---
        groups = sorted(data[group_col].unique())
        T_list, E_list, labels = [], [], []
        
        for g in groups:
            mask = data[group_col] == g
            T = data.loc[mask, time_col]
            E = data.loc[mask, event_col]
            
            # Fit K-M
            kmf.fit(T, event_observed=E, label=str(g))
            kmf.plot_survival_function(ax=ax, ci_show=False)
            
            # 🟢 เพิ่ม: เก็บข้อมูลสถิติพื้นฐานลง Dictionary
            n_total = len(T)
            n_events = E.sum()
            n_censored = n_total - n_events
            median_surv = kmf.median_survival_time_
            
            # สร้าง Key ชื่อยาวๆ ให้ชัดเจน
            stats_res[f"{g} (N)"] = n_total
            stats_res[f"{g} (Events)"] = n_events
            stats_res[f"{g} (Median Time)"] = median_surv
            
            T_list.append(T)
            E_list.append(E)
            
        # Log-Rank Test
        if len(groups) == 2:
            lr_result = logrank_test(T_list[0], T_list[1], event_observed_A=E_list[0], event_observed_B=E_list[1])
            stats_res['Log-Rank p-value'] = lr_result.p_value
            ax.set_title(f"KM Curve: {group_col} (p = {lr_result.p_value:.4f})")
        else:
             ax.set_title(f"KM Curve: {group_col}")
             
    else:
        # --- กรณีไม่มีกลุ่ม (All) ---
        T = data[time_col]
        E = data[event_col]
        kmf.fit(T, event_observed=E, label="All")
        kmf.plot_survival_function(ax=ax)
        
        # 🟢 เพิ่ม: สถิติพื้นฐาน
        stats_res["Total N"] = len(T)
        stats_res["Events"] = E.sum()
        stats_res["Censored"] = len(T) - E.sum()
        stats_res["Median Survival"] = kmf.median_survival_time_
        
        ax.set_title("Kaplan-Meier Survival Curve")
        
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Survival Probability")
    ax.grid(True, alpha=0.3)
    
    # ส่งคืนรูปกราฟ และ Dataframe สรุปผล
    return fig, pd.DataFrame(stats_res, index=["Value"]).T
    
# --- 🟢 2. Nelson-Aalen (Cumulative Hazard) ---
@st.cache_data(show_spinner=False) # 🟢 2. ADD CACHE
def fit_nelson_aalen(df, time_col, event_col, group_col=None):
    """
    สร้างกราฟ Nelson-Aalen พร้อมตารางสรุป N/Events
    """
    data = clean_survival_data(df, time_col, event_col, [group_col] if group_col else [])
    naf = NelsonAalenFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stats_res = {} # เตรียมตัวแปรเก็บค่าสถิติ
    
    if group_col:
        # กรณีมีกลุ่ม
        groups = sorted(data[group_col].unique())
        for g in groups:
            mask = data[group_col] == g
            T = data.loc[mask, time_col]
            E = data.loc[mask, event_col]
            
            naf.fit(T, event_observed=E, label=str(g))
            naf.plot_cumulative_hazard(ax=ax)
            
            # 🟢 เพิ่ม: เก็บข้อมูลสถิติพื้นฐาน (เหมือน KM)
            stats_res[f"{g} (N)"] = len(T)
            stats_res[f"{g} (Events)"] = E.sum()
            # Nelson-Aalen ไม่มี Median Time ที่นิยมใช้ จึงเก็บแค่ N กับ Events ก็พอ
            
        ax.set_title(f"Nelson-Aalen Cumulative Hazard: {group_col}")
    else:
        # กรณีรวม (All)
        T = data[time_col]
        E = data[event_col]
        naf.fit(T, event_observed=E, label="All")
        naf.plot_cumulative_hazard(ax=ax)
        
        # 🟢 เพิ่ม: สถิติพื้นฐาน
        stats_res["Total N"] = len(T)
        stats_res["Events"] = E.sum()
        stats_res["Censored"] = len(T) - E.sum()
        
        ax.set_title("Nelson-Aalen Cumulative Hazard Curve")
        
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Cumulative Hazard")
    ax.grid(True, alpha=0.3)
    
    # 🟢 ส่งคืน 2 ค่า: รูปกราฟ และ ตารางสรุป
    return fig, pd.DataFrame(stats_res, index=["Count"]).T
    
# --- 3. Cox Proportional Hazards Model ---
@st.cache_data(show_spinner=False) # 🟢 2. ADD CACHE
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
            plt.close(el['data']) # ✅ Already has close, good.
            uri = base64.b64encode(buf.getvalue()).decode('utf-8')
            html += f'<img src="data:image/png;base64,{uri}" style="max-width:100%;"/>'
            
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""
    return html
