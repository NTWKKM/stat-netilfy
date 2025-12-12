import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter, CoxTimeVaryingFitter
from lifelines.statistics import logrank_test
import io
import base64
import contextlib
import streamlit as st

# --- Helper: Clean Data ---
def clean_survival_data(df, time_col, event_col, covariates=None):
    """
    Prepare a numeric, complete-case DataFrame for survival analysis with one row per subject.
    
    This function selects the requested time, event, and optional covariate columns from the input DataFrame, coerces all selected columns to numeric (invalid parsing becomes NaN), and drops any rows containing missing values.
    
    Parameters:
        df (pd.DataFrame): Source data.
        time_col (str): Name of the duration/time-to-event column.
        event_col (str): Name of the event indicator column (typically 0/1).
        covariates (list[str] | None): Optional list of covariate column names to include.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame containing only the requested columns as numeric types and only complete cases (no missing values).
    """
    cols = [time_col, event_col]
    if covariates:
        cols += covariates
    
    # เลือกเฉพาะคอลัมน์ที่ใช้ และแปลงเป็นตัวเลข
    data = df[cols].copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
        
    # ลบแถวที่มี Missing Value (Complete Case Analysis)
    data = data.dropna()
    return data

# --- 1. Kaplan-Meier & Log-Rank ---
@st.cache_data(show_spinner=False)
def fit_km_logrank(df, time_col, event_col, group_col=None):
    """
    Fit and plot Kaplan–Meier survival curves and perform log-rank testing for group comparisons.
    
    This function accepts pre-filtered data (suitable for landmark analyses), fits Kaplan–Meier estimators either overall or by group, plots the survival curves, and computes basic summary statistics. If two groups are present, a log-rank test is performed and its p-value is included in the results and plot title.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing time, event, and optional grouping columns.
        time_col (str): Column name for survival or follow-up time.
        event_col (str): Column name for the event indicator (1=event, 0=censored).
        group_col (str, optional): Column name for a grouping variable to produce group-wise curves and comparisons. If omitted, a single overall curve is produced.
    
    Returns:
        tuple: (figure, stats_df)
            figure (matplotlib.figure.Figure): Matplotlib Figure containing the Kaplan–Meier plot.
            stats_df (pandas.DataFrame): Table of summary statistics (per-group or overall) with rows for each statistic and columns for groups or a single "Value" column.
    """
    data = clean_survival_data(df, time_col, event_col, [group_col] if group_col else [])
    
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stats_res = {}
    
    if group_col:
        # เปรียบเทียบระหว่างกลุ่ม
        groups = sorted(data[group_col].unique())
        T_list, E_list = [], []
        
        for g in groups:
            mask = data[group_col] == g
            T = data.loc[mask, time_col]
            E = data.loc[mask, event_col]
            
            if len(T) > 0:
                kmf.fit(T, event_observed=E, label=str(g))
                kmf.plot_survival_function(ax=ax, ci_show=False)
                
                # เก็บสถิติพื้นฐาน
                stats_res[f"{g} (N)"] = len(T)
                stats_res[f"{g} (Events)"] = E.sum()
                stats_res[f"{g} (Median)"] = kmf.median_survival_time_
                
                T_list.append(T)
                E_list.append(E)
            
        # Log-Rank Test (ถ้ามี 2 กลุ่มขึ้นไป)
        if len(T_list) >= 2:
            if len(T_list) == 2:
                lr_result = logrank_test(T_list[0], T_list[1], event_observed_A=E_list[0], event_observed_B=E_list[1])
                stats_res['Log-Rank p-value'] = lr_result.p_value
                ax.set_title(f"KM Curve: {group_col} (p = {lr_result.p_value:.4f})")
            else:
                ax.set_title(f"KM Curve: {group_col}")
        else:
             ax.set_title(f"KM Curve: {group_col}")
             
    else:
        # ดูภาพรวม (ไม่มีกลุ่มเปรียบเทียบ)
        T = data[time_col]
        E = data[event_col]
        kmf.fit(T, event_observed=E, label="All Patients")
        kmf.plot_survival_function(ax=ax)
        
        stats_res["Total N"] = len(T)
        stats_res["Events"] = E.sum()
        stats_res["Censored"] = len(T) - E.sum()
        stats_res["Median Survival"] = kmf.median_survival_time_
        
        ax.set_title("Kaplan-Meier Survival Curve")
        
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Survival Probability")
    ax.grid(True, alpha=0.3)
    
    return fig, pd.DataFrame(stats_res, index=["Value"]).T

# --- 2. Nelson-Aalen (Cumulative Hazard) ---
@st.cache_data(show_spinner=False)
def fit_nelson_aalen(df, time_col, event_col, group_col=None):
    """
    Create and plot a Nelson–Aalen cumulative hazard curve, optionally stratified by a grouping column.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing time, event, and optional group columns.
        time_col (str): Column name for survival time or follow-up time.
        event_col (str): Column name for the event indicator (1 for event, 0 for censored).
        group_col (str, optional): Column name to stratify the cumulative hazard by groups. If omitted, the function fits a single curve for all patients.
    
    Returns:
        tuple: A tuple (fig, stats_df) where:
            - fig (matplotlib.figure.Figure): Figure containing the Nelson–Aalen cumulative hazard plot.
            - stats_df (pandas.DataFrame): Table of counts and event totals per group (or total counts when no group_col is provided).
    """
    data = clean_survival_data(df, time_col, event_col, [group_col] if group_col else [])
    naf = NelsonAalenFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    stats_res = {}
    
    if group_col:
        groups = sorted(data[group_col].unique())
        for g in groups:
            mask = data[group_col] == g
            group_data = data.loc[mask]
            if not group_data.empty:
                naf.fit(group_data[time_col], event_observed=group_data[event_col], label=str(g))
                naf.plot_cumulative_hazard(ax=ax)
                stats_res[f"{g} (N)"] = len(group_data)
                stats_res[f"{g} (Events)"] = group_data[event_col].sum()
    else:
        T = data[time_col]
        E = data[event_col]
        naf.fit(T, event_observed=E, label="All")
        naf.plot_cumulative_hazard(ax=ax)
        stats_res["Total N"] = len(T)
        stats_res["Events"] = E.sum()
        ax.set_title("Nelson-Aalen Cumulative Hazard")
        
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Cumulative Hazard")
    ax.grid(True, alpha=0.3)
    
    return fig, pd.DataFrame(stats_res, index=["Count"]).T

# --- 3. Cox Proportional Hazards (Standard) ---
@st.cache_data(show_spinner=False)
def fit_cox_ph(df, time_col, event_col, covariates):
    """
    Fit a time-independent Cox proportional hazards model on numeric, complete-case data.
    
    Parameters:
        df (pandas.DataFrame): Input dataset.
        time_col (str): Column name containing event durations.
        event_col (str): Column name containing the event/censoring indicator (0/1).
        covariates (list[str]): List of covariate column names to include in the model.
    
    Returns:
        cph (lifelines.CoxPHFitter or None): Fitted CoxPHFitter when successful, otherwise `None`.
        summary_df (pandas.DataFrame or None): Summary table with columns `['Coef', 'HR', 'Lower 95%', 'Upper 95%', 'P-value']` when successful, otherwise `None`.
        data (pandas.DataFrame or None): Cleaned numeric complete-case DataFrame used for fitting when successful, otherwise `None`.
        error (str or None): Error message string if fitting failed, otherwise `None`.
    """
    data = clean_survival_data(df, time_col, event_col, covariates)
    cph = CoxPHFitter()
    try:
        cph.fit(data, duration_col=time_col, event_col=event_col)
        
        summary_df = cph.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
        summary_df.columns = ['Coef', 'HR', 'Lower 95%', 'Upper 95%', 'P-value']
        
        return cph, summary_df, data, None
    except Exception as e:
        return None, None, None, str(e)

# --- 🟢 4. Time-Dependent Cox Regression (New!) ---
@st.cache_data(show_spinner=False)
def fit_cox_time_varying(df, id_col, event_col, start_col, stop_col, covariates):
    """
    Fit a time-varying Cox proportional hazards model using start–stop (long-format) data.
    
    Parameters:
        df (pd.DataFrame): Source dataframe containing long-format (start–stop) rows.
        id_col (str): Column name identifying subject/patient IDs.
        event_col (str): Column name for the event indicator (1 if event occurred, 0 if censored).
        start_col (str): Column name for the interval start time.
        stop_col (str): Column name for the interval stop time.
        covariates (list[str]): List of time-varying covariate column names to include in the model.
    
    Returns:
        tuple:
            - fitted_model (CoxTimeVaryingFitter or None): The fitted CoxTimeVaryingFitter on success, otherwise None.
            - summary_df (pd.DataFrame or None): A summary table with columns ['Coef', 'HR', 'Lower 95%', 'Upper 95%', 'P-value'] on success, otherwise None.
            - error (str or None): None on success; otherwise an error message describing the failure. Possible error messages include:
                - "Error: Data is empty after selecting columns and dropping NAs." when no rows remain after selecting required columns and dropping missing values.
                - "Error: Found rows where Start Time >= Stop Time." when any interval has start greater than or equal to stop.
                - "Model Failed: <details>" when model fitting raises an exception.
    """
    # 1. เลือกคอลัมน์ที่จำเป็น
    cols = [id_col, event_col, start_col, stop_col] + covariates
    
    # 2. จัดการข้อมูล (ควรตรวจสอบว่า df มี NaN หรือไม่ก่อน drop)
    data = df[cols].copy()
    data = data.dropna()
    
    # 3. ตรวจสอบเบื้องต้น (เช่น start < stop)
    if data.empty:
         return None, None, "Error: Data is empty after selecting columns and dropping NAs."
         
    if (data[start_col] >= data[stop_col]).any():
        return None, None, "Error: Found rows where Start Time >= Stop Time."

    ctv = CoxTimeVaryingFitter()
    try:
        # fit() ต้องการ id_col เพื่อรู้ว่าแถวไหนเป็นคนเดียวกัน
        ctv.fit(data, id_col=id_col, event_col=event_col, start_col=start_col, stop_col=stop_col, show_progress=False)
        
        summary_df = ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
        summary_df.columns = ['Coef', 'HR', 'Lower 95%', 'Upper 95%', 'P-value']
        
    return ctv, summary_df, data, None
    except Exception as e:
        return None, None, None, f"Model Failed: {str(e)}"

# --- 5. Check Assumptions ---
def check_cph_assumptions(cph, data):
    """
    Check proportional hazards assumptions for a fitted Cox model and capture the textual diagnostics and any generated diagnostic plots.
    
    Parameters:
        cph: A fitted lifelines Cox model (e.g., CoxPHFitter or CoxTimeVaryingFitter) whose assumptions will be checked.
        data: pandas.DataFrame used as the input to the assumption checks.
    
    Returns:
        tuple:
            advice_text (str): Textual diagnostic output produced by lifelines' check_assumptions (or an error message if the check failed).
            figs (list[matplotlib.figure.Figure]): List of matplotlib Figure objects created by the diagnostic checks; empty if none or on error.
    """
    try:
        f = io.StringIO()
        old_figs = plt.get_fignums()
        
        with contextlib.redirect_stdout(f):
            # lifelines จะ print คำแนะนำและ plot กราฟออกมา
            cph.check_assumptions(data, p_value_threshold=0.05, show_plots=True)
        
        advice_text = f.getvalue()
        
        # ดักจับกราฟที่ถูกสร้างขึ้นใหม่
        new_figs_nums = [n for n in plt.get_fignums() if n not in old_figs]
        figs = [plt.figure(n) for n in new_figs_nums]
        
        return advice_text, figs
    except Exception as e:
        return f"Error checking assumptions: {str(e)}", []

# --- 6. Report Generator ---
def generate_report_survival(title, elements):
    """
    Assemble an HTML report from given titled elements (text, headers, tables, and plots) suitable for embedding or saving.
    
    Parameters:
        title (str): Report title displayed at the top of the report.
        elements (list): Ordered list of elements to include in the report. Each element is a dict with:
            - type (str): One of 'text', 'header', 'table', or 'plot'.
            - data: For 'text' and 'header', a string; for 'table', a pandas DataFrame; for 'plot', a matplotlib Figure.
    
    Returns:
        html (str): A complete HTML document as a string containing embedded styles, the rendered elements, and a footer.
    """
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
