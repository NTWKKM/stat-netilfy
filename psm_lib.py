import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import streamlit as st

# --- 1. Propensity Score Calculation ---
def calculate_ps(df, treatment_col, covariate_cols):
    """
    คำนวณ Propensity Score ด้วย Logistic Regression
    """
    data = df.dropna(subset=[treatment_col] + covariate_cols).copy()
    
    # Fit Logistic Regression
    X = data[covariate_cols]
    y = data[treatment_col]
    
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X, y)
    
    # Get Probability (Score)
    data['ps_score'] = clf.predict_proba(X)[:, 1]
    
    # Get Logit Score (Linear Predictor) - ดีกว่าสำหรับการ Matching
    data['ps_logit'] = np.log(data['ps_score'] / (1 - data['ps_score']))
    
    return data, clf

# --- 2. Matching Algorithm (Greedy 1:1 Without Replacement) ---
def perform_matching(df, treatment_col, ps_col='ps_logit', caliper=0.2):
    """
    จับคู่ 1:1 แบบไม่ใส่คืน (Without Replacement) โดยใช้ Caliper
    """
    # แยกกลุ่ม Case (1) และ Control (0)
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    if len(treated) == 0 or len(control) == 0:
        return None, "Error: One of the groups is empty."

    # คำนวณ Caliper Width (SD ของ Logit Score * caliper)
    sd_logit = df[ps_col].std()
    caliper_width = caliper * sd_logit
    
    # เตรียม Nearest Neighbors บนกลุ่ม Control
    # เราใช้ sklearn เพื่อหาเพื่อนบ้านที่ใกล้ที่สุด
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[[ps_col]])
    
    # หาคู่สำหรับทุก Case
    distances, indices = nbrs.kneighbors(treated[[ps_col]])
    
    # จัดการผลลัพธ์เพื่อทำ Greedy Matching (Without Replacement)
    # สร้าง DataFrame ชั่วคราวเก็บระยะทางและ Index
    match_candidates = pd.DataFrame({
        'treated_idx': treated.index,
        'control_iloc': indices.flatten(), # เป็น index 0,1,2.. ของ array control
        'distance': distances.flatten()
    })
    
    # แปลง iloc กลับเป็น index จริงของ DataFrame
    match_candidates['control_idx'] = control.iloc[match_candidates['control_iloc']].index.values
    
    # กรองด้วย Caliper
    match_candidates = match_candidates[match_candidates['distance'] <= caliper_width]
    
    # เรียงตามระยะทาง (ใครใกล้สุดได้จับคู่ก่อน)
    match_candidates = match_candidates.sort_values('distance')
    
    # เริ่มจับคู่
    matched_pairs = []
    used_control = set()
    
    for _, row in match_candidates.iterrows():
        c_idx = row['control_idx']
        if c_idx not in used_control:
            matched_pairs.append(row)
            used_control.add(c_idx)
    
    matched_df_info = pd.DataFrame(matched_pairs)
    
    if len(matched_df_info) == 0:
        return None, "No matches found within caliper."
        
    # ดึงข้อมูลจริงกลับมา
    matched_treated_ids = matched_df_info['treated_idx'].values
    matched_control_ids = matched_df_info['control_idx'].values
    
    df_matched = pd.concat([
        df.loc[matched_treated_ids].assign(match_id=range(len(matched_treated_ids))),
        df.loc[matched_control_ids].assign(match_id=range(len(matched_control_ids)))
    ])
    
    return df_matched, f"Matched {len(matched_treated_ids)} pairs."

# --- 3. Balance Check (SMD) ---
def calculate_smd(df, treatment_col, covariate_cols):
    """
    คำนวณ Standardized Mean Difference (SMD)
    SMD < 0.1 ถือว่า Balance ดีมาก, < 0.2 พอรับได้
    """
    smd_data = []
    
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    for col in covariate_cols:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_t = treated[col].mean()
            mean_c = control[col].mean()
            var_t = treated[col].var()
            var_c = control[col].var()
            
            # Formula: Cohen's d style
            pooled_sd = np.sqrt((var_t + var_c) / 2)
            if pooled_sd == 0: 
                smd = 0
            else:
                smd = abs(mean_t - mean_c) / pooled_sd
                
            smd_data.append({'Variable': col, 'SMD': smd})
            
    return pd.DataFrame(smd_data)

# --- 4. Plot Love Plot ---
def plot_love_plot(smd_pre, smd_post):
    """
    สร้างกราฟ Love Plot เปรียบเทียบ SMD ก่อนและหลัง Match
    """
    # Merge Data
    smd_pre['Stage'] = 'Unmatched'
    smd_post['Stage'] = 'Matched'
    
    df_plot = pd.concat([smd_pre, smd_post])
    
    fig, ax = plt.subplots(figsize=(8, len(smd_pre) * 0.8 + 2))
    sns.scatterplot(data=df_plot, x='SMD', y='Variable', hue='Stage', style='Stage', s=100, ax=ax)
    
    # Add Threshold lines
    ax.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(0, max(0.5, df_plot['SMD'].max() + 0.1))
    ax.set_title('Covariate Balance (Love Plot)')
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.grid(True, alpha=0.3)
    
    return fig

# --- 5. Generate Report ---
def generate_psm_report(title, elements):
    # (ใช้ Style เดียวกับ report อื่นๆ)
    css = """<style>body{font-family:'Segoe UI';padding:20px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;text-align:center;} th{background:#f2f2f2;}</style>"""
    html = f"<html><head>{css}</head><body><h2>{title}</h2>"
    
    for el in elements:
        if el['type'] == 'text': html += f"<p>{el['data']}</p>"
        elif el['type'] == 'table': html += el['data'].to_html()
        elif el['type'] == 'plot':
            import io, base64
            buf = io.BytesIO()
            el['data'].savefig(buf, format='png', bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            html += f'<br><img src="data:image/png;base64,{b64}" style="max-width:100%"><br>'
            
    return html
