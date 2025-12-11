import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import streamlit as st
import io, base64

# --- 1. Propensity Score Calculation ---
def calculate_ps(df, treatment_col, covariate_cols):
    """
    Compute propensity scores and their logit using logistic regression.
    
    Drops rows with missing values in the treatment or covariate columns, fits a logistic regression model predicting the treatment from the specified covariates, and returns the original rows augmented with `ps_score` (predicted probability of treatment) and `ps_logit` (logit of `ps_score`).
    
    Parameters:
        df (pandas.DataFrame): Input dataframe already preprocessed for missing values and encoding.
        treatment_col (str): Name of the binary treatment indicator column (expected numeric 0/1).
        covariate_cols (list[str]): List of column names to use as covariates in the propensity model.
    
    Returns:
        tuple:
            data (pandas.DataFrame): A copy of the input rows used for modeling with `ps_score` and `ps_logit` columns appended.
            clf (sklearn.linear_model.LogisticRegression): The trained logistic regression model.
    
    Raises:
        ValueError: If the treatment column is not numeric.
    """
    # Drop NA ในคอลัมน์ที่เกี่ยวข้อง
    data = df.dropna(subset=[treatment_col] + covariate_cols).copy()
    
    # X และ y
    X = data[covariate_cols]
    y = data[treatment_col]
    
    # ตรวจสอบ Data Type อีกครั้งเพื่อความชัวร์
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Treatment column '{treatment_col}' must be numeric (0/1).")
        
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X, y)
    
    # Get Probability (Score of class 1)
    data['ps_score'] = clf.predict_proba(X)[:, 1]
    
    # Get Logit Score (ป้องกัน Error log(0))
    eps = 1e-10
    data['ps_score'] = data['ps_score'].clip(eps, 1-eps)
    data['ps_logit'] = np.log(data['ps_score'] / (1 - data['ps_score']))
    
    return data, clf

# --- 2. Matching Algorithm ---
def perform_matching(df, treatment_col, ps_col='ps_logit', caliper=0.2):
    """
    Perform 1:1 greedy nearest-neighbor matching using a propensity score column with a caliper.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the treatment indicator and propensity score column.
        treatment_col (str): Name of the binary treatment column (values 0 and 1).
        ps_col (str): Name of the propensity score (or logit) column used for matching. Defaults to 'ps_logit'.
        caliper (float): Multiplier for caliper width expressed as caliper * std(ps_col). Controls allowable distance for matches.
    
    Returns:
        tuple:
            - pandas.DataFrame or None: If successful, a DataFrame containing matched treated and control rows with an added
              `match_id` column pairing each treated unit with its matched control. If matching fails, `None`.
            - str: A message describing the result. On success, reports the number of matched pairs. On failure, contains an
              explanatory error message (e.g., empty treated/control group or no matches within caliper).
    """
    # แยกกลุ่ม (ต้องเป็น 0 กับ 1 เท่านั้น)
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    if len(treated) == 0:
        return None, "Error: Treated group (1) is empty."
    if len(control) == 0:
        return None, "Error: Control group (0) is empty."

    # Caliper Calculation
    sd_logit = df[ps_col].std()
    caliper_width = caliper * sd_logit
    
    # Nearest Neighbors
    if len(control) < 1:
        return None, "Not enough controls."
        
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[[ps_col]])
    distances, indices = nbrs.kneighbors(treated[[ps_col]])
    
    # Create Match Candidates DataFrame
    match_candidates = pd.DataFrame({
        'treated_idx': treated.index,
        'control_iloc': indices.flatten(),
        'distance': distances.flatten()
    })
    
    # Map iloc to real index
    match_candidates['control_idx'] = control.iloc[match_candidates['control_iloc']].index.values
    
    # Filter by Caliper
    match_candidates = match_candidates[match_candidates['distance'] <= caliper_width]
    
    # Sort by distance (Greedy)
    match_candidates = match_candidates.sort_values('distance')
    
    # Perform Matching (Without Replacement)
    matched_pairs = []
    used_control = set()
    
    for _, row in match_candidates.iterrows():
        c_idx = row['control_idx']
        if c_idx not in used_control:
            matched_pairs.append(row)
            used_control.add(c_idx)
    
    matched_df_info = pd.DataFrame(matched_pairs)
    
    if len(matched_df_info) == 0:
        return None, "No matches found within caliper. Try increasing caliper width."
        
    # Retrieve Data
    matched_treated_ids = matched_df_info['treated_idx'].values
    matched_control_ids = matched_df_info['control_idx'].values
    
    df_matched = pd.concat([
        df.loc[matched_treated_ids].assign(match_id=range(len(matched_treated_ids))),
        df.loc[matched_control_ids].assign(match_id=range(len(matched_control_ids)))
    ])
    
    return df_matched, f"Matched {len(matched_treated_ids)} pairs."

# --- 3. SMD Calculation ---
def calculate_smd(df, treatment_col, covariate_cols):
    """
    Compute standardized mean differences (SMD) for numeric covariates between treated and control groups.
    
    Parameters:
        df (pandas.DataFrame): Input table containing the treatment indicator and covariates.
        treatment_col (str): Name of the binary treatment column where 1 indicates treated and 0 indicates control.
        covariate_cols (Iterable[str]): Sequence of covariate column names to evaluate.
    
    Returns:
        pandas.DataFrame: A DataFrame with columns `Variable` and `SMD`. `SMD` is the absolute difference in group means divided by the pooled standard deviation for each numeric covariate. Non-numeric covariates are skipped; covariates with zero variance in either group produce an SMD of 0.
    """
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    for col in covariate_cols:
        # คำนวณเฉพาะคอลัมน์ที่เป็นตัวเลข
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_t = treated[col].mean()
            mean_c = control[col].mean()
            var_t = treated[col].var()
            var_c = control[col].var()
            
            if pd.isna(var_t) or pd.isna(var_c): 
                smd = 0 # Handle constant columns
            else:
                pooled_sd = np.sqrt((var_t + var_c) / 2)
                smd = abs(mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0
                
            smd_data.append({'Variable': col, 'SMD': smd})
            
    return pd.DataFrame(smd_data)

# --- 4. Plotting & Report ---
def plot_love_plot(smd_pre, smd_post):
    """
    Create a Love plot comparing covariate standardized mean differences before and after matching.
    
    Parameters:
        smd_pre (pd.DataFrame): DataFrame containing pre-match SMDs with columns 'Variable' and 'SMD'.
        smd_post (pd.DataFrame): DataFrame containing post-match SMDs with columns 'Variable' and 'SMD'.
    
    Returns:
        fig (matplotlib.figure.Figure): Matplotlib Figure with SMD on the x-axis and covariates on the y-axis, showing pre- and post-match points and a reference line at SMD = 0.1.
    """
    smd_pre = smd_pre.copy(); smd_pre['Stage'] = 'Unmatched'
    smd_post = smd_post.copy(); smd_post['Stage'] = 'Matched'
    
    df_plot = pd.concat([smd_pre, smd_post])
    
    fig, ax = plt.subplots(figsize=(8, len(smd_pre) * 0.5 + 3))
    sns.scatterplot(data=df_plot, x='SMD', y='Variable', hue='Stage', style='Stage', s=100, ax=ax, palette={'Unmatched':'red', 'Matched':'blue'})
    
    ax.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Covariate Balance (Love Plot)')
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.grid(True, alpha=0.3)
    return fig

def generate_psm_report(title, elements):
    """
    Generate a standalone HTML report string from a sequence of report elements.
    
    Parameters:
        title (str): The report title displayed as a top-level heading.
        elements (list): Ordered list of elements to include. Each element is a dict with:
            - type (str): One of 'text', 'table', or 'plot'.
            - data: For 'text', a string; for 'table', a pandas DataFrame; for 'plot', a matplotlib Figure.
    
    Returns:
        html (str): A complete HTML document as a string. Tables are rendered as HTML tables and plots are embedded inline as base64-encoded PNG images.
    """
    css = """<style>body{font-family:'Segoe UI';padding:20px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;text-align:center;} th{background:#f2f2f2;}</style>"""
    html = f"<html><head>{css}</head><body><h2>{title}</h2>"
    for el in elements:
        if el['type'] == 'text': html += f"<p>{el['data']}</p>"
        elif el['type'] == 'table': html += el['data'].to_html()
        elif el['type'] == 'plot':
            buf = io.BytesIO()
            el['data'].savefig(buf, format='png', bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            html += f'<br><img src="data:image/png;base64,{b64}" style="max-width:100%"><br>'
    return html