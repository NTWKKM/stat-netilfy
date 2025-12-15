import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler  # Added for robust scaling of continuous covariates
import io, base64
import html as _html

# --- 1. Propensity Score Calculation ---
def calculate_ps(df, treatment_col, covariate_cols):
    """ 
    Estimate propensity scores and their logit (log-odds) for a binary treatment using logistic regression.
    Includes Robust Scaling for continuous variables to ensure model stability.
    """
    # Drop NA ในคอลัมน์ที่เกี่ยวข้อง
    data = df.dropna(subset=[treatment_col, *covariate_cols]).copy()
    
    # 🟢 ROBUST UPDATE: Separate Numeric vs Binary for Scaling
    # Sklearn LogisticRegression applies L2 Regularization by default.
    # Unscaled continuous variables can distort the model.
    X_processed = data[covariate_cols].copy()
    
    # Identify continuous columns to scale
    cont_cols = []
    for col in covariate_cols:
        if pd.api.types.is_numeric_dtype(X_processed[col]):
            # If > 2 unique values, treat as continuous/ordinal and scale it
            # (Keep binary variables 0/1 as is for interpretability/stability)
            if X_processed[col].nunique() > 2:
                cont_cols.append(col)
    
    if cont_cols:
        scaler = StandardScaler()
        X_processed.loc[:, cont_cols] = scaler.fit_transform(X_processed[cont_cols])
    
    X = X_processed
    y = data[treatment_col]
    
    # ตรวจสอบ Data Type อีกครั้งเพื่อความชัวร์
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Treatment column '{treatment_col}' must be numeric (0/1).")
    
    # Use liblinear with L2 penalty (Robust default)
    clf = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
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
    Perform 1:1 greedy nearest-neighbor matching of treated and control units based on a propensity score logit.
    Matches each treated unit to the nearest control within a caliper defined as (caliper * standard deviation of ps_col).
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
    
    # 🟢 Safety Check for zero variance
    if not np.isfinite(sd_logit) or sd_logit < 1e-9:
        return None, f"Error: {ps_col} has zero or undefined variance; cannot apply caliper matching."
        
    caliper_width = caliper * sd_logit
    
    # Nearest Neighbors
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
    for row in match_candidates.itertuples():
        c_idx = row.control_idx
        if c_idx not in used_control:
            matched_pairs.append({
                'treated_idx': row.treated_idx,
                'control_idx': c_idx,
                'distance': row.distance,
                'control_iloc': row.control_iloc
            })
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
    Compute standardized mean differences (SMD) for numeric covariates.
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
                smd = 0  # Handle constant columns
            else:
                pooled_sd = np.sqrt((var_t + var_c) / 2)
                smd = abs(mean_t - mean_c) / pooled_sd if pooled_sd > 1e-9 else 0
            
            smd_data.append({'Variable': col, 'SMD': smd})
    
    return pd.DataFrame(smd_data)

# --- 4. Plotting & Report (Plotly) ---
def plot_love_plot(smd_pre, smd_post):
    """ 
    Create an interactive Love plot using Plotly.
    """
    smd_pre = smd_pre.copy()
    smd_pre['Stage'] = 'Unmatched'
    smd_post = smd_post.copy()
    smd_post['Stage'] = 'Matched'
    df_plot = pd.concat([smd_pre, smd_post])
    
    # สร้าง Plotly scatter plot
    fig = px.scatter(
        df_plot,
        x='SMD',
        y='Variable',
        color='Stage',
        symbol='Stage',
        title='Covariate Balance (Love Plot)',
        labels={
            'SMD': 'Standardized Mean Difference (SMD)',
            'Variable': 'Variable',
            'Stage': 'Stage'
        },
        color_discrete_map={
            'Unmatched': 'red',
            'Matched': 'blue'
        },
        symbol_map={
            'Unmatched': 'circle',
            'Matched': 'circle'
        },
        size_max=10
    )
    
    # เพิ่มเส้นอ้างอิง (SMD = 0.1)
    fig.add_vline(
        x=0.1,
        line_dash='dash',
        line_color='gray',
        opacity=0.5,
        annotation_text='SMD = 0.1',
        annotation_position='top right'
    )
    
    # ปรับแต่งเค้าโครง
    fig.update_layout(
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis_title='Standardized Mean Difference (SMD)',
        yaxis_title='Variable',
        font={'size': 12},
        height=max(400, len(smd_pre) * 40 + 100) # Dynamic height
    )
    
    # ปรับแต่ง marker
    fig.update_traces(
        marker={'size': 8, 'opacity': 0.7},
        selector={'mode': 'markers'}
    )
    
    return fig

def generate_psm_report(title, elements):
    """ 
    Generate a styled HTML report.
    """
    css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
        }
        h2 {
            color: #0066cc;
            margin-top: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table th, table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        table th {
            background-color: #0066cc;
            color: white;
        }
        table tr:hover {
            background-color: #f0f0f0;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        p {
            line-height: 1.6;
            color: #333;
        }
    </style>
    """

    plotly_cdn = "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
    html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{css}{plotly_cdn}</head>\n<body>"
    html += f"<h1>{_html.escape(str(title))}</h1>"
    
    for el in elements:
        if el['type'] == 'text':
            html += f"<p>{_html.escape(str(el['data']))}</p>"
        elif el['type'] == 'table':
            html += el['data'].to_html(classes='report-table', border=0)
        elif el['type'] == 'plot':
            # ตรวจสอบว่าเป็น Plotly Figure หรือ Matplotlib Figure
            plot_obj = el['data']
            
            # ถ้าเป็น Plotly Figure
            if hasattr(plot_obj, 'to_html'):
                html += plot_obj.to_html(full_html=False, include_plotlyjs=False, div_id=f"plot_{id(plot_obj)}")
            else:
                # ถ้าเป็น Matplotlib Figure - แปลงเป็น PNG และ embed
                buf = io.BytesIO()
                plot_obj.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{b64}" />'
    
    html += "</body>\n</html>"
    return html
