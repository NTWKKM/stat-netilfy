import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import io, base64
import html as _html

# Helper function for standardization
def _standardize_numeric_cols(data, cols):
    """
    Standardize numeric columns in-place, BUT SKIP BINARY columns (0/1).
    This prevents numerical instability in Cox models.
    """
    for col in cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            # Check if binary (only 2 unique values, e.g., 0 and 1)
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                continue # Skip standardization for binary variables
            
            std = data[col].std()
            if pd.isna(std) or std == 0:
                warnings.warn(f"Covariate '{col}' has zero variance", stacklevel=3)
            else:
                data[col] = (data[col] - data[col].mean()) / std

# --- 1. Kaplan-Meier & Log-Rank (With Robust CI) 🟢 FIX KM CI ---
def fit_km_logrank(df, duration_col, event_col, group_col):
    """
    Fits KM curves and performs Log-rank test.
    Returns: (fig, stats_df)
    """
    data = df.dropna(subset=[duration_col, event_col])
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Missing group column: {group_col}")
        data = data.dropna(subset=[group_col])
        groups = sorted(data[group_col].unique(), key=lambda v: str(v))
    else:
        groups = ['Overall']

    if len(data) == 0:
        raise ValueError("No valid data.")

    # Plotly Figure
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, g in enumerate(groups):
        if group_col:
            df_g = data[data[group_col] == g]
            label = f"{group_col}={g}"
        else:
            df_g = data
            label = "Overall"
        
        # Check if enough data
        if len(df_g) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(df_g[duration_col], df_g[event_col], label=label)

            # --- 🟢 FIX: Check existence and access CI by position ---
            ci_exists = hasattr(kmf, 'confidence_interval_') and not kmf.confidence_interval_.empty
            
            if ci_exists and kmf.confidence_interval_.shape[1] >= 2:
                # Use .iloc[:, 0] for lower bound and .iloc[:, 1] for upper bound
                ci_lower = kmf.confidence_interval_.iloc[:, 0]
                ci_upper = kmf.confidence_interval_.iloc[:, 1]

                # 1. Add Shaded Area (Confidence Interval)
                fig.add_trace(go.Scatter(
                    x=list(ci_lower.index) + list(ci_upper.index)[::-1], # Times forward and backward
                    y=list(ci_lower.values) + list(ci_upper.values)[::-1], # CI lower forward, CI upper backward
                    fill='toself',
                    fillcolor=colors[i % len(colors)] + '30', # Add transparency (30)
                    line=dict(color='rgba(255,255,255,0)'), # Invisible line
                    hoverinfo="skip", 
                    name=f'{label} 95% CI',
                    showlegend=False
                ))
            
            # 2. Survival Curve (KM Estimate)
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'{label}<br>Time: %{{x:.1f}}<br>Surv: %{{y:.3f}}<extra></extra>'
            ))
            

    fig.update_layout(
        title='Kaplan-Meier Survival Curves (with 95% CI)',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    fig.update_yaxes(range=[0, 1.05])

    # Log-Rank Test
    stats_data = {}
    try:
        if len(groups) == 2 and group_col:
            g1, g2 = groups
            res = logrank_test(
                data[data[group_col] == g1][duration_col],
                data[data[group_col] == g2][duration_col],
                event_observed_A=data[data[group_col] == g1][event_col],
                event_observed_B=data[data[group_col] == g2][event_col]
            )
            stats_data = {
                'Test': 'Log-Rank (Pairwise)',
                'Statistic': res.test_statistic,
                'P-value': res.p_value,
                'Comparison': f'{g1} vs {g2}'
            }
        elif len(groups) > 2 and group_col:
            res = multivariate_logrank_test(data[duration_col], data[group_col], data[event_col])
            stats_data = {
                'Test': 'Log-Rank (Multivariate)',
                'Statistic': res.test_statistic,
                'P-value': res.p_value,
                'Comparison': 'All groups'
            }
        else:
            stats_data = {'Test': 'None', 'Note': 'Single group or no group selected'}
    except Exception as e:
        stats_data = {'Test': 'Error', 'Note': str(e)}

    return fig, pd.DataFrame([stats_data])

# --- 2. Nelson-Aalen (With Robust CI) 🟢 FIX NA CI ---
def fit_nelson_aalen(df, duration_col, event_col, group_col):
    """
    Fits Nelson-Aalen cumulative hazard.
    Returns: (fig, stats_df)
    """
    data = df.dropna(subset=[duration_col, event_col])
    if len(data) == 0:
        raise ValueError("No valid data.")
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Missing group column: {group_col}")
        data = data.dropna(subset=[group_col])
        groups = sorted(data[group_col].unique(), key=lambda v: str(v))
    else:
        groups = ['Overall']

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    stats_list = []

    for i, g in enumerate(groups):
        if group_col:
            df_g = data[data[group_col] == g]
            label = f"{group_col}={g}"
        else:
            df_g = data
            label = "Overall"

        if len(df_g) > 0:
            naf = NelsonAalenFitter()
            naf.fit(df_g[duration_col], event_observed=df_g[event_col], label=label)
            
            # --- 🟢 FIX: Check existence and access CI by position ---
            ci_exists = hasattr(naf, 'confidence_interval_') and not naf.confidence_interval_.empty

            if ci_exists and naf.confidence_interval_.shape[1] >= 2:
                # Use .iloc[:, 0] for lower bound and .iloc[:, 1] for upper bound
                ci_lower = naf.confidence_interval_.iloc[:, 0]
                ci_upper = naf.confidence_interval_.iloc[:, 1]

                # 1. Add Shaded Area (Confidence Interval)
                fig.add_trace(go.Scatter(
                    x=list(ci_lower.index) + list(ci_upper.index)[::-1], 
                    y=list(ci_lower.values) + list(ci_upper.values)[::-1], 
                    fill='toself',
                    fillcolor=colors[i % len(colors)] + '30', 
                    line=dict(color='rgba(255,255,255,0)'), 
                    hoverinfo="skip", 
                    name=f'{label} 95% CI',
                    showlegend=False
                ))

            # 2. Cumulative Hazard Curve (NA Estimate)
            fig.add_trace(go.Scatter(
                x=naf.cumulative_hazard_.index,
                y=naf.cumulative_hazard_.iloc[:, 0],
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
            
            stats_list.append({
                'Group': label,
                'N': len(df_g),
                'Events': df_g[event_col].sum()
            })

    fig.update_layout(
        title='Nelson-Aalen Cumulative Hazard (with 95% CI)',
        xaxis_title='Time',
        yaxis_title='Cumulative Hazard',
        template='plotly_white',
        height=500
    )

    return fig, pd.DataFrame(stats_list)

# --- 3. Cox Proportional Hazards (Robust Version with Firth Fallback) ---
def fit_cox_ph(df, duration_col, event_col, covariate_cols):
    # ... (โค้ด fit_cox_ph เดิม ถูกต้องแล้ว)
    # 1. Validation
    missing = [c for c in [duration_col, event_col, *covariate_cols] if c not in df.columns]
    if missing:
        return None, None, df, f"Missing columns: {missing}"

    data = df.dropna(subset=[duration_col, event_col, *covariate_cols]).copy()
    if len(data) == 0:
        return None, None, data, "No valid data after dropping missing values."

    if data[event_col].sum() == 0:
        return None, None, data, "No events observed (all censored). CoxPH requires at least one event." 
    
    # 2. Check variance
    for col in covariate_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            std = data[col].std()
            if pd.isna(std) or std == 0:
                return None, None, data, f"Covariate '{col}' has zero variance (or insufficient rows)."
    
    # 3. Standardize (Skip binary to improve stability)
    _standardize_numeric_cols(data, covariate_cols)
    
    # 4. Fitting Strategy (Progressive Robustness)
    penalizers_L2 = [0.0, 0.1, 1.0, 10.0] 
    cph = None
    last_error = None
    method_used = None
    
    for p in penalizers_L2:
        try:
            temp_cph = CoxPHFitter(penalizer=p) 
            temp_cph.fit(data, duration_col=duration_col, event_col=event_col)
            cph = temp_cph
            method_used = f"L2 Penalized CoxPH (p={p})"
            if p == 0.0: method_used = "Standard CoxPH"
            break
        except Exception as e:
            last_error = e
            continue

    if cph is None:
        return None, None, data, f"Model Convergence Failed. Last attempt used: {method_used if method_used else 'None'}. Details: {last_error}.\nTry checking for high correlation (multicollinearity) or perfect separation."

    # Format Results
    summary = cph.summary.copy()
    summary['HR'] = np.exp(summary['coef'])
    summary['95% CI Lower'] = np.exp(summary['lower 95% bound'])
    summary['95% CI Upper'] = np.exp(summary['upper 95% bound'])
    
    # Add Method used to results table
    summary['Method'] = method_used
    summary.index.name = "Covariate"
    
    res_df = summary[['HR', '95% CI Lower', '95% CI Upper', 'p', 'Method']].rename(columns={'p': 'P-value'})
    
    return cph, res_df, data, None

# ... (check_cph_assumptions เหมือนเดิม)
def check_cph_assumptions(cph, data):
    # ... (โค้ด check_cph_assumptions เดิม)
    try:
        # 1. Statistical Test
        results = proportional_hazard_test(cph, data, time_transform='rank')
        text_report = "Proportional Hazards Test Results:\n" + results.summary.to_string()
        
        # 2. Schoenfeld Residual Plots
        img_bytes_list = []
        
        # Compute residuals
        scaled_schoenfeld = cph.compute_residuals(data, 'scaled_schoenfeld')
        
        for col in scaled_schoenfeld.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(data[cph.duration_col], scaled_schoenfeld[col], alpha=0.5)
            
            # Trend line (optional)
            try:
                z = np.polyfit(data[cph.duration_col], scaled_schoenfeld[col], 1)
                p = np.poly1d(z)
                ax.plot(data[cph.duration_col], p(data[cph.duration_col]), "r--", alpha=0.8)
            except Exception as e:
                warnings.warn(f"Could not fit trend line for {col}: {e}", stacklevel=2)
                
            ax.set_title(f"Schoenfeld Residuals: {col}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Scaled Residuals")
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_bytes_list.append(buf.getvalue())
            plt.close(fig)

        return text_report, img_bytes_list

    except Exception as e:
        return f"Assumption check failed: {e}", []


# --- 4. Landmark Analysis (KM) 🟢 FIX LM CI ---
def fit_km_landmark(df, duration_col, event_col, group_col, landmark_time):
    """
    Performs Kaplan-Meier Survival Analysis using the Landmark Method.
    
    Returns: (fig, stats_df, n_pre_filter, n_post_filter)
    """
    
    # 1. Data Cleaning
    data = df.dropna(subset=[duration_col, event_col, group_col])
    n_pre_filter = len(data)

    # 2. Filtering (The Landmark Step)
    landmark_data = data[data[duration_col] >= landmark_time].copy()
    n_post_filter = len(landmark_data)
    
    if n_post_filter < 2:
        return None, None, n_pre_filter, n_post_filter, "Error: Insufficient patients (N < 2) survived until the landmark time."
    
    # 3. Recalculate Duration (Crucial Step)
    landmark_data['New_Duration'] = landmark_data[duration_col] - landmark_time
    
    # 4. KM Fitting (Standardized Plotting)
    groups = sorted(landmark_data[group_col].unique(), key=lambda v: str(v))
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, g in enumerate(groups):
        df_g = landmark_data[landmark_data[group_col] == g]
        label = f"{group_col}={g}"
        
        if len(df_g) > 0:
            kmf = KaplanMeierFitter()
            
            # Fit using the New_Duration
            kmf.fit(df_g['New_Duration'], df_g[event_col], label=label)

            # --- 🟢 FIX: Check existence and access CI by position ---
            ci_exists = hasattr(kmf, 'confidence_interval_') and not kmf.confidence_interval_.empty
            
            if ci_exists and kmf.confidence_interval_.shape[1] >= 2:
                # Use .iloc[:, 0] for lower bound and .iloc[:, 1] for upper bound
                ci_lower = kmf.confidence_interval_.iloc[:, 0]
                ci_upper = kmf.confidence_interval_.iloc[:, 1]

                # 1. Add Shaded Area (Confidence Interval)
                fig.add_trace(go.Scatter(
                    x=list(ci_lower.index) + list(ci_upper.index)[::-1],
                    y=list(ci_lower.values) + list(ci_upper.values)[::-1],
                    fill='toself',
                    fillcolor=colors[i % len(colors)] + '30',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name=f'{label} 95% CI',
                    showlegend=False
                ))
            
            # 2. Survival Curve (KM Estimate)
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'{label}<br>Time: %{{x:.1f}}<br>Surv: %{{y:.3f}}<extra></extra>'
            ))

    fig.update_layout(
        title=f'Kaplan-Meier Survival Curves (Landmark Time: {landmark_time})',
        xaxis_title=f'Time Since Landmark ({duration_col} - {landmark_time})', # Important X-axis labeling
        yaxis_title='Survival Probability',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    fig.update_yaxes(range=[0, 1.05])

    # 5. Log-Rank Test (using New_Duration)
    stats_data = {}
    try:
        if len(groups) == 2:
            g1, g2 = groups
            res = logrank_test(
                landmark_data[landmark_data[group_col] == g1]['New_Duration'],
                landmark_data[landmark_data[group_col] == g2]['New_Duration'],
                event_observed_A=landmark_data[landmark_data[group_col] == g1][event_col],
                event_observed_B=landmark_data[landmark_data[group_col] == g2][event_col]
            )
            stats_data = {
                'Test': 'Log-Rank (Pairwise)',
                'Statistic': res.test_statistic,
                'P-value': res.p_value,
                'Comparison': f'{g1} vs {g2}',
                'Method': f'Landmark at {landmark_time}'
            }
        elif len(groups) > 2:
            res = multivariate_logrank_test(landmark_data['New_Duration'], landmark_data[group_col], landmark_data[event_col])
            stats_data = {
                'Test': 'Log-Rank (Multivariate)',
                'Statistic': res.test_statistic,
                'P-value': res.p_value,
                'Comparison': 'All groups',
                'Method': f'Landmark at {landmark_time}'
            }
        
    except Exception as e:
        stats_data = {'Test': 'Error', 'Note': str(e), 'Method': f'Landmark at {landmark_time}'}

    return fig, pd.DataFrame([stats_data]), n_pre_filter, n_post_filter, None

# --- 5. Report Generation ---
def generate_report_survival(title, elements):
# ... (โค้ด generate_report_survival เดิม ถูกต้องแล้ว)
    """
    Generate HTML report (Renamed to match tab_survival.py calls)
    """
    css_style = """<style>
        body{font-family:Arial;margin:20px;}
        table{border-collapse:collapse;width:100%;margin:10px 0;}
        th,td{border:1px solid #ddd;padding:8px;}
        th{background-color:#0066cc;color:white;}
        h1{color:#333;border-bottom:2px solid #0066cc;}
        h2{color:#0066cc;}
    </style>"""
    
    plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    
    safe_title = _html.escape(str(title))
    html_doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css_style}{plotly_js}</head><body><h1>{safe_title}</h1>"
    
    for el in elements:
        t = el.get('type')
        d = el.get('data')
        
        if t == 'header':
            html_doc += f"<h2>{_html.escape(str(d))}</h2>"
        elif t == 'text':
            html_doc += f"<p>{_html.escape(str(d))}</p>"
        elif t == 'preformatted':
            html_doc += f"<pre>{_html.escape(str(d))}</pre>"
        elif t == 'table':
            html_doc += d.to_html(classes='table')
        elif t == 'plot': 
            if hasattr(d, 'to_html'):
                html_doc += d.to_html(full_html=False, include_plotlyjs=False)
            elif hasattr(d, 'savefig'):
                 buf = io.BytesIO()
                 d.savefig(buf, format='png', bbox_inches='tight')
                 b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                 html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
        elif t == 'image': 
            b64 = base64.b64encode(d).decode('utf-8')
            html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
             
    html_doc += "</body></html>"
    return html
