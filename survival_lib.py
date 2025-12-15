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

# --- 1. Kaplan-Meier & Log-Rank ---
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

            # Survival Curve
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'{label}<br>Time: %{{x:.1f}}<br>Surv: %{{y:.3f}}<extra></extra>'
            ))

    fig.update_layout(
        title='Kaplan-Meier Survival Curves',
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

# --- 2. Nelson-Aalen ---
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

    naf = NelsonAalenFitter()

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
        title='Nelson-Aalen Cumulative Hazard',
        xaxis_title='Time',
        yaxis_title='Cumulative Hazard',
        template='plotly_white',
        height=500
    )

    return fig, pd.DataFrame(stats_list)

# --- 3. Cox Proportional Hazards (Robust Version) ---
def fit_cox_ph(df, duration_col, event_col, covariate_cols):
    """
    Fits CoxPH model with auto-retry and stability enhancements.
    Returns: (cph_object, results_df, model_data, error_message)
    """
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

    # 4. Progressive Fitting Loop (The Fix)
    # step_size < 1.0 helps convergence on difficult datasets
    penalizers = [0.0, 0.1, 1.0, 10.0]
    cph = None
    last_error = None

    for p in penalizers:
        try:
            # Use step_size=0.5 for better stability
            temp_cph = CoxPHFitter(penalizer=p)
            temp_cph.fit(data, duration_col=duration_col, event_col=event_col, step_size=0.5)
            cph = temp_cph
            break
        except Exception as e:
            last_error = e
            continue

    if cph is None:
        return None, None, data, f"Model Convergence Failed. Details: {last_error}.\nTry checking for high correlation between variables."

    # Format Results
    summary = cph.summary.copy()
    summary['HR'] = np.exp(summary['coef'])
    summary['95% CI Lower'] = np.exp(summary['lower 95% bound'])
    summary['95% CI Upper'] = np.exp(summary['upper 95% bound'])
    
    res_df = summary[['HR', '95% CI Lower', '95% CI Upper', 'p']].rename(columns={'p': 'P-value'})
    
    return cph, res_df, data, None

def check_cph_assumptions(cph, data):
    """
    Checks PH assumptions.
    Returns: (text_report, list_of_image_bytes)
    """
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

# --- 4. Report Generation ---
def generate_report_survival(title, elements):
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
    return html_doc
