import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import plotly.graph_objects as go
import plotly.express as px
import warnings
import io, base64
import html as _html

# Helper function for standardization
def _standardize_numeric_cols(data, cols):
    """Standardize numeric columns in-place, warning on zero-variance."""
    for col in cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            std = data[col].std()
            if std == 0:
                warnings.warn(f"Covariate '{col}' has zero variance", stacklevel=3)
            else:
                data[col] = (data[col] - data[col].mean()) / std

# --- 1. Kaplan-Meier Estimator (with Plotly) ---
def estimate_km(df, duration_col, event_col, group_col=None, group_val=None):
    """ 
    Fit Kaplan-Meier survival curves using the provided dataframe.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing duration, event, and optional grouping columns.
        duration_col (str): Column name with event/censoring times.
        event_col (str): Column name with event indicator (1=event, 0=censored).
        group_col (str, optional): Column name for group/stratification variable.
        group_val (optional): Specific group value to subset (if None, uses entire dataset).
    
    Returns:
        tuple: (kmf, fig)
            kmf (lifelines.KaplanMeierFitter): Fitted KM estimator object.
            fig (plotly.graph_objects.Figure): Interactive Plotly figure showing KM curve with confidence intervals.
    """
    data = df.dropna(subset=[duration_col, event_col])
    
    if len(data) == 0:
        raise ValueError("No valid data after removing missing values")
        
    if group_col is not None:
        if group_val is not None:
            data = data[data[group_col] == group_val]
            if len(data) == 0:
                raise ValueError(f"No data for group {group_col}={group_val}")

    if data[event_col].sum() == 0:
        warnings.warn("No events observed in the data; survival curve will be flat at 1.0.", stacklevel=2)
    
    kmf = KaplanMeierFitter()
    label = "Overall" if (group_col is None or group_val is None) else f"{group_col}={group_val}"
    kmf.fit(data[duration_col], data[event_col], label=label)
    
    # 🟢 UPDATED: สร้างกราฟ Plotly แทน Matplotlib
    fig = go.Figure()
    
    # เพิ่ม survival curve
    fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_.iloc[:, 0],
        mode='lines',
        name='Survival Probability',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x:.1f}<br>Survival: %{y:.3f}<extra></extra>'
    ))
    
    # 🟢 MODIFIED: Add safety check for Confidence Interval before plotting
    ci_available = (
        hasattr(kmf, 'confidence_interval_survival_function_') and 
        kmf.confidence_interval_survival_function_ is not None and
        kmf.confidence_interval_survival_function_.shape[1] >= 2
    )

    if ci_available:
        # เพิ่ม confidence interval (upper)
        ci_upper = kmf.confidence_interval_survival_function_.iloc[:, 1]
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=ci_upper,
            mode='lines',
            name='95% CI Upper',
            line=dict(color='rgba(0, 0, 255, 0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # เพิ่ม confidence interval (lower)
        ci_lower = kmf.confidence_interval_survival_function_.iloc[:, 0]
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=ci_lower,
            mode='lines',
            name='95% CI Lower',
            line=dict(color='rgba(0, 0, 255, 0)'),
            fillcolor='rgba(0, 100, 200, 0.2)',
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        warnings.warn("Confidence interval for KM curve could not be computed or is missing.", stacklevel=2)
    
    # ปรับแต่งเค้าโครง
    fig.update_layout(
        title='Kaplan-Meier Survival Curve',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        width=800,
        font=dict(size=12)
    )
    
    # ปรับแต่ง y-axis range
    fig.update_yaxes(range=[0, 1.05])
    
    return kmf, fig


def fit_km_logrank(df, duration_col, event_col, group_col):
    """ 
    Compare Kaplan-Meier survival curves between groups and perform log-rank test.
    
    Parameters:
        df (pandas.DataFrame): Input dataset.
        duration_col (str): Column name with event/censoring times.
        event_col (str): Column name with event indicator (1=event, 0=censored).
        group_col (str): Column name for group/stratification variable.
    
    Returns:
        tuple: (fig, test_result)
            fig (plotly.graph_objects.Figure): Interactive Plotly figure comparing survival curves by group.
            test_result (dict): Log-rank test results (statistic, p-value, degrees of freedom, test name).
    """
    data = df.dropna(subset=[duration_col, event_col, group_col])

    if len(data) == 0:
        raise ValueError("No valid data after removing missing values")    
    
    groups = sorted(data[group_col].unique(), key=lambda v: str(v))

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups for comparison, found {len(groups)}")
    
    # 🟢 UPDATED: ใช้ Plotly สำหรับการวาดกราฟ
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, g in enumerate(groups):
        group_data = data[data[group_col] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(group_data[duration_col], group_data[event_col], label=f"{group_col}={g}")
        
        color = colors[i % len(colors)]
        
        # เพิ่ม survival curve
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_.iloc[:, 0],
            mode='lines',
            name=f'{g} (n={len(group_data)})',
            line=dict(color=color, width=2),
            hovertemplate=f'{group_col}={g}<br>Time: %{{x:.1f}}<br>Survival: %{{y:.3f}}<extra></extra>'
        ))
    
    # ปรับแต่งเค้าโครง
    fig.update_layout(
        title=f'Kaplan-Meier Survival Curves by {group_col}',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        width=900,
        font=dict(size=12)
    )
    
    fig.update_yaxes(range=[0, 1.05])
    
    # 🟢 MODIFIED: Log-rank test logic for 2 groups vs >2 groups
    if len(groups) == 2:
        # Standard Log-rank test for 2 groups
        g1, g2 = groups
        data_g1 = data[data[group_col] == g1]
        data_g2 = data[data[group_col] == g2]
        
        result = logrank_test(
            data_g1[duration_col],
            data_g2[duration_col],
            event_observed_A=data_g1[event_col],
            event_observed_B=data_g2[event_col]
        )
        
        test_result = {
            'Test': 'Log-Rank (Pairwise)',
            'Statistic': result.test_statistic,
            'P-value': result.p_value,
            'Degrees of Freedom': 1,
            'Comparison': f'{g1} vs {g2}'
        }
    else:
        # Multivariate log-rank test for >2 groups (Omnibus test)
        result = multivariate_logrank_test(data[duration_col], data[group_col], data[event_col])
        
        test_result = {
            'Test': 'Log-Rank (Omnibus)',
            'Statistic': result.test_statistic,
            'P-value': result.p_value,
            'Degrees of Freedom': len(groups) - 1,
            'Comparison': 'All groups combined'
        }
    
    return fig, test_result


# --- 2. Cox Proportional Hazards Model (Standard) ---
def fit_cox_model(df, duration_col, event_col, covariate_cols):
    """ 
    Fit a Cox proportional hazards model using provided covariates.
    
    Parameters:
        df (pandas.DataFrame): Input dataset.
        duration_col (str): Column name with event/censoring times.
        event_col (str): Column name with event indicator (1=event, 0=censored).
        covariate_cols (list[str]): List of covariate column names.
    
    Returns:
        tuple: (cph, fig_forest, fig_cumhaz)
            cph (lifelines.CoxPHFitter): Fitted Cox model object.
            fig_forest (plotly.graph_objects.Figure): Interactive forest plot (hazard ratios with CIs).
            fig_cumhaz (plotly.graph_objects.Figure): Cumulative hazard plot (for assumption checking).
    """
    missing = [c for c in [duration_col, event_col, *covariate_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    data = df.dropna(subset=[duration_col, event_col, *covariate_cols]).copy()

    if len(data) == 0:
        raise ValueError("No valid data after removing missing values")
    
    if data[event_col].sum() == 0:
        raise ValueError("No events observed in the data")
        
    # Standardize numeric covariates
    _standardize_numeric_cols(data, covariate_cols)
    
    cph = CoxPHFitter()
    try:
        cph.fit(data, duration_col=duration_col, event_col=event_col)
    except Exception as e:
        raise RuntimeError(f"Cox model fitting failed: {e}") from e
    
    # 🟢 UPDATED: สร้าง Forest Plot ด้วย Plotly
    # Extract hazard ratios and confidence intervals
    hr = np.exp(cph.params_)
    ci = np.exp(cph.confidence_intervals_)
    
    variables = hr.index.tolist()
    hr_vals = hr.values
    ci_lower = ci.iloc[:, 0].values
    ci_upper = ci.iloc[:, 1].values
    
    fig_forest = go.Figure()
    
    # เพิ่ม points (Hazard Ratios)
    fig_forest.add_trace(go.Scatter(
        x=hr_vals,
        y=variables,
        mode='markers',
        marker=dict(size=8, color='darkblue'),
        name='HR',
        hovertemplate='%{y}<br>HR: %{x:.3f}<extra></extra>'
    ))
    
    # เพิ่ม error bars (Confidence Intervals)
    fig_forest.add_trace(go.Scatter(
        x=hr_vals,
        y=variables,
        error_x=dict(
            type='data',
            symmetric=False,
            array=ci_upper - hr_vals,
            arrayminus=hr_vals - ci_lower,
            visible=True,
            color='darkblue',
            thickness=2
        ),
        mode='markers',
        marker=dict(size=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # เพิ่มเส้นอ้างอิง (HR = 1)
    fig_forest.add_vline(
        x=1,
        line_dash='dash',
        line_color='red',
        opacity=0.5,
        annotation_text='HR = 1 (No Effect)',
        annotation_position='top right'
    )
    
    # ปรับแต่งเค้าโครง
    fig_forest.update_layout(
        title='Cox Model: Hazard Ratios with 95% CI',
        xaxis_title='Hazard Ratio (log scale)',
        yaxis_title='Variable',
        xaxis_type='log',
        hovermode='y unified',
        template='plotly_white',
        height=min(400 + len(variables) * 30, 1200),  # Cap at reasonable max
        width=800,
        font=dict(size=11)
    )
    
    # Cumulative Hazard Plot (for assumption checking)
    fig_cumhaz = go.Figure()
    
    cumhaz = cph.cumulative_hazard_
    fig_cumhaz.add_trace(go.Scatter(
        x=cumhaz.index,
        y=cumhaz.iloc[:, 0],
        mode='lines',
        name='Cumulative Hazard',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x:.1f}<br>Cumulative Hazard: %{y:.3f}<extra></extra>'
    ))
    
    fig_cumhaz.update_layout(
        title='Cox Model: Cumulative Hazard Function',
        xaxis_title='Time',
        yaxis_title='Cumulative Hazard',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        width=800,
        font=dict(size=12)
    )
    
    return cph, fig_forest, fig_cumhaz

# --- 2. Cox Proportional Hazards Model (Time-Varying) ---
def fit_cox_time_varying(df, id_col, event_col, start_col, stop_col, covariate_cols):
    """
    Fit a Cox Time-Varying Proportional Hazards model using lifelines.CoxTimeVaryingFitter.
    
    Parameters:
        df (pandas.DataFrame): Data in long (start-stop) format.
        id_col (str): Column identifying individuals.
        event_col (str): Event indicator column.
        start_col (str): Start time column.
        stop_col (str): Stop time column.
        covariate_cols (list[str]): List of covariate columns.
        
    Returns:
        tuple: (ctv, results_df, data_used, error_msg)
            ctv (lifelines.CoxTimeVaryingFitter): Fitted model object.
            results_df (pandas.DataFrame): Formatted results (HR, CI, P-value).
            data_used (pandas.DataFrame): Data used after dropping NAs.
            error_msg (str | None): Error message if fitting fails, otherwise None.
    """
    
    # 1. Prepare data
    required_cols = [id_col, start_col, stop_col, event_col, *covariate_cols]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, None, df, f"Missing required columns: {missing}"
    data = df.dropna(subset=required_cols).copy()

    if len(data) < 2:
        return None, None, data, "Insufficient data after removing missing values."
    
    if data[event_col].sum() == 0:
        return None, None, data, "No events observed in the data."

    # 2. Fit model
    ctv = CoxTimeVaryingFitter()
    formula = " + ".join(covariate_cols)
    try:
        # CoxTimeVaryingFitter.fit uses the columns directly for start, stop, and event
        ctv.fit(
            data,
            id_col=id_col,
            event_col=event_col,
            start_col=start_col,
            stop_col=stop_col,
            formula=formula  # Pass covariates via formula
        )
    except Exception as e:
        return None, None, data, f"Cox Time-Varying model fitting failed: {e}"
        
    # 3. Format results
    summary = ctv.summary.copy()
    
    # Calculate HR and CI for output consistency with CoxPHFitter
    summary['HR'] = np.exp(summary['coef'])
    summary['95% CI Lower'] = np.exp(summary['lower 95% bound'])
    summary['95% CI Upper'] = np.exp(summary['upper 95% bound'])
    summary['P-value'] = summary['p']
    
    # Select and reorder columns for display
    results_df = summary[['HR', '95% CI Lower', '95% CI Upper', 'P-value']].sort_index()

    return ctv, results_df, data, None

def check_cox_assumptions(df, duration_col, event_col, covariate_cols):
    """ 
    Check Cox proportional hazards assumptions using Schoenfeld residuals.
    Returns a summary of assumption tests.
    
    Parameters:
        df (pandas.DataFrame): Input dataset.
        duration_col (str): Column name with event/censoring times.
        event_col (str): Column name with event indicator (1=event, 0=censored).
        covariate_cols (list[str]): List of covariate column names.
    
    Returns:
        str: HTML-formatted summary of proportional hazards test results.
    """
    data = df.dropna(subset=[duration_col, event_col, *covariate_cols]).copy()

    if len(data) == 0:
        raise ValueError("No valid data after removing missing values")
      
    # Standardize numeric covariates
    _standardize_numeric_cols(data, covariate_cols)
    
    cph = CoxPHFitter()
    try:
        cph.fit(data, duration_col=duration_col, event_col=event_col)
    except Exception as e:
        raise RuntimeError(f"Cox model fitting failed: {e}") from e
    
    # Test proportional hazards assumption
    from lifelines.statistics import proportional_hazard_test
    
    try:
        ph_test_results = proportional_hazard_test(cph, data, time_transform='rank')
    except Exception as e:
        raise RuntimeError(f"Proportional hazards test failed: {e}") from e
    
    html_output = "<h3>Proportional Hazards Test Results</h3>"
    html_output += "<table border='1' cellpadding='10'>"
    html_output += "<tr><th>Variable</th><th>Test Statistic</th><th>P-value</th><th>Interpretation</th></tr>"
    
    for var in ph_test_results.index:
        stat = ph_test_results.loc[var, 'test_statistic']
        p_val = ph_test_results.loc[var, 'p']
        interpretation = "✓ Assumption held" if p_val > 0.05 else "✗ Assumption violated"
        html_output += f"<tr><td>{_html.escape(str(var))}</td><td>{stat:.4f}</td><td>{p_val:.4f}</td><td>{interpretation}</td></tr>"
    
    html_output += "</table>"
    
    return html_output


# --- 3. Report Generation (with Plotly support) ---
def generate_survival_report(title, elements):
    """ 
    Generate a complete HTML report containing survival analysis results.
    
    Parameters:
        title (str): Report title displayed at the top of the page.
        elements (list[dict]): Ordered list of report elements. Each element must include:
            - type (str): One of 'text', 'header', 'table', 'preformatted', 'plot', or 'image'.
            - data: Content for the element (see generate_survival_report for details).
    
    Returns:
        str: Complete HTML document as a string, styled and containing the rendered elements.
    """
    css_style = """ 
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
        p {
            line-height: 1.6;
            color: #333;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
    """

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    html_doc = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{css_style}{plotly_cdn}</head>\n<body>"
    html_doc += f"<h1>{_html.escape(str(title))}</h1>"
    
    for el in elements:
        element_type = el.get('type')
        data = el.get('data')
        
        if element_type == 'text':
            html_doc += f"<p>{_html.escape(str(data))}</p>"
        
        elif element_type == 'preformatted':
            html_doc += f"<pre>{_html.escape(str(data))}</pre>"
        
        elif element_type == 'header':
            html_doc += f"\n<h2>{_html.escape(str(data))}</h2>"
        
        elif element_type == 'table':
            html_doc += data.to_html(classes='table')
        
        elif element_type == 'plot':
            # ตรวจสอบว่าเป็น Plotly Figure หรือ Matplotlib Figure
            plot_obj = data
            
            # ถ้าเป็น Plotly Figure
            if hasattr(plot_obj, 'to_html'):
                html_doc += plot_obj.to_html(full_html=False, include_plotlyjs=False, div_id=f"plot_{id(plot_obj)}")
            else:
                # ถ้าเป็น Matplotlib Figure - แปลงเป็น PNG และ embed
                buf = io.BytesIO()
                plot_obj.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%; margin: 20px 0;" />'
        
        elif element_type == 'image':
            # el['data'] คือ bytes อยู่แล้ว
            uri = base64.b64encode(data).decode('utf-8')
            html_doc += f'<img src="data:image/png;base64,{uri}" style="max-width:100%; margin: 20px 0;" />'
    
    html_doc += """ 
    """
    html_doc += "</body>\n</html>"
    
    return html_doc
