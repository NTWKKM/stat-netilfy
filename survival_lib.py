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
import logging
from tabs._common import get_color_palette

# Get unified color palette
COLORS = get_color_palette()

_logger = logging.getLogger(__name__)

# Helper function for standardization
def _standardize_numeric_cols(data, cols) -> None:
    """
    Standardize numeric columns in-place while preserving binary (0/1) columns.
    
    Numeric columns listed in `cols` are centered to mean zero and scaled to unit variance.
    Columns containing only the values 0 and 1 are left unchanged. If a column has
    zero or undefined standard deviation, a warning is emitted and the column is not modified.
    
    Parameters:
        data (pandas.DataFrame): DataFrame whose columns will be standardized in-place.
        cols (Iterable[str]): Column names to consider for standardization.
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

# 🟢 NEW HELPER: Convert Hex to RGBA string for Plotly fillcolor
def _hex_to_rgba(hex_color, alpha) -> str:
    """Convert hex color to RGBA string. Expects 6-digit hex format (e.g., '#RRGGBB')."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: got {len(hex_color)} chars, expected 6")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'

# --- 1. Kaplan-Meier & Log-Rank (With Robust CI) 🟢 FIX KM CI ---
def fit_km_logrank(df, duration_col, event_col, group_col):
    """
    Fits KM curves and performs Log-rank test.
    Uses unified teal color palette from _common.py.
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
                
                # 🟢 FIX: Use RGBA string instead of 8-digit hex
                rgba_color = _hex_to_rgba(colors[i % len(colors)], 0.2) # Alpha 0.2 for transparency

                # 1. Add Shaded Area (Confidence Interval)
                fig.add_trace(go.Scatter(
                    x=list(ci_lower.index) + list(ci_upper.index)[::-1], # Times forward and backward
                    y=list(ci_lower.values) + list(ci_upper.values)[::-1], # CI lower forward, CI upper backward
                    fill='toself',
                    fillcolor=rgba_color, # 🟢 APPLIED FIX
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
    Fit Nelson-Aalen cumulative hazard curves optionally stratified by a grouping column and return a Plotly figure plus group-level statistics.
    Uses unified teal color palette from _common.py.
    
    Drops rows with missing duration or event values. If a group column is provided, rows with missing group values are dropped and curves are plotted per group; otherwise a single overall curve is plotted. When the fitter provides a confidence interval with at least two columns, a shaded 95% CI is added for each group.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing duration, event, and optional group columns.
        duration_col (str): Name of the column with follow-up time or duration.
        event_col (str): Name of the column with event indicator (1 for event, 0 for censored).
        group_col (str or None): Name of the column to stratify by, or None to compute an overall curve.
    
    Returns:
        fig (plotly.graph_objs.Figure): Plotly figure showing cumulative hazard curves and shaded 95% CIs when available.
        stats_df (pandas.DataFrame): DataFrame with one row per plotted group containing columns `Group`, `N`, and `Events`.
    
    Raises:
        ValueError: If no valid rows remain after dropping missing duration/event values, or if a specified group column is not present in `df`.
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
                
                # 🟢 FIX: Use RGBA string instead of 8-digit hex
                rgba_color = _hex_to_rgba(colors[i % len(colors)], 0.2) # Alpha 0.2 for transparency

                # 1. Add Shaded Area (Confidence Interval)
                fig.add_trace(go.Scatter(
                    x=list(ci_lower.index) + list(ci_upper.index)[::-1], 
                    y=list(ci_lower.values) + list(ci_upper.values)[::-1], 
                    fill='toself',
                    fillcolor=rgba_color, # 🟢 APPLIED FIX
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

# --- 3. Cox Proportional Hazards (Robust with Progressive L2 Penalization & Data Validation) ---
def fit_cox_ph(df, duration_col, event_col, covariate_cols):
    """
    Fit a Cox Proportional Hazards model after validating and preprocessing covariates.         
    
    Validates input columns and rows, performs automatic one-hot encoding for categorical covariates (drop_first=True), checks numeric covariates for infinite or extreme values, zero variance, potential perfect separation, and high multicollinearity, standardizes numeric covariates (skipping binary 0/1), and attempts a progressive fitting strategy (standard CoxPH then increasing L2 penalization) until a successful fit is obtained or all attempts fail.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing duration, event, and covariates.
        duration_col (str): Name of the duration/time-to-event column.
        event_col (str): Name of the event indicator column (0/1).
        covariate_cols (list[str]): List of covariate column names to include in the model.
    
    Returns:
        cph (lifelines.CoxPHFitter or None): Fitted CoxPHFitter instance on success, otherwise None.
        res_df (pandas.DataFrame or None): Results table with hazard ratios (`HR`), 95% CI bounds, `P-value`, and `Method` when fit succeeds; None on failure.
        data (pandas.DataFrame): Processed DataFrame used for fitting (after dropping missing rows and any encoding/standardization). Returned even on failure to aid debugging.
        error_message (str or None): Detailed error message when fitting or validation fails; None on success.
    """
    # 1. Basic Validation
    missing = [c for c in [duration_col, event_col, *covariate_cols] if c not in df.columns]
    if missing:
        return None, None, df, f"Missing columns: {missing}"

    # 🟢 FIX: Explicitly select ONLY relevant columns here to prevent unused columns from leaking into the model
    data = df[[duration_col, event_col, *covariate_cols]].dropna().copy()
    
    if len(data) == 0:
        return None, None, data, "No valid data after dropping missing values."

    if data[event_col].sum() == 0:
        return None, None, data, "No events observed (all censored). CoxPH requires at least one event." 

    # 🟢 NEW: Automatic One-Hot Encoding for Categorical/Object columns
    # Essential for Cox Regression to handle categorical variables
    original_covariate_cols = list(covariate_cols) # 🟢 Preserve original names for debugging
    try:
        covars_only = data[covariate_cols]
        # Find categorical/object columns
        cat_cols = [c for c in covariate_cols if not pd.api.types.is_numeric_dtype(data[c])]
        
        if cat_cols:
            # Use drop_first=True to prevent multicollinearity (dummy variable trap)
            covars_encoded = pd.get_dummies(covars_only, columns=cat_cols, drop_first=True)
            # Update data and covariate list
            data = pd.concat([data[[duration_col, event_col]], covars_encoded], axis=1)
            covariate_cols = covars_encoded.columns.tolist()
    except (ValueError, TypeError, KeyError) as e:
        return None, None, data, f"Encoding Error (Original vars: {original_covariate_cols}): Failed to convert categorical variables. {e}"
    
    # 🟢 NEW: Comprehensive Data Validation BEFORE attempting fit
    validation_errors = []
    
    for col in covariate_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            # Check 1: Infinite values
            if np.isinf(data[col]).any():
                n_inf = np.isinf(data[col]).sum()
                validation_errors.append(f"Covariate '{col}': Contains {n_inf} infinite values (Inf, -Inf). Check data source.")
            
            # Check 2: Extreme values (>±1e10)
            if (data[col].abs() > 1e10).any():
                max_val = data[col].abs().max()
                validation_errors.append(f"Covariate '{col}': Contains extreme values (max={max_val:.2e}). Consider scaling (divide by 1000, log transform, or standardize).")
            
            # Check 3: Zero variance (constant column)
            std = data[col].std()
            if pd.isna(std) or std == 0:
                validation_errors.append(f"Covariate '{col}': Has zero variance (constant values only). Remove this column.")
            
            # Check 4: Perfect separation (outcome completely predictable)
            try:
                outcomes_0 = data[data[event_col] == 0][col]
                outcomes_1 = data[data[event_col] == 1][col]
                
                if len(outcomes_0) > 0 and len(outcomes_1) > 0:
                    # Check if ranges completely separate (no overlap)
                    if (outcomes_0.max() < outcomes_1.min()) or (outcomes_1.max() < outcomes_0.min()):
                        validation_errors.append(f"Covariate '{col}': Perfect separation detected - outcomes completely separated by this variable. Try removing, combining with other variables, or grouping.")
            except Exception as e:
                _logger.debug("Perfect separation check failed for '%s': %s", col, e)
    
    # Check 5: Multicollinearity (high correlation between numeric covariates)
    numeric_covs = [c for c in covariate_cols if pd.api.types.is_numeric_dtype(data[c])]
    if len(numeric_covs) > 1:
        try:
            corr_matrix = data[numeric_covs].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        r = corr_matrix.iloc[i, j]
                        high_corr_pairs.append(f"{col_i} <-> {col_j} (r={r:.3f})")
            
            if high_corr_pairs:
                validation_errors.append("High multicollinearity detected (r > 0.95): " + ", ".join(high_corr_pairs) + ". Try removing one of each correlated pair.")
        except Exception as e:
            _logger.debug("Multicollinearity check failed: %s", e)
    
    # If validation errors found, report them NOW before trying to fit
    if validation_errors:
        error_msg = ("[DATA QUALITY ISSUES] Fix Before Fitting:\n\n" + 
                     "\n\n".join(f"[ERROR] {e}" for e in validation_errors))
        return None, None, data, error_msg
    
    # 2. Standardize (Skip binary to improve stability)
    _standardize_numeric_cols(data, covariate_cols)
    
    # 3. Fitting Strategy (Progressive Robustness)
    # Try: Standard -> L2(0.1) -> L2(1.0)
    penalizers = [
        {"p": 0.0, "name": "Standard CoxPH (Maximum Partial Likelihood)"},
        {"p": 0.1, "name": "L2 Penalized CoxPH (p=0.1) - Ridge Regression"},
        {"p": 1.0, "name": "L2 Penalized CoxPH (p=1.0) - Strong Regularization"}
    ]
    
    cph = None
    last_error = None
    method_used = None
    methods_tried = []  # 🟢 Track methods for error reporting

    for conf in penalizers:
        p = conf['p']
        current_method = conf['name']
        
        methods_tried.append(current_method)
        
        try:
            temp_cph = CoxPHFitter(penalizer=p) 
            # 🎫 FIX: Removed invalid step_size parameter
            # CoxPHFitter.fit() only accepts: duration_col, event_col, show_progress
            temp_cph.fit(data, duration_col=duration_col, event_col=event_col, show_progress=False)
            cph = temp_cph
            method_used = current_method  # ✅ SET on success
            break  # Stop trying - success!
        except Exception as e:
            last_error = e
            continue

    # 4. Error handling
    if cph is None:
        # 🟢 Show which methods were tried + troubleshooting guide
        methods_str = "\n".join(f"  [ERROR] {m}" for m in methods_tried)
        error_msg = (
            f"Cox Model Convergence Failed\n\n"
            f"Fitting Methods Attempted:\n{methods_str}\n\n"
            f"Last Error: {last_error!s}\n\n"
            f"Troubleshooting Guide:\n"
            f"  1. Verify your data passed validation checks above\n"
            f"  2. Try removing ONE covariate at a time to isolate the problem\n"
            f"  3. For categorical variables: Check if categories separated from outcome\n"
            f"  4. Try scaling numeric variables to 0-100 or 0-1 range\n"
            f"  5. Check for rare categories in categorical variables\n"
            f"  6. Try with fewer covariates (e.g., 2-3 instead of many)\n"
            f"  7. See: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model"
        )
        return None, None, data, error_msg

    # 5. Format Results
    summary = cph.summary.copy()
    summary['HR'] = np.exp(summary['coef'])
    ci = cph.confidence_intervals_
    summary['95% CI Lower'] = np.exp(ci.iloc[:, 0])
    summary['95% CI Upper'] = np.exp(ci.iloc[:, 1])

    # Add Method used to results table
    summary['Method'] = method_used # 🟢 Show which method succeeded
    summary.index.name = "Covariate"

    res_df = summary[['HR', '95% CI Lower', '95% CI Upper', 'p', 'Method']].rename(columns={'p': 'P-value'})
    
    return cph, res_df, data, None

def check_cph_assumptions(cph, data):
    try:
        # 1. Statistical Test
        results = proportional_hazard_test(cph, data, time_transform='rank')
        text_report = "Proportional Hazards Test Results:\n" + results.summary.to_string()
        
        # 2. Schoenfeld Residual Plots
        img_bytes_list = []
        
        # Compute residuals
        scaled_schoenfeld = cph.compute_residuals(data, 'scaled_schoenfeld')
        
        # 🟢 FIX: Align 'times' with the residuals (residuals only exist for events)
        # scaled_schoenfeld index corresponds to the original data index for event rows
        times = data.loc[scaled_schoenfeld.index, cph.duration_col]
        
        for col in scaled_schoenfeld.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Use the aligned 'times'
            ax.scatter(times, scaled_schoenfeld[col], alpha=0.5)
            
            # Trend line (optional)
            try:
                z = np.polyfit(times, scaled_schoenfeld[col], 1)
                p = np.poly1d(z)
                sorted_times = np.sort(times)
                ax.plot(sorted_times, p(sorted_times), "r--", alpha=0.8)
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


# --- 🟢 NEW: Create interactive Plotly forest plot for Cox Regression (Web UI) ---
def create_forest_plot_cox(res_df):
    """
    Create an interactive Plotly forest plot for Cox regression hazard ratios.
    Used for visualization in Streamlit web UI.
    
    Parameters:
        res_df (pandas.DataFrame): Results DataFrame with columns 'HR', '95% CI Lower', '95% CI Upper', 'P-value'.
    
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object.
    """
    if res_df is None or res_df.empty:
        raise ValueError("No Cox regression results available for forest plot.")
    
    # Prepare data
    variables = res_df.index.tolist()
    hrs = res_df['HR'].values
    ci_lows = res_df['95% CI Lower'].values
    ci_highs = res_df['95% CI Upper'].values
    p_vals = res_df['P-value'].values
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add HR points and CI error bars
    fig.add_trace(go.Scatter(
        x=hrs,
        y=variables,
        mode='markers',
        marker=dict(
            size=10,
            color=COLORS['success'],  # Use success color from palette
            line=dict(width=2, color='rgba(255,255,255,0.8)')
        ),
        name='Hazard Ratio',
        hovertemplate='<b>%{y}</b><br>HR: %{x:.4f}<extra></extra>'
    ))
    
    # Add error bars (CI)
    fig.add_trace(go.Scatter(
        x=ci_highs,
        y=variables,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='none'
    ))
    
    fig.add_trace(go.Scatter(
        x=ci_lows,
        y=variables,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='none',
        fill='tonextx',
        fillcolor=_hex_to_rgba(COLORS['success'].lstrip('#'), 0.2) if COLORS['success'].startswith('#') else 'rgba(50,184,198,0.2)',
        name='95% CI'
    ))
    
    # Add vertical line at HR = 1 (null effect)
    fig.add_vline(
        x=1,
        line_dash='dash',
        line_color='rgba(192, 21, 47, 0.5)',
        annotation_text="HR = 1 (No Effect)",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title='🌳 Forest Plot: Cox Regression Hazard Ratios (95% CI)',
        xaxis_title='Hazard Ratio (log scale)',
        yaxis_title='Variable',
        xaxis_type='log',
        template='plotly_white',
        height=max(400, len(variables) * 40),
        hovermode='y unified',
        margin=dict(l=200, r=100)
    )
    
    return fig


# --- 🟢 NEW: Generate Forest Plot HTML for Cox Regression (HTML Report) ---
def generate_forest_plot_cox_html(res_df):
    """
    Generate an HTML forest plot for Cox regression hazard ratios using Plotly.
    Embeds Plotly JS locally (no CDN needed) for offline-friendly reports.
    
    Parameters:
        res_df (pandas.DataFrame): Results DataFrame with columns 'HR', '95% CI Lower', '95% CI Upper', 'P-value'.
    
    Returns:
        html_str (str): HTML string containing the forest plot as Plotly embed + summary table + interpretation.
    """
    if res_df is None or res_df.empty:
        return "<p>No Cox regression results available for forest plot.</p>"
    
    # Create interactive forest plot (same function as web UI)
    fig = create_forest_plot_cox(res_df)
    
    # 🟢 CRITICAL FIX: Include Plotly JS with CDN (will be used by generate_report_survival)
    plot_html = fig.to_html(include_plotlyjs=False, div_id='cox_forest_plot')
    
    # Prepare data for summary table
    variables = res_df.index.tolist()
    hrs = res_df['HR'].values
    ci_lows = res_df['95% CI Lower'].values
    ci_highs = res_df['95% CI Upper'].values
    p_vals = res_df['P-value'].values
    
    # Create summary table HTML
    table_html = "<h3>Summary Table: Hazard Ratios</h3>"
    table_html += "<table style='border-collapse: collapse; width: 100%; margin: 10px 0;'>"
    table_html += "<tr style='background-color: " + COLORS.get('primary', '#1f8085') + "; color: white;'>"
    table_html += "<th style='border: 1px solid #ddd; padding: 8px;'>Variable</th>"
    table_html += "<th style='border: 1px solid #ddd; padding: 8px;'>HR</th>"
    table_html += "<th style='border: 1px solid #ddd; padding: 8px;'>95% CI Lower</th>"
    table_html += "<th style='border: 1px solid #ddd; padding: 8px;'>95% CI Upper</th>"
    table_html += "<th style='border: 1px solid #ddd; padding: 8px;'>P-value</th>"
    table_html += "</tr>"
    
    for i, var in enumerate(variables):
        sig = "✅" if p_vals[i] < 0.05 else "⚠️"
        table_html += f"<tr style='background-color: #f8f9fa;'>"
        table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'><b>{var}</b></td>"
        table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{hrs[i]:.4f}</td>"
        table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{ci_lows[i]:.4f}</td>"
        table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{ci_highs[i]:.4f}</td>"
        table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{p_vals[i]:.4f} {sig}</td>"
        table_html += "</tr>"
    
    table_html += "</table>"
    
    # Interpretation guide
    interp_html = """
    <h3>💡 Interpretation Guide</h3>
    <ul>
        <li><b>HR > 1:</b> Increased hazard (Risk Factor) 🔴</li>
        <li><b>HR < 1:</b> Decreased hazard (Protective Factor) 🟢</li>
        <li><b>HR = 1:</b> No effect (null)</li>
        <li><b>CI crosses 1.0:</b> Not statistically significant ⚠️</li>
        <li><b>CI doesn't cross 1.0:</b> Statistically significant ✅</li>
        <li><b>P < 0.05:</b> Statistically significant ✅</li>
    </ul>
    """
    
    return f"<div style='margin: 20px 0;'>{plot_html}{table_html}{interp_html}</div>"

# --- 4. Landmark Analysis (KM) 🟢 FIX LM CI ---
def fit_km_landmark(df, duration_col, event_col, group_col, landmark_time):
    """
    Perform Kaplan-Meier survival analysis using a landmark-time approach.
    Uses unified teal color palette from _common.py.
                
    Parameters:
        df (pandas.DataFrame): Input data containing duration, event indicator, and group columns.
        duration_col (str): Name of the column with observed times-to-event.
        event_col (str): Name of the column with event indicator (1=event, 0=censored).
        group_col (str): Name of the grouping/stratification column; required for stratified curves and tests.
        landmark_time (numeric): Time threshold used for the landmark analysis; only records with duration >= landmark_time are included and their times are re-based to time since landmark.
    
    Returns:
        fig (plotly.graph_objects.Figure or None): Plotly figure showing Kaplan–Meier curves and shaded 95% confidence intervals for each group, or None if an error occurred before plotting.
        stats_df (pandas.DataFrame or None): Single-row DataFrame summarizing the performed log-rank test (test name, statistic, p-value, comparison, Method) or a DataFrame describing an error/note; None if not applicable.
        n_pre_filter (int): Number of records remaining after dropping rows with missing duration, event, or group (before applying the landmark filter).
        n_post_filter (int): Number of records remaining after applying the landmark filter (duration >= landmark_time).
        error (str or None): Error message when the function fails early (e.g., missing columns or insufficient records), otherwise None.
    """
    # 1. Data Cleaning
    missing = [c for c in [duration_col, event_col, group_col] if c not in df.columns]
    if missing:
        return None, None, len(df), 0, f"Missing columns: {missing}"

    data = df.dropna(subset=[duration_col, event_col, group_col])
    n_pre_filter = len(data)

    # 2. Filtering (The Landmark Step)
    landmark_data = data[data[duration_col] >= landmark_time].copy()
    n_post_filter = len(landmark_data)
    
    if n_post_filter < 2:
        return None, None, n_pre_filter, n_post_filter, "Error: Insufficient patients (N < 2) survived until the landmark time."
    
    # 3. Recalculate Duration (Crucial Step)
    _adj_duration = '_landmark_adjusted_duration'
    landmark_data[_adj_duration] = landmark_data[duration_col] - landmark_time
    
    # 4. KM Fitting (Standardized Plotting)
    groups = sorted(landmark_data[group_col].unique(), key=lambda v: str(v))
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, g in enumerate(groups):
        df_g = landmark_data[landmark_data[group_col] == g]
        label = f"{group_col}={g}"
        
        if len(df_g) > 0:
            kmf = KaplanMeierFitter()
            
            # Fit using the adjusted duration
            kmf.fit(df_g[_adj_duration], df_g[event_col], label=label)

            # --- 🟢 FIX: Check existence and access CI by position ---
            ci_exists = hasattr(kmf, 'confidence_interval_') and not kmf.confidence_interval_.empty
            
            if ci_exists and kmf.confidence_interval_.shape[1] >= 2:
                # Use .iloc[:, 0] for lower bound and .iloc[:, 1] for upper bound
                ci_lower = kmf.confidence_interval_.iloc[:, 0]
                ci_upper = kmf.confidence_interval_.iloc[:, 1]
                
                # 🟢 FIX: Use RGBA string instead of 8-digit hex
                rgba_color = _hex_to_rgba(colors[i % len(colors)], 0.2) # Alpha 0.2 for transparency

                # 1. Add Shaded Area (Confidence Interval)
                fig.add_trace(go.Scatter(
                    x=list(ci_lower.index) + list(ci_upper.index)[::-1],
                    y=list(ci_lower.values) + list(ci_upper.values)[::-1],
                    fill='toself',
                    fillcolor=rgba_color, # 🟢 APPLIED FIX
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
                landmark_data[landmark_data[group_col] == g1][_adj_duration],
                landmark_data[landmark_data[group_col] == g2][_adj_duration],
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
            res = multivariate_logrank_test(landmark_data[_adj_duration], landmark_data[group_col], landmark_data[event_col])
            stats_data = {
                'Test': 'Log-Rank (Multivariate)',
                'Statistic': res.test_statistic,
                'P-value': res.p_value,
                'Comparison': 'All groups',
                'Method': f'Landmark at {landmark_time}'
            }
        
        else:
            stats_data = {
                'Test': 'None',
                'Note': 'Single group or no group at landmark',
                'Method': f'Landmark at {landmark_time}',
            }

    except Exception as e:
        stats_data = {'Test': 'Error', 'Note': str(e), 'Method': f'Landmark at {landmark_time}'}

    return fig, pd.DataFrame([stats_data]), n_pre_filter, n_post_filter, None

# --- 5. Report Generation 🟢 FIXED: Embed Plotly JS in Head for All Reports ---
def generate_report_survival(title, elements):
    """
    Assemble a complete HTML report from a sequence of content elements, embedding tables, figures, and images for offline-friendly consumption.
    Uses unified teal color palette from _common.py.
    
    Builds an HTML document with the given title and iterates over `elements` to render supported content types. For Plotly figures, Plotly JS is embedded in the <head> once and reused for all plots. Supported element types and expected `data` values:
    - "header": a string rendered as an H2 section header.
    - "text": a plain string rendered as a paragraph.
    - "preformatted": a string rendered inside a <pre> block.
    - "table": a pandas DataFrame (or DataFrame-like) rendered via DataFrame.to_html().
    - "plot": a Plotly Figure-like object (with to_html) or a Matplotlib Figure-like object (with savefig).
    - "image": raw image bytes (PNG) which will be embedded as a base64 data URL.
    - "html": raw HTML string to embed directly (used for forest plots).
    
    Parameters:
        title: The report title; will be HTML-escaped.
        elements: An iterable of dicts describing report elements; each dict should include keys
            'type' (one of the supported types above) and 'data' (the corresponding content).
    
    Returns:
        html_doc (str): A self-contained HTML string representing the assembled report with embedded Plotly JS.
    """
    
    primary_color = COLORS['primary']
    primary_dark = COLORS['primary_dark']
    text_color = COLORS['text']
    
    css_style = f"""<style>
        body{{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: {text_color};
            line-height: 1.6;
        }}
        h1{{
            color: {primary_dark};
            border-bottom: 3px solid {primary_color};
            padding-bottom: 12px;
            font-size: 2em;
            margin-bottom: 20px;
        }}
        h2{{
            color: {primary_dark};
            border-left: 5px solid {primary_color};
            padding-left: 12px;
            margin: 25px 0 15px 0;
        }}
        h3{{
            color: {primary_dark};
            margin: 15px 0 10px 0;
        }}
        table{{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            background-color: white;
            border-radius: 6px;
        }}
        th, td{{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th{{
            background-color: {primary_color};
            color: white;
            font-weight: 600;
        }}
        tr:hover{{
            background-color: #f8f9fa;
        }}
        tr:nth-child(even){{
            background-color: #fcfcfc;
        }}
        p{{
            margin: 12px 0;
            color: {text_color};
        }}
        pre{{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid {primary_color};
        }}
        ul, ol {{
            margin: 12px 0;
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
        }}
        .report-footer {{
            text-align: center;
            font-size: 0.75em;
            color: #666;
            margin-top: 40px;
            border-top: 1px dashed #ccc;
            padding-top: 10px;
        }}
        .report-footer a {{
            color: {primary_color};
            text-decoration: none;
        }}
        .report-footer a:hover {{
            color: {primary_dark};
            text-decoration: underline;
        }}
    </style>"""
    
    safe_title = _html.escape(str(title))
    # 🟢 FIXED: Include Plotly JS from CDN in <head>
    html_doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-latest.min.js'></script>{css_style}</head><body><h1>{safe_title}</h1>"""
    
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
            html_doc += d.to_html()
        elif t == 'plot':
            if hasattr(d, 'to_html'):
                # 🟢 FIX: Don't include Plotly JS in plots (already loaded in head)
                html_doc += d.to_html(full_html=False, include_plotlyjs=False)
            elif hasattr(d, 'savefig'):
                buf = io.BytesIO()
                d.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
        elif t == 'image':
            b64 = base64.b64encode(d).decode('utf-8')
            html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
        elif t == 'html':
            # 🟢 NEW: Embed raw HTML (used for forest plots)
            html_doc += str(d)
    
    html_doc += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM n Donate</a> | All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div>"""
    html_doc += "</body>\n</html>"
    
    return html_doc
