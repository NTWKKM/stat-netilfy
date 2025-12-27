import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
# 🟢 NEW: Import utility for Median CI calculation
from lifelines.utils import median_survival_times 
import plotly.graph_objects as go
import plotly.express as px
# import matplotlib.pyplot as plt  <-- REMOVED: No longer needed
import warnings
import io, base64
import html as _html
import logging
from tabs._common import get_color_palette
from forest_plot_lib import create_forest_plot  # 🟢 IMPORT NEW LIBRARY

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

# 🟢 HELPER: Convert Hex to RGBA string for Plotly fillcolor
def _hex_to_rgba(hex_color, alpha) -> str:
    """
    Convert a 6-digit hex color to a CSS-style `rgba(r,g,b,a)` string.
    
    Parameters:
        hex_color (str): Hex color string in `RRGGBB` format, with or without a leading `#`.
        alpha (float): Opacity value between 0 and 1.
    
    Returns:
        str: RGBA string formatted as `rgba(r,g,b,a)` where `r`, `g`, `b` are integers 0–255 and `a` is the provided alpha.
    
    Raises:
        ValueError: If `hex_color` does not contain exactly 6 hexadecimal characters after removing a leading `#`.
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: got {len(hex_color)} chars, expected 6")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'

# --- 🟢 NEW: Calculate Median Survival Table ---
def calculate_median_survival(df, duration_col, event_col, group_col):
    """
    Calculate Median Survival Time and 95% CI for each group.
    
    Returns:
        pd.DataFrame: A table containing 'Group', 'N', 'Events', and 'Median (95% CI)'.
    """
    # Validate required columns exist
    missing = []
    if duration_col not in df.columns: missing.append(duration_col)
    if event_col not in df.columns: missing.append(event_col)
    if group_col and group_col not in df.columns: missing.append(group_col)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    data = df.dropna(subset=[duration_col, event_col])
    
    # Validate numeric data types
    if not pd.api.types.is_numeric_dtype(data[duration_col]):
        raise ValueError(f"Duration column '{duration_col}' must contain numeric values")
    if not pd.api.types.is_numeric_dtype(data[event_col]):
        raise ValueError(f"Event column '{event_col}' must contain numeric values (0/1 or boolean)")
    
    # Validate event column values
    unique_events = data[event_col].dropna().unique()
    if not all(v in [0, 1, True, False] for v in unique_events):
        raise ValueError(f"Event column '{event_col}' must contain only 0/1 or boolean values")
    
    if group_col:
        data = data.dropna(subset=[group_col])
        groups = sorted(data[group_col].unique(), key=lambda v: str(v))
    else:
        groups = ['Overall']
        
    results = []
    
    for g in groups:
        if group_col:
            df_g = data[data[group_col] == g]
            label = f"{g}" # Short label for table
        else:
            df_g = data
            label = "Overall"
            
        n = len(df_g)
        events = df_g[event_col].sum()
        
        if n > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(df_g[duration_col], df_g[event_col], label=label)
            
            # Median
            median_val = kmf.median_survival_time_
            
            # CI for Median
            try:
                # median_survival_times returns a DF with columns like "KM_estimate_lower_0.95", "KM_estimate_upper_0.95"
                ci_df = median_survival_times(kmf.confidence_interval_)
                # Validate shape before accessing
                if ci_df.shape[0] > 0 and ci_df.shape[1] >= 2:
                    lower = ci_df.iloc[0, 0]
                    upper = ci_df.iloc[0, 1]
                else:
                    lower, upper = np.nan, np.nan
            except (KeyError, IndexError, AttributeError) as e:
                # Expected failures when CI cannot be computed
                _logger.debug(f"Could not compute CI for group {label}: {e}")
                lower, upper = np.nan, np.nan
            
            # Formatting Helper
            def fmt(v):
                if pd.isna(v) or np.isinf(v): return "NR" # Not Reached
                return f"{v:.1f}"
            
            med_str = fmt(median_val)
            low_str = fmt(lower)
            up_str = fmt(upper)
            
            # Combine into "Median (Lower-Upper)"
            display_str = f"{med_str} ({low_str}-{up_str})"
            if med_str == "NR" and low_str == "NR" and up_str == "NR":
                 display_str = "Not Reached"
        else:
            display_str = "-"

        results.append({
            "Group": label,
            "N": n,
            "Events": events,
            "Median Time (95% CI)": display_str
        })
        
    return pd.DataFrame(results)

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
    Fit Nelson–Aalen cumulative hazard curves optionally stratified by a grouping column.
    
    For each group (or overall), fits a Nelson–Aalen estimator, adds a cumulative hazard trace to a Plotly figure, and adds a shaded 95% confidence interval when available. Also collects per-group counts and event totals.
    
    Parameters:
        df (pandas.DataFrame): Input data containing duration, event, and optional group columns.
        duration_col (str): Name of the column containing follow-up times.
        event_col (str): Name of the column containing event indicators (typically 0/1).
        group_col (str or None): Name of the column to stratify by; if falsy, no stratification is applied.
    
    Returns:
        tuple: (fig, stats_df)
            - fig (plotly.graph_objects.Figure): Plotly figure containing cumulative hazard curves and optional shaded 95% CIs.
            - stats_df (pandas.DataFrame): DataFrame with columns ['Group', 'N', 'Events'] summarizing each plotted group.
    
    Raises:
        ValueError: If the input contains no valid rows after dropping missing durations/events or if a specified group column is missing.
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
    Fit a Cox proportional hazards model with preprocessing, validation, and a progressive penalization strategy.
    
    Performs column presence checks, drops rows with missing values, encodes categorical covariates (one-hot with drop_first), validates numeric covariates for infinities, extreme values, zero variance, perfect separation and high multicollinearity, standardizes numeric covariates (preserving binary 0/1), then attempts to fit CoxPH using increasing L2 penalization until a fit succeeds or all methods fail.
    
    Parameters:
        df (pandas.DataFrame): Input dataset containing the duration, event, and covariate columns.
        duration_col (str): Name of the duration / time-to-event column.
        event_col (str): Name of the event indicator column (expected 0/1).
        covariate_cols (list[str]): List of covariate column names to include in the model.
    
    Returns:
        cph (lifelines.CoxPHFitter or None): Fitted CoxPHFitter instance when fitting succeeds, otherwise None.
        res_df (pandas.DataFrame or None): Results table with columns `HR`, `95% CI Lower`, `95% CI Upper`, `P-value`, and `Method` when fitting succeeds; None on failure.
        data (pandas.DataFrame): Processed DataFrame used for fitting (after selecting columns, dropping missing values, encoding, and standardization); returned even on failure to aid debugging.
        error_message (str or None): Detailed error message when validation or fitting fails; None on success.
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
    """
    Generate a proportional-hazards test report and a list of Plotly figures showing scaled Schoenfeld residuals for each covariate.
    
    Parameters:
        cph: A fitted lifelines CoxPHFitter instance whose `duration_col` attribute indicates the duration column in `data`.
        data: pandas.DataFrame used to fit `cph`; must contain the duration and event columns used for the model.
    
    Returns:
        tuple:
            - report (str): A human-readable text summary of the proportional hazards test results. On failure this will contain an error message prefixed with "Assumption check failed:".
            - figures (list[plotly.graph_objects.Figure]): One Plotly Figure per covariate showing scaled Schoenfeld residuals versus event times. Residuals and times are aligned to event rows only; an empty list is returned on failure.
    """
    try:
        # 1. Statistical Test
        results = proportional_hazard_test(cph, data, time_transform='rank')
        text_report = "Proportional Hazards Test Results:\n" + results.summary.to_string()
        
        # 2. Schoenfeld Residual Plots (Plotly Version)
        figs_list = []
        
        # Compute residuals
        scaled_schoenfeld = cph.compute_residuals(data, 'scaled_schoenfeld')
        
        # 🟢 FIX: Align 'times' with the residuals (residuals only exist for events)
        # scaled_schoenfeld index corresponds to the original data index for event rows
        times = data.loc[scaled_schoenfeld.index, cph.duration_col]
        
        for col in scaled_schoenfeld.columns:
            fig = go.Figure()
            
            # Scatter plot of residuals
            fig.add_trace(go.Scatter(
                x=times, 
                y=scaled_schoenfeld[col],
                mode='markers',
                name='Residuals',
                marker={'color': COLORS['primary'], 'opacity': 0.6, 'size': 6}
            ))
            
            # Trend line (Linear) - Replicating original logic
            try:
                # Fit linear trend
                z = np.polyfit(times, scaled_schoenfeld[col], 1)
                p = np.poly1d(z)
                
                # Generate line points
                sorted_times = np.sort(times)
                trend_y = p(sorted_times)
                
                fig.add_trace(go.Scatter(
                    x=sorted_times,
                    y=trend_y,
                    mode='lines',
                    name='Trend (Linear)',
                    line={'color': COLORS['danger'], 'dash': 'dash', 'width': 2}
                ))
            except Exception as e:
                warnings.warn(f"Could not fit trend line for {col}: {e}", stacklevel=2)
            
            # Zero Line
            fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, line_width=1)
            
            # Update Layout
            fig.update_layout(
                title=f"Schoenfeld Residuals: {col}",
                xaxis_title="Time",
                yaxis_title="Scaled Residuals",
                template='plotly_white',
                height=450,
                showlegend=True,
                hovermode="closest"
            )
            
            figs_list.append(fig)

        return text_report, figs_list

    except Exception as e:
        return f"Assumption check failed: {e}", []


# --- 🟢 UPDATED: Create interactive Plotly forest plot for Cox Regression (Using shared lib) ---
def create_forest_plot_cox(res_df):
    """
    Create a publication-quality interactive forest plot of hazard ratios from a Cox regression.
    
    Parameters:
        res_df (pandas.DataFrame): Results DataFrame indexed by covariate label and containing columns
            'HR', '95% CI Lower', '95% CI Upper', and 'P-value'.
    
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure containing the forest plot and accompanying table.
    
    Raises:
        ValueError: If `res_df` is None or empty.
    """
    if res_df is None or res_df.empty:
        raise ValueError("No Cox regression results available for forest plot.")
    
    # Prepare data for the shared library
    df_plot = res_df.copy()
    df_plot['variable'] = df_plot.index
    
    # Call shared library to generate publication-quality plot (Table + Graph)
    fig = create_forest_plot(
        data=df_plot,
        estimate_col='HR',
        ci_low_col='95% CI Lower',
        ci_high_col='95% CI Upper',
        pval_col='P-value',
        label_col='variable',
        title="<b>Multivariable Cox Regression: Forest Plot (HR & 95% CI)</b>",
        x_label="Hazard Ratio (HR)",
        ref_line=1.0
    )
    
    return fig


# --- 🟢 UPDATED: Generate Forest Plot HTML for Cox Regression (HTML Report) ---
def generate_forest_plot_cox_html(res_df):
    """
    Generate an HTML snippet that embeds an interactive Cox regression forest plot and an interpretation guide.
    
    Parameters:
        res_df (pandas.DataFrame): Results DataFrame with columns 'HR', '95% CI Lower', '95% CI Upper', and 'P-value'.
    
    Returns:
        str: HTML string containing the Plotly-embedded forest plot (Plotly JS included for offline use) followed by a concise interpretation guide; if `res_df` is None or empty, returns a short HTML message indicating no results are available.
    """
    if res_df is None or res_df.empty:
        return "<p>No Cox regression results available for forest plot.</p>"
    
    # Create interactive forest plot (same function as web UI)
    fig = create_forest_plot_cox(res_df)
    
    # 🟢 OFFLINE SUPPORT: Embed Plotly JS directly in the plot
    # The figure now contains the Data Table inside it (via subplots), so no need for separate HTML table.
    plot_html = fig.to_html(include_plotlyjs=True, div_id='cox_forest_plot')
    
    # Interpretation guide (matching logistic regression style)
    interp_html = f"""
    <div style='margin-top:20px; padding:15px; background:#f8f9fa; border-left:4px solid {COLORS.get('primary', '#218084')}; border-radius:4px;'>
        <h4 style='color:{COLORS.get('primary_dark', '#1f8085')}; margin-top:0;'>💡 Interpretation Guide</h4>
        <ul style='margin:10px 0; padding-left:20px;'>
            <li><b>HR > 1:</b> Increased hazard (Risk Factor) 🔴</li>
            <li><b>HR < 1:</b> Decreased hazard (Protective Factor) 🟢</li>
            <li><b>HR = 1:</b> No effect (null)</li>
            <li><b>CI crosses 1.0:</b> Not statistically significant ⚠️</li>
            <li><b>CI doesn't cross 1.0:</b> Statistically significant ✅</li>
            <li><b>P < 0.05:</b> Statistically significant ✅</li>
        </ul>
    </div>
    """
    
    return f"<div style='margin:20px 0;'>{plot_html}{interp_html}</div>"

# --- 4. Landmark Analysis (KM) 🟢 FIX LM CI ---
def fit_km_landmark(df, duration_col, event_col, group_col, landmark_time):
    """
    Perform a landmark-time Kaplan–Meier analysis and plot group-specific survival curves re-based to time since the landmark.
    
    Only rows with observed time >= landmark_time are used; their times are re-baselined to time since the landmark and survival curves (with 95% CI shading when available) are produced per group. The function also runs an appropriate log-rank test on the landmarked data and returns a one-row summary of the test or an error note.
    
    Parameters:
        df (pandas.DataFrame): Source dataset containing the required columns.
        duration_col (str): Name of the column with observed times-to-event.
        event_col (str): Name of the column with the event indicator (1 = event, 0 = censored).
        group_col (str): Name of the column used to stratify groups for curves and comparisons.
        landmark_time (numeric): Landmark time threshold; rows with duration < landmark_time are excluded and remaining durations are replaced by (duration - landmark_time).
    
    Returns:
        fig (plotly.graph_objects.Figure or None): Plotly figure with Kaplan–Meier curves and shaded 95% confidence intervals for each group, or None if execution failed before plotting.
        stats_df (pandas.DataFrame or None): Single-row DataFrame summarizing the performed log-rank test or containing an error/note; None if not applicable.
        n_pre_filter (int): Number of records after dropping rows with missing duration, event, or group (before applying the landmark filter).
        n_post_filter (int): Number of records remaining after applying the landmark filter (duration >= landmark_time).
        error (str or None): Error message describing early failures (e.g., missing columns or insufficient records), otherwise None.
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

# --- 5. Report Generation 🟢 OFFLINE SUPPORT: Embed Plotly JS in Report ---
def generate_report_survival(title, elements):
    """
    Assemble a complete HTML report from a sequence of content elements, embedding tables, figures, and images for offline-friendly consumption.
    Uses unified teal color palette from _common.py.
    
    Builds an HTML document with the given title and iterates over `elements` to render supported content types. For Plotly figures, all JS is embedded in each plot for offline viewing. Supported element types and expected `data` values:
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
        html_doc (str): A self-contained, offline-friendly HTML string with all resources embedded.
    """
    
    primary_color = COLORS['primary']
    primary_dark = COLORS['primary_dark']
    text_color = COLORS['text']
    danger = COLORS['danger']
    
    css_style = f"""<style>
        body{{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            margin: 20px;
            background-color: #f4f6f8;
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
        h4{{
            color: {primary_dark};
            margin: 10px 0 8px 0;
        }}
        table{{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            background-color: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th, td{{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th{{
            background-color: {primary_dark};
            color: white;
            font-weight: 600;
        }}
        tr:hover{{
            background-color: #f8f9fa;
        }}
        tr:nth-child(even){{
            background-color: #fcfcfc;
        }}
        .sig-p {{
            color: {danger};
            font-weight: bold;
            background-color: #ffebee;
            padding: 2px 4px;
            border-radius: 4px;
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
    # 🟢 OFFLINE SUPPORT: No CDN, each Plotly figure will embed its own JS
    html_doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>{css_style}</head><body><h1>{safe_title}</h1>"""
    
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
                # 🟢 OFFLINE: include_plotlyjs=True embeds JS in EACH plot
                html_doc += d.to_html(full_html=False, include_plotlyjs=True)
            elif hasattr(d, 'savefig'):
                buf = io.BytesIO()
                d.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
        elif t == 'image':
            b64 = base64.b64encode(d).decode('utf-8')
            html_doc += f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
        elif t == 'html':
            # 🟢 OFFLINE: Raw HTML (forest plot) already has JS embedded via include_plotlyjs=True
            html_doc += str(d)
    
    html_doc += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM n Donate</a> | All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div>"""
    html_doc += "</body>\n</html>"
    
    return html_doc
