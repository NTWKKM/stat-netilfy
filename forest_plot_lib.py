"""
Forest Plot Visualization Module

For displaying effect sizes (OR, HR, RR) with confidence intervals across multiple variables.
Supports logistic regression, survival analysis, and epidemiological studies.
Optimized for Multivariable Analysis (standard Regression output) + Subgroup Analysis.

Author: NTWKKM (Updated by Gemini)
License: MIT
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from logger import get_logger
from tabs._common import get_color_palette
from statsmodels.formula.api import logit
import warnings
# Suppress specific convergence warnings only
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*")

logger = get_logger(__name__)
COLORS = get_color_palette()


class ForestPlot:
    """
    Interactive forest plot generator for statistical results.
    Optimized for Multivariable Analysis (Logistic/Cox Regression).
    
    Attributes:
        data (pd.DataFrame): DataFrame containing estimate and CI columns
        estimate_col (str): Column name for point estimates
        ci_low_col (str): Column name for CI lower bounds
        ci_high_col (str): Column name for CI upper bounds
        label_col (str): Column name for variable labels
        pval_col (str): Column name for P-values (Optional)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        estimate_col: str,
        ci_low_col: str,
        ci_high_col: str,
        label_col: str,
        pval_col: str = None,
    ):
        """
        Initialize a ForestPlot with validated, plotting-ready data and column mappings.
        
        Validates that the DataFrame is non-empty and contains the required columns, coerces the estimate and confidence-interval columns to numeric (dropping rows that become NaN), reverses row order for top-down display, and stores the provided column names on the instance.
        
        Parameters:
            data (pd.DataFrame): Source table containing effect estimates and confidence intervals.
            estimate_col (str): Column name for the point estimate (e.g., OR, HR, RR).
            ci_low_col (str): Column name for the lower 95% confidence interval bound.
            ci_high_col (str): Column name for the upper 95% confidence interval bound.
            label_col (str): Column name used for row labels displayed on the plot.
            pval_col (str, optional): Column name containing p-values; omit or pass None if not available.
        
        Raises:
            ValueError: If `data` is empty, if any required columns are missing, or if no valid rows remain after numeric coercion.
        """
        # Validation
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_cols = {estimate_col, ci_low_col, ci_high_col, label_col}
        if pval_col:
            required_cols.add(pval_col)

        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create a copy to avoid SettingWithCopyWarning
        self.data = data.copy()

        # --- FIX: Force numeric conversion to prevent str vs float errors ---
        # Coerce errors='coerce' turns non-numeric strings into NaN
        numeric_cols = [estimate_col, ci_low_col, ci_high_col]
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Drop rows where essential plotting data is missing (NaN)
        # Note: This checks for NaN but allows Infinite values (Perfect Separation needs specific handling later)
        self.data = self.data.dropna(subset=numeric_cols)
        
        if self.data.empty:
            raise ValueError("No valid data after removing NaN values")
        
        # Reverse for top-down display
        self.data = self.data.iloc[::-1].reset_index(drop=True)
        
        self.estimate_col = estimate_col
        self.ci_low_col = ci_low_col
        self.ci_high_col = ci_high_col
        self.label_col = label_col
        self.pval_col = pval_col
        
        # Log range safely
        try:
            est_min = self.data[estimate_col].min()
            est_max = self.data[estimate_col].max()
            logger.info(
                f"ForestPlot initialized: {len(self.data)} variables, "
                f"estimate range [{est_min:.3f}, {est_max:.3f}]"
            )
        except Exception as e:
            logger.warning(f"Could not log estimate range: {e}")
    
    def _add_significance_stars(self, p):
        """
        Convert a p-value into conventional significance star notation.
        
        Parameters:
            p (float | str | None): P-value to evaluate; accepts numeric values, strings with inequality signs (e.g., "<0.001" or ">0.05"), or None/NaN.
        
        Returns:
            str: `"***"` if p < 0.001, `"**"` if p < 0.01, `"*"` if p < 0.05, otherwise an empty string. Returns an empty string for NaN or non-convertible inputs.
        """
        # --- FIX: Handle string p-values (e.g., "<0.001") robustly ---
        try:
            if pd.isna(p):
                return ""
            
            p_val = p
            if isinstance(p, str):
                # Remove common characters and whitespace
                clean_p = p.replace('<', '').replace('>', '').strip()
                p_val = float(clean_p)
            
            if p_val < 0.001:
                return "***"
            if p_val < 0.01:
                return "**"
            if p_val < 0.05:
                return "*"
        except (ValueError, TypeError):
            # If conversion fails, return empty string
            return ""
            
        return ""
    
    def _get_ci_width_colors(self, base_color: str) -> list:
        """
        Generate per-row RGBA marker colors with opacity scaled inversely to each row's confidence-interval width.
        
        Parameters:
            base_color (str): Hex color string to use as the RGB base for markers (e.g. "#21808D"). If parsing fails or a non-6-digit hex is provided, a default teal color is used.
        
        Returns:
            marker_colors (list): List of `rgba(r, g, b, a)` color strings for each row; rows with narrower CIs have higher opacity, rows with wider CIs have lower opacity.
            ci_normalized (numpy.ndarray): Array of CI widths normalized to the range [0, 1], aligned with `marker_colors` (0 = narrowest CI, 1 = widest CI).
        """
        # Ensure values are float for calculation
        ci_high = self.data[self.ci_high_col]
        ci_low = self.data[self.ci_low_col]
        
        ci_width = ci_high - ci_low
        
        # --- FIX: Handle Infinite Widths (Perfect Separation) ---
        # If CI is infinite, width is infinite. This breaks normalization.
        # Strategy: Normalize only finite widths. Set infinite widths to 1.0 (max width -> min opacity).
        
        is_finite = np.isfinite(ci_width)
        finite_widths = ci_width[is_finite]
        
        # Initialize ci_normalized with NaNs
        ci_normalized = pd.Series(np.nan, index=ci_width.index)
        
        if not finite_widths.empty:
            ci_min, ci_max = finite_widths.min(), finite_widths.max()
            
            # Avoid division by zero
            if ci_max > ci_min:
                ci_normalized[is_finite] = (finite_widths - ci_min) / (ci_max - ci_min)
            else:
                ci_normalized[is_finite] = 0.5
        
        # Handle infinite/NaN widths: treat as maximum width (1.0) => 0.5 opacity
        ci_normalized = ci_normalized.fillna(1.0)
        
        # Parse base color (hex to RGB)
        hex_color = base_color.lstrip('#')
        try:
            if len(hex_color) == 6:
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            else:
                rgb = (33, 128, 141)  # Default teal
        except ValueError:
            rgb = (33, 128, 141)  # Default teal
        
        # Generate colors with varying opacity
        # CI narrow (0) = full opacity (1.0)
        # CI wide (1) = partial opacity (0.5)
        # ci_normalized can be float or Series here, ensure float
        marker_colors = [
            f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {1.0 - 0.5*float(norm_val):.2f})"
            for norm_val in ci_normalized
        ]
        
        return marker_colors, ci_normalized.values
    
    def get_summary_stats(self, ref_line: float = 1.0):
        """
        Compute dataset-level summary metrics for the plotted estimates relative to a reference line.
        
        Parameters:
            ref_line (float): Reference value used to assess whether confidence intervals cross the reference (e.g., 1.0 for ratios).
        
        Returns:
            dict: A mapping with the following keys:
                - 'n_variables': number of rows used in the plot.
                - 'median_est': median of the estimate column.
                - 'min_est': minimum of the estimate column.
                - 'max_est': maximum of the estimate column.
                - 'n_significant': count of rows with p-value < 0.05, or `None` if no p-value column is available.
                - 'pct_significant': percentage (0–100) of rows with p-value < 0.05, or `None` if no p-value column is available.
                - 'n_ci_significant': count of rows whose confidence interval does not cross `ref_line`.
        """
        n_sig = 0
        pct_sig = 0

        # --- FIX: Count significant (p < 0.05) safely ---
        if self.pval_col and self.pval_col in self.data.columns:
            # Convert to numeric temporarily for counting, coercing errors
            p_numeric = pd.to_numeric(
                self.data[self.pval_col].astype(str).str.replace('<', '').str.replace('>', ''), 
                errors='coerce'
            )
            n_sig = (p_numeric < 0.05).sum()
            pct_sig = 100 * n_sig / len(self.data) if len(self.data) > 0 else 0
        else:
            n_sig = pct_sig = None
        
        # Count CI doesn't cross ref_line (graphical significance)
        # Data is already forced to numeric in __init__
        ci_low = self.data[self.ci_low_col]
        ci_high = self.data[self.ci_high_col]

        ci_sig = (
            ((ci_low > ref_line) | (ci_high < ref_line))
            if ref_line > 0
            else ((ci_low * ci_high) > 0)
        )
        n_ci_sig = ci_sig.sum()
        
        return {
            'n_variables': len(self.data),
            'median_est': self.data[self.estimate_col].median(),
            'min_est': self.data[self.estimate_col].min(),
            'max_est': self.data[self.estimate_col].max(),
            'n_significant': n_sig,
            'pct_significant': pct_sig,
            'n_ci_significant': n_ci_sig,  # CI doesn't cross ref_line
        }
    
    def create(
        self,
        title: str = "Forest Plot",
        x_label: str = "Effect Size (95% CI)",
        ref_line: float = 1.0,
        show_ref_line: bool = True,
        show_sig_stars: bool = True,
        show_ci_width_colors: bool = True,
        show_sig_divider: bool = True,
        height: int = None,
        color: str = None,
    ) -> go.Figure:
        """
        Builds an interactive, publication-quality forest plot of effect estimates with 95% CIs.
        
        Creates a multi-column Plotly figure showing variable labels, formatted estimates (estimate with 95% CI), optional p-values, and a forest panel with point estimates and asymmetric error bars. The plot can display a reference (no-effect) line, annotate significance with stars, color markers by CI width, and optionally draw a divider between CI-significant and non-significant rows. The x-axis uses a log scale automatically when effect estimates are strictly positive and widely dispersed.
        
        Parameters:
            title (str): Main plot title.
            x_label (str): Label for the forest plot x-axis.
            ref_line (float): Reference (no-effect) line value drawn vertically on the forest panel.
            show_ref_line (bool): Whether to display the reference line.
            show_sig_stars (bool): Whether to append significance stars to variable labels when a p-value column is present.
            show_ci_width_colors (bool): Whether to color markers with opacity scaled by CI width (narrow CI → more opaque).
            show_sig_divider (bool): Whether to draw a horizontal divider between CI-significant and non-significant rows when both are present.
            height (int | None): Figure height in pixels; if None, height is computed from row count.
            color (str | None): Base hex/RGB color used for markers; a default palette color is used when None.
        
        Returns:
            go.Figure: A Plotly Figure containing the assembled forest plot.
        """
        if color is None:
            color = COLORS['primary']
        
        # --- Pre-processing Data for Display ---
        # 1. Format Estimate (95% CI)
        # FIX: Handle infinite values in display string
        def fmt_est_ci(x) -> str:
            try:
                est = x[self.estimate_col]
                low = x[self.ci_low_col]
                high = x[self.ci_high_col]
                
                est_str = f"{est:.2f}" if np.isfinite(est) else "Inf"
                low_str = f"{low:.2f}" if np.isfinite(low) else "Inf"
                high_str = f"{high:.2f}" if np.isfinite(high) else "Inf"
                return f"{est_str} ({low_str}-{high_str})"
            except (KeyError, TypeError, ValueError):
                return "Error"
        self.data['__display_est'] = self.data.apply(fmt_est_ci, axis=1)

        # 2. Format P-value (if available)
        if self.pval_col:
            def fmt_p(p):
                """
                Format a p-value for human-readable display following common reporting conventions.
                
                Parameters:
                    p: A numeric p-value or string representation (may include leading '<' or '>').
                
                Returns:
                    A string: "<0.001" if the parsed numeric value is less than 0.001, a three-decimal string (e.g. "0.123") for other numeric values, or the original input converted to a string if the value cannot be parsed as a number.
                """
                try:
                    # Clean string first if necessary
                    p_str = str(p).replace('<', '').replace('>', '').strip()
                    p_float = float(p_str)
                    if p_float < 0.001: return "<0.001"
                    return f"{p_float:.3f}"
                except (ValueError, TypeError): 
                    return str(p) # Return original if conversion fails
            
            self.data['__display_p'] = self.data[self.pval_col].apply(fmt_p)
            
            # Helper for color logic
            def get_p_color(p):
                """
                Determine a display color name based on a p-value threshold of 0.05.
                
                Parameters:
                    p (float | str): The p-value to evaluate. Strings may include a leading '<' (e.g., '<0.001').
                
                Returns:
                    color (str): `"red"` if the numeric p-value is less than 0.05, `"black"` otherwise. Non-convertible inputs default to `"black"`.
                """
                try:
                    if isinstance(p, str) and '<' in p: 
                        # Assume <0.05 is red
                        val = float(p.replace('<','').strip())
                        return "red" if val < 0.05 else "black"
                    
                    val = float(p)
                    return "red" if val < 0.05 else "black"
                except (ValueError, TypeError):
                    return "black"

            p_text_colors = self.data[self.pval_col].apply(get_p_color).tolist()
        else:
            self.data['__display_p'] = ""
            p_text_colors = ["black"] * len(self.data)
        
        # 3. Add significance stars to labels
        if show_sig_stars and self.pval_col:
            self.data['__sig_stars'] = self.data[self.pval_col].apply(self._add_significance_stars)
            self.data['__display_label'] = (
                self.data[self.label_col].astype(str) + " " + self.data['__sig_stars']
            ).str.rstrip()
        else:
            self.data['__display_label'] = self.data[self.label_col]
        
        # 4. 🎨 Get CI width colors
        marker_colors, _ = self._get_ci_width_colors(color) if show_ci_width_colors else ([color] * len(self.data), None)

        # --- Dynamic Column Layout ---
        # Check if pval column is present and not all NaN
        has_pval = self.pval_col is not None and not self.data[self.pval_col].isna().all()
        column_widths = [0.25, 0.20, 0.10, 0.45] if has_pval else [0.25, 0.20, 0.55]
        num_cols = 4 if has_pval else 3
        plot_col = 4 if has_pval else 3
        
        fig = make_subplots(
            rows=1, cols=num_cols,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            column_widths=column_widths,
            specs=[[{"type": "scatter"} for _ in range(num_cols)]]
        )

        y_pos = list(range(len(self.data)))
        
        # Column 1: Variable Labels
        fig.add_trace(go.Scatter(x=[0]*len(y_pos), y=y_pos, text=self.data['__display_label'], mode="text", textposition="middle right", textfont=dict(size=13, color="black"), hoverinfo="none", showlegend=False), row=1, col=1)

        # Column 2: Estimate (95% CI)
        fig.add_trace(go.Scatter(x=[0]*len(y_pos), y=y_pos, text=self.data['__display_est'], mode="text", textposition="middle center", textfont=dict(size=13, color="black"), hoverinfo="none", showlegend=False), row=1, col=2)

        # Column 3: P-value
        if has_pval:
            fig.add_trace(go.Scatter(x=[0]*len(y_pos), y=y_pos, text=self.data['__display_p'], mode="text", textposition="middle center", textfont=dict(size=13, color=p_text_colors), hoverinfo="none", showlegend=False), row=1, col=3)

        # Column N: Forest Plot
        # FIX: Check for infinity in min/max to avoid log scale errors
        finite_data = self.data[np.isfinite(self.data[self.estimate_col])]
        if not finite_data.empty:
            est_min = finite_data[self.estimate_col].min()
            est_max = finite_data[self.estimate_col].max()
            # Use log scale if values are positive and spread is large
            use_log_scale = (est_min > 0) and ((est_max / est_min) > 5)
        else:
            use_log_scale = False

        if show_ref_line:
            fig.add_vline(x=ref_line, line_dash='dash', line_color='rgba(192, 21, 47, 0.6)', line_width=2, annotation_text=f'No Effect ({ref_line})', annotation_position='top', row=1, col=plot_col)
        
        # ✂️ Add horizontal divider between Significant and Non-significant
        if show_sig_divider:
            # Use numeric comparison (already coerced in __init__)
            ci_low = self.data[self.ci_low_col]
            ci_high = self.data[self.ci_high_col]
            
            ci_sig = ((ci_low > ref_line) | (ci_high < ref_line)) if ref_line > 0 else ((ci_low * ci_high) > 0)
            
            # Simple divider logic: if sorted by significance, find the flip point
            if ci_sig.any() and (~ci_sig).any():
                # Find index where sign changes (approximation for visualization)
                divider_y = ci_sig.idxmin() - 0.5
                fig.add_hline(y=divider_y, line_dash='dot', line_color='rgba(100, 100, 100, 0.3)', line_width=1.5, row=1, col=plot_col)

        hover_parts = ["<b>%{text}</b><br>", f"<b>{self.estimate_col}:</b> %{{x:.3f}}<br>", "<b>95% CI:</b> %{customdata[0]:.3f} - %{customdata[1]:.3f}<br>"]
        if has_pval: hover_parts.append("<b>P-value:</b> %{customdata[2]}<br>")
        if show_ci_width_colors: hover_parts.append("<b>CI Width:</b> %{customdata[3]:.3f}<br>")
        hover_parts.append("<extra></extra>")
        hovertemplate = "".join(hover_parts)
        
        customdata = np.stack((
            self.data[self.ci_low_col], 
            self.data[self.ci_high_col], 
            self.data['__display_p'] if has_pval else [None]*len(self.data), # Use formatted P for hover
            self.data[self.ci_high_col] - self.data[self.ci_low_col]
        ), axis=-1)

        fig.add_trace(go.Scatter(
            x=self.data[self.estimate_col], y=y_pos,
            error_x=dict(type='data', symmetric=False, array=self.data[self.ci_high_col] - self.data[self.estimate_col], arrayminus=self.data[self.estimate_col] - self.data[self.ci_low_col], color='rgba(100,100,100,0.5)', thickness=2, width=4),
            mode='markers', marker=dict(size=10, color=marker_colors, symbol='square', line=dict(width=1.5, color='white')), # Square markers for Regression
            text=self.data['__display_label'], customdata=customdata, hovertemplate=hovertemplate, showlegend=False
        ), row=1, col=plot_col)

        # --- Update Layout ---
        if height is None: height = max(400, len(self.data) * 35 + 120)
        
        summary = self.get_summary_stats(ref_line)
        summary_text = f"N={summary['n_variables']}, Median={summary['median_est']:.2f}"
        if summary['pct_significant'] is not None: summary_text += f", Sig={summary['pct_significant']:.0f}%"
        
        title_with_summary = f"<b>{title}</b><br><span style='font-size: 13px; color: rgba(100,100,100,0.9);'>{summary_text}</span>"

        fig.update_layout(
            title=dict(text=title_with_summary, x=0.01, xanchor='left', font=dict(size=18)),
            height=height, showlegend=False, template='plotly_white',
            margin=dict(l=10, r=20, t=120, b=40),
            plot_bgcolor='white', autosize=True
        )

        for c in range(1, plot_col):
            fig.update_xaxes(visible=False, showgrid=False, zeroline=False, row=1, col=c)
            fig.update_yaxes(visible=False, showgrid=False, zeroline=False, row=1, col=c)

        fig.update_yaxes(visible=False, range=[-0.5, len(self.data)-0.5], row=1, col=plot_col)
        fig.update_xaxes(title_text=x_label, type='log' if use_log_scale else 'linear', row=1, col=plot_col, gridcolor='rgba(200, 200, 200, 0.2)')

        headers = ["Variable", "Estimate (95% CI)"] + (["P-value", f"{x_label} Plot"] if has_pval else [f"{x_label} Plot"])
        
        for i, h in enumerate(headers, 1):
            xref_val = f"x{i} domain" if i > 1 else "x domain"
            fig.add_annotation(x=0.5 if i != 1 else 1.0, y=1.0, xref=xref_val, yref="paper", text=f"<b>{h}</b>", showarrow=False, yanchor="bottom", font=dict(size=14, color="black"))

        logger.info(f"Forest plot generated: {title}, {len(self.data)} variables (Multivariable Analysis)")
        return fig


def create_forest_plot(
    data: pd.DataFrame, estimate_col: str, ci_low_col: str, ci_high_col: str, label_col: str,
    pval_col: str = None, title: str = "Forest Plot", x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0, height: int = None, **kwargs
) -> go.Figure:
    """
    Create a forest plot from a DataFrame using a single convenience call.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing point estimates and confidence intervals.
        estimate_col (str): Column name for point estimates.
        ci_low_col (str): Column name for lower confidence interval bounds.
        ci_high_col (str): Column name for upper confidence interval bounds.
        label_col (str): Column name for row labels displayed on the plot.
        pval_col (str, optional): Column name for p-values used for significance annotations.
        title (str, optional): Plot title.
        x_label (str, optional): X-axis label.
        ref_line (float, optional): Reference line value shown on the plot (e.g., 1.0 for ratios).
        height (int, optional): Plot height in pixels.
        **kwargs: Additional keyword arguments forwarded to ForestPlot.create.
    
    Returns:
        go.Figure: Plotly Figure with the constructed forest plot, or an empty Figure if plotting failed due to invalid input.
    """
    try:
        fp = ForestPlot(data, estimate_col, ci_low_col, ci_high_col, label_col, pval_col)
        return fp.create(title=title, x_label=x_label, ref_line=ref_line, height=height, **kwargs)
    except ValueError as e:
        logger.error(f"Forest plot creation failed: {e}")
        st.error(f"Could not create forest plot: {e}")
        return go.Figure()


def create_forest_plot_from_logit(aor_dict: dict, title: str = "Adjusted Odds Ratios") -> go.Figure:
    """
    Builds a forest plot from a dictionary of logistic regression adjusted odds ratios.
    
    Parameters:
        aor_dict (dict): Mapping from variable name to a result dict. Each result dict should contain
            'ci_low' and 'ci_high' and either 'aor' or 'or' for the point estimate. Optionally include
            'p_value' or 'p' to display p-values; entries missing estimate or CI are skipped.
        title (str): Plot title.
    
    Returns:
        go.Figure: A Plotly Figure containing the forest plot of odds ratios with 95% CIs.
    """
    data = []
    
    for var, result in aor_dict.items():
        estimate = result.get('aor', result.get('or'))
        ci_low = result.get('ci_low')
        ci_high = result.get('ci_high')
        p_val = result.get('p_value', result.get('p'))
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        row = {
            'variable': var,
            'aor': float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
        if p_val is not None:
            # Keep p_val as is, let ForestPlot handle parsing
            row['p_value'] = p_val
            
        data.append(row)
    
    if not data:
        logger.error("No valid aOR values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    p_col = 'p_value' if 'p_value' in df.columns else None
    
    return create_forest_plot(
        df,
        estimate_col='aor',
        ci_low_col='ci_low',
        ci_high_col='ci_high',
        label_col='variable',
        pval_col=p_col,
        title=title,
        x_label='Odds Ratio (95% CI)',
        ref_line=1.0,
    )


def create_forest_plot_from_cox(hr_dict: dict, title: str = "Hazard Ratios (Cox Regression)") -> go.Figure:
    """
    Builds a forest plot from Cox regression hazard ratio results.
    
    Parameters:
        hr_dict (dict): Mapping of variable names to result dicts. Each result dict must include an estimate and confidence interval using one of the supported key sets:
            - estimate: 'hr' or 'HR'
            - lower CI: 'ci_low' or 'CI Lower'
            - upper CI: 'ci_high' or 'CI Upper'
            An optional p-value may be provided as 'p_value' or 'p'.
        title (str): Plot title.
    
    Returns:
        go.Figure: Plotly figure showing hazard ratios with 95% confidence intervals; an empty Figure if no valid hazard ratio rows are found.
    """
    data = []
    
    for var, result in hr_dict.items():
        estimate = result.get('hr', result.get('HR'))
        ci_low = result.get('ci_low', result.get('CI Lower'))
        ci_high = result.get('ci_high', result.get('CI Upper'))
        p_val = result.get('p_value', result.get('p'))

        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        row = {
            'variable': var,
            'hr': float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
        if p_val is not None:
            row['p_value'] = p_val

        data.append(row)
    
    if not data:
        logger.error("No valid HR values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    p_col = 'p_value' if 'p_value' in df.columns else None
    
    return create_forest_plot(
        df,
        estimate_col='hr',
        ci_low_col='ci_low',
        ci_high_col='ci_high',
        label_col='variable',
        pval_col=p_col,
        title=title,
        x_label='Hazard Ratio (95% CI)',
        ref_line=1.0,
    )


def create_forest_plot_from_rr(
    rr_or_dict: dict,
    title: str = "Risk/Odds Ratios",
    effect_type: str = 'RR'
) -> go.Figure:
    """
    Builds a forest plot of risk or odds ratios from a mapping of group labels to result dictionaries.
    
    Parameters:
        rr_or_dict (dict): Mapping from group label to a result dict. Each result dict must contain the effect estimate under the lowercase `effect_type` (e.g., `'rr'` or `'or'`) or the exact `effect_type` key, and confidence bounds under `'ci_low'`/`'ci_high'` or `'CI Lower'`/`'CI Upper'`. An optional p-value may be provided under `'p_value'` or `'p'`.
        title (str): Plot title.
        effect_type (str): Effect measure name used for extracting and labeling the estimate (commonly `'RR'` or `'OR'`); the lowercase form is used to read values from each result dict.
    
    Returns:
        go.Figure: Plotly Figure containing the forest plot; returns an empty Figure if no valid rows were found.
    """
    data = []
    metric_key = effect_type.lower()
    
    for group_name, result in rr_or_dict.items():
        estimate = result.get(metric_key, result.get(effect_type))
        ci_low = result.get('ci_low', result.get('CI Lower'))
        ci_high = result.get('ci_high', result.get('CI Upper'))
        p_val = result.get('p_value', result.get('p'))
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        row = {
            'variable': group_name,
            metric_key: float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
        if p_val is not None:
            row['p_value'] = p_val
            
        data.append(row)
    
    if not data:
        logger.error(f"No valid {effect_type} values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    p_col = 'p_value' if 'p_value' in df.columns else None

    return create_forest_plot(
        df,
        estimate_col=metric_key,
        ci_low_col='ci_low',
        ci_high_col='ci_high',
        label_col='variable',
        pval_col=p_col,
        title=title,
        x_label=f'{effect_type} (95% CI)',
        ref_line=1.0,
    )


# ============================================================================
# SUBGROUP ANALYSIS FUNCTIONS
# ============================================================================

def subgroup_analysis_logit(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    subgroup_col: str,
    adjustment_cols: list = None,
    title: str = "Subgroup Analysis (Logistic Regression)",
    x_label: str = "Odds Ratio (95% CI)",
    return_stats: bool = True,
) -> tuple:
    """
    Perform subgroup logistic regression analyses and produce a forest plot of odds ratios.
    
    Fits an overall logistic regression and subgroup-specific logistic models (with optional covariate adjustment), computes odds ratios with 95% CIs and p-values, performs an interaction test between treatment and subgroup, and builds an interactive Plotly forest plot summarizing results.
    
    Parameters:
        df (pd.DataFrame): Input dataset containing all required columns.
        outcome_col (str): Name of the binary outcome variable.
        treatment_col (str): Name of the treatment/exposure variable.
        subgroup_col (str): Name of the categorical variable to stratify by.
        adjustment_cols (list, optional): List of covariate names to include as adjustments.
        title (str): Plot title.
        x_label (str): X-axis label for the plot (e.g., "Odds Ratio (95% CI)").
        return_stats (bool): If True, return a statistics dictionary alongside the figure; otherwise return the result DataFrame.
    
    Returns:
        tuple:
            - go.Figure: Interactive Plotly forest plot showing overall and subgroup odds ratios with 95% CIs.
            - dict or pd.DataFrame: If `return_stats` is True, a dict containing overall OR, CI, p-value, subgroup entries, interaction p-value, heterogeneity flag, and the result DataFrame under 'result_df'; otherwise the result DataFrame used to create the plot.
    
    """
    try:
        if adjustment_cols is None:
            adjustment_cols = []
        
        # Validate inputs
        required_cols = {outcome_col, treatment_col, subgroup_col} | set(adjustment_cols)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        
        # Remove missing values
        df_clean = df[list(required_cols)].dropna()
        if len(df_clean) < 10:
            raise ValueError(f"Insufficient data after removing NaN: only {len(df_clean)} rows")
        
        # Build formula string
        formula_base = f'{outcome_col} ~ {treatment_col}'
        if adjustment_cols:
            formula_base += ' + ' + ' + '.join(adjustment_cols)
        
        results_list = []
        subgroup_models = {}
        
        # === OVERALL MODEL ===
        try:
            model_overall = logit(formula_base, data=df_clean).fit(disp=0)
            or_overall = np.exp(model_overall.params[treatment_col])
            ci_overall = np.exp(model_overall.conf_int().loc[treatment_col])
            p_overall = model_overall.pvalues[treatment_col]
            
            results_list.append({
                'variable': f'Overall (N={len(df_clean)})',
                'or': or_overall,
                'ci_low': ci_overall[0],
                'ci_high': ci_overall[1],
                'p_value': p_overall,
                'n': len(df_clean),
                'type': 'overall'
            })
            logger.info(f"Overall model: OR={or_overall:.3f}, P={p_overall:.4f}")
        except Exception as e:
            logger.error(f"Overall model fitting failed: {e}")
            st.error(f"Could not fit overall model: {e}")
            return go.Figure(), {}
        
        # === SUBGROUP MODELS ===
        subgroups = sorted(df_clean[subgroup_col].dropna().unique())
        if len(subgroups) < 2:
            raise ValueError(f"Subgroup variable '{subgroup_col}' has fewer than 2 unique values")
        
        for subgroup_val in subgroups:
            df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
            
            if len(df_sub) < 5:
                logger.warning(f"Subgroup '{subgroup_col}={subgroup_val}' too small (N={len(df_sub)}), skipping")
                continue
            
            try:
                model_sub = logit(formula_base, data=df_sub).fit(disp=0)
                or_sub = np.exp(model_sub.params[treatment_col])
                ci_sub = np.exp(model_sub.conf_int().loc[treatment_col])
                p_sub = model_sub.pvalues[treatment_col]
                
                results_list.append({
                    'variable': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                    'or': or_sub,
                    'ci_low': ci_sub[0],
                    'ci_high': ci_sub[1],
                    'p_value': p_sub,
                    'n': len(df_sub),
                    'subgroup_val': subgroup_val,
                    'type': 'subgroup'
                })
                subgroup_models[subgroup_val] = model_sub
                logger.info(f"Subgroup {subgroup_col}={subgroup_val}: OR={or_sub:.3f}, P={p_sub:.4f}")
            except Exception as e:
                logger.warning(f"Model fitting failed for subgroup {subgroup_val}: {e}")
        
        # === INTERACTION TEST ===
        try:
            formula_int = f'{outcome_col} ~ {treatment_col} * {subgroup_col}'
            if adjustment_cols:
                formula_int += ' + ' + ' + '.join(adjustment_cols)
            
            model_int = logit(formula_int, data=df_clean).fit(disp=0)
            interaction_term = f'{treatment_col}:{subgroup_col}'
            
            if interaction_term in model_int.pvalues.index:
                p_interaction = model_int.pvalues[interaction_term]
            else:
                # Alternative: Try with different naming
                interaction_cols = [col for col in model_int.pvalues.index if ':' in col]
                if interaction_cols:
                    p_interaction = model_int.pvalues[interaction_cols[0]]
                else:
                    p_interaction = np.nan
            
            logger.info(f"Interaction test: P={p_interaction:.4f}")
        except Exception as e:
            logger.warning(f"Interaction test failed: {e}")
            p_interaction = np.nan
        
        # === CREATE FOREST PLOT ===
        result_df = pd.DataFrame(results_list)
        
        # Add P-interaction to title
        if not np.isnan(p_interaction):
            het_text = "Heterogeneous" if p_interaction < 0.05 else "Homogeneous"
            title_final = f"{title}<br><span style='font-size: 12px; color: rgba(100,100,100,0.9);'>P for interaction = {p_interaction:.4f} ({het_text})</span>"
        else:
            title_final = title
        
        fig = create_forest_plot(
            data=result_df,
            estimate_col='or',
            ci_low_col='ci_low',
            ci_high_col='ci_high',
            label_col='variable',
            pval_col='p_value',
            title=title_final,
            x_label=x_label,
            ref_line=1.0,
            height=max(400, len(result_df) * 50 + 100)
        )
        
        if return_stats:
            stats_dict = {
                'overall_or': or_overall,
                'overall_ci': (ci_overall[0], ci_overall[1]),
                'overall_p': p_overall,
                'overall_n': len(df_clean),
                'subgroups': {sg['subgroup_val']: sg for sg in results_list if sg['type'] == 'subgroup'},
                'p_interaction': p_interaction,
                'heterogeneous': p_interaction < 0.05 if not np.isnan(p_interaction) else None,
                'result_df': result_df
            }
            return fig, stats_dict
        else:
            return fig, result_df
    
    except Exception as e:
        logger.error(f"Subgroup analysis failed: {e}")
        st.error(f"Subgroup analysis error: {e}")
        return go.Figure(), {}


def subgroup_analysis_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    treatment_col: str,
    subgroup_col: str,
    adjustment_cols: list = None,
    title: str = "Subgroup Analysis (Cox Regression)",
    x_label: str = "Hazard Ratio (95% CI)",
    return_stats: bool = True,
) -> tuple:
    """
    Perform subgroup analysis for Cox proportional hazards models and produce a forest plot of hazard ratios.
    
    Parameters:
        df (pd.DataFrame): Dataset containing time, event, treatment, subgroup, and covariates.
        time_col (str): Column name for follow-up duration.
        event_col (str): Column name for event indicator (0/1 or False/True).
        treatment_col (str): Column name for the treatment/exposure variable whose HR is estimated.
        subgroup_col (str): Column name for the categorical subgroup variable to stratify by.
        adjustment_cols (list, optional): List of covariate column names to adjust for in models.
        title (str, optional): Plot title.
        x_label (str, optional): X-axis label for the forest plot.
        return_stats (bool, optional): If True, return a statistics dictionary alongside the figure.
    
    Returns:
        tuple:
            If return_stats is True: (fig, stats_dict)
                - fig (plotly.graph_objects.Figure): Interactive forest plot of overall and subgroup HRs.
                - stats_dict (dict): Contains overall HR, overall CI and p-value, per-subgroup summaries, interaction p-value, heterogeneity flag, and the result DataFrame.
            If return_stats is False: (fig, result_df)
                - result_df (pd.DataFrame): DataFrame with rows for overall and each subgroup containing 'variable', 'hr', 'ci_low', 'ci_high', 'p_value', and related metadata.
    
    Notes:
        - Requires the lifelines package; if not available, the function returns an empty figure.
        - Subgroups with too few observations or events are skipped.
    """
    try:
        from lifelines import CoxPHFitter
        
        if adjustment_cols is None:
            adjustment_cols = []
        
        # Validate inputs
        required_cols = {time_col, event_col, treatment_col, subgroup_col} | set(adjustment_cols)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        
        # Remove missing values
        cols_for_clean = list(required_cols)
        df_clean = df[cols_for_clean].dropna()
        if len(df_clean) < 10:
            raise ValueError(f"Insufficient data after removing NaN: only {len(df_clean)} rows")
        
        cph = CoxPHFitter()
        results_list = []
        
        # === OVERALL MODEL ===
        try:
            covariates = [treatment_col] + adjustment_cols
            # Select only the columns needed for the model
            model_cols = [time_col, event_col] + covariates
            cph.fit(df_clean[model_cols], duration_col=time_col, event_col=event_col, show_progress=False)
            
            hr_overall = np.exp(cph.params_[treatment_col])
            ci_overall = np.exp(cph.confidence_intervals_.loc[treatment_col])
            p_overall = cph.summary.loc[treatment_col, 'p']
            
            results_list.append({
                'variable': f'Overall (N={len(df_clean)})',
                'hr': hr_overall,
                'ci_low': ci_overall[0],
                'ci_high': ci_overall[1],
                'p_value': p_overall,
                'n': len(df_clean),
                'type': 'overall'
            })
            logger.info(f"Overall Cox model: HR={hr_overall:.3f}, P={p_overall:.4f}")
        except Exception as e:
            logger.error(f"Overall Cox model fitting failed: {e}")
            st.error(f"Could not fit overall Cox model: {e}")
            return go.Figure(), {}
        
        # === SUBGROUP MODELS ===
        subgroups = sorted(df_clean[subgroup_col].dropna().unique())
        if len(subgroups) < 2:
            raise ValueError(f"Subgroup variable '{subgroup_col}' has fewer than 2 unique values")
        
        for subgroup_val in subgroups:
            df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
            
            if len(df_sub) < 5 or df_sub[event_col].sum() < 2:
                logger.warning(f"Subgroup '{subgroup_col}={subgroup_val}' too small or few events (N={len(df_sub)}, events={df_sub[event_col].sum()}), skipping")
                continue
            
            try:
                cph_sub = CoxPHFitter()
                cph_sub.fit(df_sub[model_cols], duration_col=time_col, event_col=event_col, show_progress=False)
                
                hr_sub = np.exp(cph_sub.params_[treatment_col])
                ci_sub = np.exp(cph_sub.confidence_intervals_.loc[treatment_col])
                p_sub = cph_sub.summary.loc[treatment_col, 'p']
                
                results_list.append({
                    'variable': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                    'hr': hr_sub,
                    'ci_low': ci_sub[0],
                    'ci_high': ci_sub[1],
                    'p_value': p_sub,
                    'n': len(df_sub),
                    'events': int(df_sub[event_col].sum()),
                    'subgroup_val': subgroup_val,
                    'type': 'subgroup'
                })
                logger.info(f"Subgroup {subgroup_col}={subgroup_val}: HR={hr_sub:.3f}, P={p_sub:.4f}")
            except Exception as e:
                logger.warning(f"Cox model fitting failed for subgroup {subgroup_val}: {e}")
        
        # === INTERACTION TEST (Wald test) ===
        try:
            # Create interaction term manually
            df_clean_copy = df_clean.copy()
            # Handle categorical subgroups properly
            subgroup_values = df_clean_copy[subgroup_col]
            if not pd.api.types.is_numeric_dtype(subgroup_values):
                # Convert to numeric codes for interaction
                subgroup_values = pd.Categorical(subgroup_values).codes
            else:
                subgroup_values = df_clean_copy[subgroup_col]

            df_clean_copy['__interaction'] = df_clean_copy[treatment_col] * subgroup_values
            
            # Fit model with interaction
            covariates_with_int = [treatment_col, '__interaction'] + adjustment_cols
            cph_int = CoxPHFitter()
            cph_int.fit(df_clean_copy[[time_col, event_col] + covariates_with_int], 
                        duration_col=time_col, event_col=event_col, show_progress=False)
            
            p_interaction = cph_int.summary.loc['__interaction', 'p']
            logger.info(f"Interaction test (Cox): P={p_interaction:.4f}")
        except Exception as e:
            logger.warning(f"Interaction test failed: {e}")
            p_interaction = np.nan
        
        # === CREATE FOREST PLOT ===
        result_df = pd.DataFrame(results_list)
        
        # Add P-interaction to title
        if not np.isnan(p_interaction):
            het_text = "Heterogeneous" if p_interaction < 0.05 else "Homogeneous"
            title_final = f"{title}<br><span style='font-size: 12px; color: rgba(100,100,100,0.9);'>P for interaction = {p_interaction:.4f} ({het_text})</span>"
        else:
            title_final = title
        
        fig = create_forest_plot(
            data=result_df,
            estimate_col='hr',
            ci_low_col='ci_low',
            ci_high_col='ci_high',
            label_col='variable',
            pval_col='p_value',
            title=title_final,
            x_label=x_label,
            ref_line=1.0,
            height=max(400, len(result_df) * 50 + 100)
        )
        
        if return_stats:
            stats_dict = {
                'overall_hr': hr_overall,
                'overall_ci': (ci_overall[0], ci_overall[1]),
                'overall_p': p_overall,
                'overall_n': len(df_clean),
                'subgroups': {sg.get('subgroup_val', str(i)): sg for i, sg in enumerate(results_list) if sg['type'] == 'subgroup'},
                'p_interaction': p_interaction,
                'heterogeneous': p_interaction < 0.05 if not np.isnan(p_interaction) else None,
                'result_df': result_df
            }
            return fig, stats_dict
        else:
            return fig, result_df
    
    except ImportError:
        st.error("Lifelines library required for Cox regression. Install: pip install lifelines")
        return go.Figure(), {}
    except Exception as e:
        logger.error(f"Cox subgroup analysis failed: {e}")
        st.error(f"Cox subgroup analysis error: {e}")
        return go.Figure(), {}

import re

def _sanitize_filename(name: str) -> str:
    """
    Convert a string into a filesystem-safe, lowercase filename.
    
    Parameters:
        name (str): Input string to sanitize.
    
    Returns:
        str: Sanitized filename where all characters are lowercased and any character other than letters, digits, underscore, or hyphen is replaced with an underscore.
    """
    return re.sub(r'[^\w\-]', '_', name.lower())

def render_forest_plot_in_streamlit(
    data: pd.DataFrame,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str,
    pval_col: str = None,
    title: str = "Forest Plot",
    x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0,
    allow_download: bool = True,
) -> None:
    """
    Render an interactive forest plot in Streamlit and optionally provide HTML and CSV download buttons.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing effect estimates and confidence intervals.
        estimate_col (str): Column name for point estimates.
        ci_low_col (str): Column name for lower bound of the 95% CI.
        ci_high_col (str): Column name for upper bound of the 95% CI.
        label_col (str): Column name used for row labels displayed alongside the plot.
        pval_col (str, optional): Column name for p-values to display; omit to hide p-value column.
        title (str, optional): Plot title used for display and as the base filename for downloads.
        x_label (str, optional): Label for the x axis.
        ref_line (float, optional): Reference vertical line value (e.g., 1.0 for ratios) shown on the plot.
        allow_download (bool, optional): If True, show buttons to download the plot as HTML and the input data as CSV.
    
    Notes:
        - Download filenames are sanitized using the module's filename sanitizer.
        - Errors encountered while creating the plot are surfaced to the user via Streamlit and logged.
    """
    try:
        fig = create_forest_plot(
            data,
            estimate_col=estimate_col,
            ci_low_col=ci_low_col,
            ci_high_col=ci_high_col,
            label_col=label_col,
            pval_col=pval_col,
            title=title,
            x_label=x_label,
            ref_line=ref_line,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if allow_download:
            col1, col2 = st.columns(2)
            
            with col1:
                html_str = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label='📥 Download (HTML)',
                    data=html_str,
                    file_name=f'{_sanitize_filename(title)}.html',
                    mime='text/html',
                )
            
            with col2:
                csv = data.to_csv(index=False)
                st.download_button(
                    label='📥 Download (CSV)',
                    data=csv,
                    file_name=f'{_sanitize_filename(title)}_data.csv',
                    mime='text/csv',
                )
    
    except ValueError as e:
        st.error(f"❌ Error creating forest plot: {e}")
        logger.exception("Forest plot rendering failed")
