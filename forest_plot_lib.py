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
from statsmodels.formula.api import logit, glm
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
import warnings
warnings.filterwarnings('ignore')

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
        Initialize a ForestPlot instance with validated and prepared data for plotting.
        
        Converts the specified estimate and CI columns to numeric (invalid values become NaN), drops rows missing any of those numeric values, reverses the row order for top-down display, and stores column names as instance attributes.
        
        Parameters:
            data (pd.DataFrame): Input dataframe containing effect estimates, confidence bounds, and labels.
            estimate_col (str): Column name for the point estimates.
            ci_low_col (str): Column name for the lower bounds of the confidence intervals.
            ci_high_col (str): Column name for the upper bounds of the confidence intervals.
            label_col (str): Column name for the display labels (variable names).
            pval_col (str, optional): Column name for P-values; include only if P-values should be available for plotting.
        
        Raises:
            ValueError: If `data` is empty, if any required columns are missing, or if no valid rows remain after coercing numeric columns and dropping NaNs.
        
        Side effects:
            - Coerces the estimate and CI columns to numeric, converting non-numeric entries to NaN.
            - Drops rows with NaN in any of the essential numeric columns.
            - Reverses the DataFrame order and resets the index for top-down plotting.
            - Stores the provided column names on the instance (estimate_col, ci_low_col, ci_high_col, label_col, pval_col).
            - Attempts to log the number of variables and the estimate range.
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
        Map a p-value to conventional significance star notation.
        
        Parameters:
            p (float | str | None): The p-value to convert; may be numeric, a string (e.g., "<0.001"), or None/NaN.
        
        Returns:
            str: `'***'` if p < 0.001, `**` if p < 0.01, `*` if p < 0.05, or `''` (empty string) if p is missing or cannot be interpreted as a numeric p-value.
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
        Generate per-row RGBA marker colors based on confidence-interval width and return normalized CI widths.
        
        The provided hex `base_color` is converted to RGB (falls back to teal if parsing fails). Each row's CI width is normalized to the range [0, 1] (smaller = more precise). Marker opacities are mapped so that narrower CIs produce higher opacity and wider CIs produce lower opacity.
        
        Parameters:
            base_color (str): Hex color string (e.g. "#1f9e9d" or "1f9e9d") used as the base for generated RGBA colors.
        
        Returns:
            tuple:
                - marker_colors (list[str]): RGBA color strings for each row, with opacity scaled by CI width.
                - ci_normalized (numpy.ndarray): Normalized CI widths in [0, 1] for each row (smaller means narrower CI).
        """
        # Ensure values are float for calculation
        ci_high = self.data[self.ci_high_col]
        ci_low = self.data[self.ci_low_col]
        
        ci_width = ci_high - ci_low
        
        # Normalize CI width to [0, 1]
        ci_min, ci_max = ci_width.min(), ci_width.max()
        
        # Avoid division by zero
        if ci_max > ci_min:
            ci_normalized = (ci_width - ci_min) / (ci_max - ci_min)
        else:
            ci_normalized = pd.Series([0.5] * len(ci_width))
        
        # Parse base color (hex to RGB)
        hex_color = base_color.lstrip('#')
        if len(hex_color) == 6:
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            rgb = (33, 128, 141)  # Default teal
        
        # Generate colors with varying opacity
        # CI narrow (0) = full opacity (1.0)
        # CI wide (1) = partial opacity (0.5)
        marker_colors = [
            f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {1.0 - 0.5*norm_val:.2f})"
            for norm_val in ci_normalized
        ]
        
        return marker_colors, ci_normalized.values
    
    def get_summary_stats(self, ref_line: float = 1.0):
        """
        Compute summary statistics for the plotted effect estimates and their confidence intervals.
        
        Parameters:
        	ref_line (float): Reference value used to assess whether confidence intervals cross the reference (e.g., 1.0 for ratios). When ref_line > 0, an interval is considered "graphically significant" if its lower bound is greater than ref_line or its upper bound is less than ref_line. When ref_line == 0, significance is assessed by the product of CI bounds being greater than 0.
        
        Returns:
        	dict: Summary values with keys:
        		- n_variables (int): Number of rows (variables) in the plot data.
        		- median_est (float): Median of the estimate column.
        		- min_est (float): Minimum of the estimate column.
        		- max_est (float): Maximum of the estimate column.
        		- n_significant (int or None): Count of rows with p-value < 0.05 if a p-value column was provided, otherwise None.
        		- pct_significant (float or None): Percentage of rows with p-value < 0.05 (0–100) if a p-value column was provided, otherwise None.
        		- n_ci_significant (int): Count of rows whose confidence interval does not cross the reference line (graphical significance).
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
        Create an interactive Plotly forest plot for effect estimates with confidence intervals.
        
        Creates a multi-column, publication-quality forest plot that displays variable labels, formatted estimates with 95% CIs, optional P-values, and a graphical forest plot column with markers and error bars. The layout adapts to the presence of P-values, optionally highlights significance with stars, colors markers by CI width, can draw a reference (no-effect) line, and will use a log x-axis when effect sizes are positive and widely spread.
        
        Parameters:
            title (str): Plot title.
            x_label (str): Label for the plot's x-axis (effect size).
            ref_line (float): Reference line position (no-effect value).
            show_ref_line (bool): If True, draw and annotate the reference line.
            show_sig_stars (bool): If True and P-values are present, append significance stars to labels.
            show_ci_width_colors (bool): If True, color markers by CI width (narrow = more opaque).
            show_sig_divider (bool): If True, add a horizontal divider between significant and non-significant rows when detectable.
            height (int | None): Figure height in pixels; if None, computed from number of rows.
            color (str | None): Base hex color for markers; defaults to module primary color if None.
        
        Returns:
            go.Figure: A Plotly Figure containing the composed forest plot.
        """
        if color is None:
            color = COLORS['primary']
        
        # --- Pre-processing Data for Display ---
        # 1. Format Estimate (95% CI)
        self.data['__display_est'] = self.data.apply(
            lambda x: f"{x[self.estimate_col]:.2f} ({x[self.ci_low_col]:.2f}-{x[self.ci_high_col]:.2f})", axis=1
        )

        # 2. Format P-value (if available)
        if self.pval_col:
            def fmt_p(p):
                """
                Format a p-value into a concise display string.
                
                Parameters:
                    p: The p-value to format; may be numeric, a numeric string, or a string containing comparison symbols (e.g., "<0.001").
                
                Returns:
                    A string representation suitable for display: "<0.001" when the numeric value is less than 0.001, a three-decimal string like "0.123" for numeric values, or the original value converted to string if it cannot be parsed as a number.
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
                Map a p-value to a color label indicating significance.
                
                Parameters:
                    p (str|int|float): P-value or string representation of a p-value (e.g., "0.03", "<0.05").
                
                Returns:
                    str: `"red"` if the numeric p-value is less than 0.05, `"black"` otherwise. If `p` cannot be parsed as a number, returns `"black"`.
                """
                try:
                    if isinstance(p, str) and '<' in p: 
                        # Assume <0.05 is red
                        val = float(p.replace('<','').strip())
                        return "red" if val < 0.05 else "black"
                    
                    val = float(p)
                    return "red" if val < 0.05 else "black"
                except: return "black"

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
        est_min, est_max = self.data[self.estimate_col].min(), self.data[self.estimate_col].max()
        # Use log scale if values are positive and spread is large
        use_log_scale = (est_min > 0) and ((est_max / est_min) > 5)

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
    Create a Plotly forest plot from a DataFrame using a one-line convenience call.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing effect estimates and CIs.
        estimate_col (str): Column name for point estimates.
        ci_low_col (str): Column name for lower confidence interval bounds.
        ci_high_col (str): Column name for upper confidence interval bounds.
        label_col (str): Column name for row labels (variable names).
        pval_col (str, optional): Column name for p-values to display; omit to exclude p-value column.
        title (str, optional): Plot title.
        x_label (str, optional): X-axis label.
        ref_line (float, optional): Reference line value drawn on the effect axis (e.g., 1.0 for ratios).
        height (int, optional): Figure height in pixels.
        **kwargs: Additional keyword arguments forwarded to ForestPlot.create().
    
    Returns:
        go.Figure: A Plotly Figure containing the forest plot. If input validation or plot creation fails,
        logs the error, shows a Streamlit error message, and returns an empty Figure.
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
    Create a forest plot from a dictionary of adjusted odds ratio (aOR) results.
    
    Each entry in `aor_dict` should map a variable name to a mapping containing at least the effect estimate and its confidence bounds, for example:
    {"age": {"aor": 1.5, "ci_low": 1.1, "ci_high": 2.0, "p_value": 0.02}}.
    The function ignores entries missing any of `estimate`, `ci_low`, or `ci_high`.
    
    Parameters:
        aor_dict (dict): Mapping from label to result dict. For each result dict the function looks for keys `'aor'` or `'or'` (estimate), `'ci_low'`, `'ci_high'`, and optionally `'p_value'` or `'p'`.
        title (str): Plot title. Defaults to "Adjusted Odds Ratios".
    
    Returns:
        go.Figure: A Plotly Figure containing the forest plot. Returns an empty Figure if no valid rows are found.
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
    Builds a forest plot figure from a dictionary of Cox regression hazard ratio results.
    
    Parameters:
        hr_dict (dict): Mapping from variable name to a result dict containing hazard ratio and confidence interval.
            Accepted keys in each result dict: 'hr' or 'HR' for the hazard ratio, 'ci_low' or 'CI Lower' for the lower CI bound,
            'ci_high' or 'CI Upper' for the upper CI bound, and optionally 'p_value' or 'p' for the p-value. Entries missing
            any of the HR or CI bounds are ignored.
        title (str): Title for the resulting plot.
    
    Returns:
        go.Figure: A Plotly Figure containing the forest plot configured with hazard ratios, 95% CIs, and optional p-values.
            Returns an empty Figure if no valid results are found.
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
    Create a forest plot from a dictionary of risk-ratio or odds-ratio results.
    
    Converts a mapping of group name -> result dict into a DataFrame and delegates to create_forest_plot. Each result dict should contain the effect estimate (key matching the lowercased `effect_type`, e.g., `'rr'` or `'or'`, or the exact `effect_type` string), lower and upper confidence bounds (keys like `'ci_low'`/`'CI Lower'` and `'ci_high'`/`'CI Upper'`), and optionally a p-value (`'p_value'` or `'p'`). Entries missing the estimate or CI bounds are skipped; if no valid rows remain, an empty Plotly Figure is returned.
    
    Parameters:
        rr_or_dict (dict): Mapping from group label (str) to a result dict containing effect estimate and CIs.
        title (str): Plot title displayed above the forest plot.
        effect_type (str): Metric name, either `'RR'` or `'OR'` (case-insensitive); determines which key is used to read the estimate from each result dict.
    
    Returns:
        go.Figure: A Plotly Figure with the forest plot for the provided effects, or an empty Figure if no valid data was found.
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
    Perform subgroup logistic regression analysis with interaction testing and generate a forest plot.
    
    Fits an overall logistic regression and separate models within each level of `subgroup_col` to extract odds ratios, 95% confidence intervals, and p-values; conducts an interaction test between `treatment_col` and `subgroup_col`; builds a Plotly forest plot summarizing overall and subgroup estimates. Subgroups with fewer than 5 observations are skipped. Raises ValueError if required columns are missing or if fewer than 10 rows remain after dropping missing values.
    
    Parameters:
        df (pd.DataFrame): Input data containing all variables.
        outcome_col (str): Binary outcome column name.
        treatment_col (str): Treatment or exposure column name whose effect is estimated.
        subgroup_col (str): Column name used to define subgroups.
        adjustment_cols (list, optional): Covariate column names to adjust for (default: None).
        title (str, optional): Plot title (default: "Subgroup Analysis (Logistic Regression)").
        x_label (str, optional): X-axis label for the forest plot (default: "Odds Ratio (95% CI)").
        return_stats (bool, optional): If True, return a statistics dictionary alongside the figure; if False, return the figure and the result DataFrame (default: True).
    
    Returns:
        tuple:
            - fig (go.Figure): Plotly figure containing the forest plot.
            - stats_dict (dict) if `return_stats` is True: Dictionary with keys
                'overall_or', 'overall_ci' (tuple), 'overall_p', 'overall_n',
                'subgroups' (mapping of subgroup value to subgroup result dict),
                'p_interaction' (float or NaN), 'heterogeneous' (bool or None),
                and 'result_df' (pd.DataFrame).
              If `return_stats` is False, the second element is the result DataFrame with per-row estimates.
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
    
    Fits an overall Cox model and separate Cox models for each level of `subgroup_col`, performs a Wald-style interaction test between treatment and subgroup, and returns an interactive Plotly forest plot plus optional summary statistics.
    
    Parameters:
        df (pd.DataFrame): Input data containing time, event, treatment, subgroup, and adjustment columns.
        time_col (str): Column name for follow-up duration (duration/time-to-event).
        event_col (str): Column name for event indicator (0/1 or False/True).
        treatment_col (str): Column name for the treatment/exposure variable whose effect is estimated.
        subgroup_col (str): Column name for the categorical subgroup variable to stratify by.
        adjustment_cols (list, optional): List of covariate column names to adjust for in models. Defaults to None.
        title (str, optional): Title for the resulting forest plot. Defaults to "Subgroup Analysis (Cox Regression)".
        x_label (str, optional): X-axis label for the plot. Defaults to "Hazard Ratio (95% CI)".
        return_stats (bool, optional): If True, return a statistics dictionary alongside the figure; if False, return the result DataFrame. Defaults to True.
    
    Returns:
        tuple:
            fig (go.Figure): Plotly Figure containing the forest plot of hazard ratios and confidence intervals.
            stats_dict (dict) or result_df (pd.DataFrame):
                - If `return_stats` is True, a dict with keys including:
                    - 'overall_hr': overall hazard ratio for `treatment_col`
                    - 'overall_ci': tuple (ci_low, ci_high) for the overall HR
                    - 'overall_p': p-value for the treatment effect in the overall model
                    - 'overall_n': number of observations used in the overall model
                    - 'subgroups': mapping of subgroup identifiers to subgroup result entries
                    - 'p_interaction': p-value from the treatment-by-subgroup interaction test (NaN if unavailable)
                    - 'heterogeneous': True if p_interaction < 0.05, False if >= 0.05, None if p_interaction is NaN
                    - 'result_df': DataFrame of the assembled results used to build the plot
                - If `return_stats` is False, the assembled results DataFrame used to build the plot.
    
    Notes:
        - Requires the `lifelines` package for Cox model fitting; raises an ImportError if lifelines is not available.
        - The function skips subgroup levels with insufficient sample size or too few events.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
        
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
                cph_sub.fit(df_sub, duration_col=time_col, event_col=event_col, show_progress=False)
                
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
    Render a forest plot in a Streamlit app and optionally provide download buttons for the plot HTML and the underlying data CSV.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing effect estimates and confidence intervals.
        estimate_col (str): Column name with point estimates.
        ci_low_col (str): Column name with lower confidence interval bounds.
        ci_high_col (str): Column name with upper confidence interval bounds.
        label_col (str): Column name with labels to display for each row.
        pval_col (str, optional): Column name with p-values to display; omit to hide the p-value column.
        title (str, optional): Plot title used for display and as the download file base name.
        x_label (str, optional): X-axis label for the plot.
        ref_line (float, optional): Reference line value drawn on the plot (e.g., 1.0 for ratios).
        allow_download (bool, optional): If True, show buttons to download the figure as HTML and the input data as CSV.
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
                    file_name=f'{title.lower().replace(" ", "_")}.html',
                    mime='text/html',
                )
            
            with col2:
                csv = data.to_csv(index=False)
                st.download_button(
                    label='📥 Download (CSV)',
                    data=csv,
                    file_name=f'{title.lower().replace(" ", "_")}_data.csv',
                    mime='text/csv',
                )
    
    except ValueError as e:
        st.error(f"❌ Error creating forest plot: {e}")
        logger.error(f"Forest plot rendering failed: {e}")