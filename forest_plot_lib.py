"""
Forest Plot Visualization Module

For displaying effect sizes (OR, HR, RR) with confidence intervals across multiple variables.
Supports logistic regression, survival analysis, and epidemiological studies.

Author: NTWKKM
License: MIT
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


class ForestPlot:
    """
    Interactive forest plot generator for statistical results.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing estimate and CI columns
        estimate_col (str): Column name for point estimates
        ci_low_col (str): Column name for CI lower bounds
        ci_high_col (str): Column name for CI upper bounds
        label_col (str): Column name for variable labels
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        estimate_col: str,
        ci_low_col: str,
        ci_high_col: str,
        label_col: str,
    ):
        """
        Initialize ForestPlot with data and column specifications.
        
        Parameters:
            data: DataFrame with results
            estimate_col: Column with point estimates
            ci_low_col: Column with CI lower bounds
            ci_high_col: Column with CI upper bounds
            label_col: Column with variable names/labels
        
        Raises:
            ValueError: If required columns missing or data empty
        """
        # Validation
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_cols = {estimate_col, ci_low_col, ci_high_col, label_col}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.data = data.dropna(subset=[estimate_col, ci_low_col, ci_high_col]).copy()
        if self.data.empty:
            raise ValueError("No valid data after removing NaN values")
        
        # Reverse for top-down display (largest effect at top)
        self.data = self.data.iloc[::-1].reset_index(drop=True)
        
        self.estimate_col = estimate_col
        self.ci_low_col = ci_low_col
        self.ci_high_col = ci_high_col
        self.label_col = label_col
        
        logger.info(
            f"ForestPlot initialized: {len(self.data)} variables, "
            f"estimate range [{self.data[estimate_col].min():.3f}, {self.data[estimate_col].max():.3f}]"
        )
    
    def create(
        self,
        title: str = "Forest Plot",
        x_label: str = "Effect Size (95% CI)",
        ref_line: float = 1.0,
        show_ref_line: bool = True,
        height: int = 600,
        show_values: bool = True,
        color: str = None,
    ) -> go.Figure:
        """
        Generate interactive Plotly forest plot.
        
        Parameters:
            title: Plot title
            x_label: X-axis label
            ref_line: Position of reference line (e.g., 1.0 for OR/HR)
            show_ref_line: Whether to display reference line
            height: Plot height in pixels
            show_values: Display point estimates on plot
            color: Custom marker color (uses COLORS['primary'] if None)
        
        Returns:
            go.Figure: Plotly figure object
        """
        if color is None:
            color = COLORS['primary']
        
        # Extract data
        labels = self.data[self.label_col].astype(str)
        estimates = self.data[self.estimate_col]
        ci_low = self.data[self.ci_low_col]
        ci_high = self.data[self.ci_high_col]
        
        # Calculate error bar sizes
        error_plus = ci_high - estimates
        error_minus = estimates - ci_low
        
        # Create figure
        fig = go.Figure()
        
        # Add main scatter plot with error bars
        fig.add_trace(go.Scatter(
            y=labels,
            x=estimates,
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color=color,
                thickness=2,
                width=6,
            ),
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                line=dict(color='white', width=2),
            ),
            text=[
                f"<b>{label}</b><br>"
                f"Estimate: {est:.3f}<br>"
                f"95% CI: ({low:.3f} - {high:.3f})"
                for label, est, low, high in zip(labels, estimates, ci_low, ci_high)
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Effect Size',
            showlegend=False,
        ))
        
        # Add reference line
        if show_ref_line:
            fig.add_vline(
                x=ref_line,
                line_dash='dash',
                line_color=COLORS['danger'],
                line_width=2,
                annotation_text=f"No Effect (ref={ref_line})",
                annotation_position='top right',
                annotation_font_size=11,
                annotation_font_color=COLORS['danger'],
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color=COLORS['text']),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title=x_label,
                zeroline=False,
                gridcolor='rgba(200, 200, 200, 0.2)',
            ),
            yaxis=dict(
                title='Variable',
                tickfont=dict(size=11),
            ),
            plot_bgcolor='rgba(245, 245, 245, 0.8)',
            height=height,
            hovermode='closest',
            margin=dict(l=200, r=100, t=80, b=60),
        )
        
        logger.info(
            f"Forest plot generated: {title}, {len(self.data)} variables, "
            f"ref_line={ref_line}"
        )
        
        return fig


def create_forest_plot(
    data: pd.DataFrame,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str,
    title: str = "Forest Plot",
    x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0,
    height: int = 600,
    **kwargs
) -> go.Figure:
    """
    Convenience function to create forest plot in one call.
    
    Parameters:
        data: DataFrame with results
        estimate_col: Column with point estimates
        ci_low_col: Column with CI lower bounds
        ci_high_col: Column with CI upper bounds
        label_col: Column with variable labels
        title: Plot title
        x_label: X-axis label
        ref_line: Reference line position
        height: Plot height
        **kwargs: Additional arguments passed to ForestPlot.create()
    
    Returns:
        go.Figure: Plotly figure
    
    Example:
        >>> import pandas as pd
        >>> from forest_plot_lib import create_forest_plot
        >>> 
        >>> results = pd.DataFrame({
        ...     'variable': ['Age', 'Sex', 'BMI'],
        ...     'aor': [1.05, 0.92, 1.12],
        ...     'ci_low': [1.01, 0.75, 1.03],
        ...     'ci_high': [1.09, 1.14, 1.23]
        ... })
        >>> 
        >>> fig = create_forest_plot(
        ...     results,
        ...     estimate_col='aor',
        ...     ci_low_col='ci_low',
        ...     ci_high_col='ci_high',
        ...     label_col='variable',
        ...     title='Adjusted Odds Ratios',
        ...     x_label='aOR (95% CI)'
        ... )
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    try:
        fp = ForestPlot(data, estimate_col, ci_low_col, ci_high_col, label_col)
        fig = fp.create(
            title=title,
            x_label=x_label,
            ref_line=ref_line,
            height=height,
            **kwargs
        )
        return fig
    except ValueError as e:
        logger.error(f"Forest plot creation failed: {e}")
        raise


def create_forest_plot_from_logit(aor_dict: dict, title: str = "Adjusted Odds Ratios") -> go.Figure:
    """
    Convenience function to create forest plot directly from logistic regression aOR results.
    
    Parameters:
        aor_dict: Dictionary with format {variable: {'aor': float, 'ci_low': float, 'ci_high': float}}
        title: Plot title
    
    Returns:
        go.Figure: Plotly figure
    
    Example:
        >>> aor_results = {
        ...     'Age': {'aor': 1.05, 'ci_low': 1.01, 'ci_high': 1.09},
        ...     'Sex': {'aor': 0.92, 'ci_low': 0.75, 'ci_high': 1.14},
        ... }
        >>> fig = create_forest_plot_from_logit(aor_results)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    # Parse aOR results
    data = []
    
    for var, result in aor_dict.items():
        estimate = result.get('aor', result.get('or'))
        ci_low = result.get('ci_low')
        ci_high = result.get('ci_high')
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        data.append({
            'variable': var,
            'aor': float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        })
    
    if not data:
        logger.error("No valid aOR values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    return create_forest_plot(
        df,
        estimate_col='aor',
        ci_low_col='ci_low',
        ci_high_col='ci_high',
        label_col='variable',
        title=title,
        x_label='Odds Ratio (95% CI)',
        ref_line=1.0,
    )


def create_forest_plot_from_cox(hr_dict: dict, title: str = "Hazard Ratios (Cox Regression)") -> go.Figure:
    """
    Convenience function for Cox regression hazard ratios.
    
    Parameters:
        hr_dict: Dictionary with format {variable: {'hr': float, 'ci_low': float, 'ci_high': float}}
        title: Plot title
    
    Returns:
        go.Figure: Plotly figure
    """
    # Parse HR results
    data = []
    
    for var, result in hr_dict.items():
        estimate = result.get('hr', result.get('HR'))
        ci_low = result.get('ci_low', result.get('CI Lower'))
        ci_high = result.get('ci_high', result.get('CI Upper'))
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        data.append({
            'variable': var,
            'hr': float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        })
    
    if not data:
        logger.error("No valid HR values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    return create_forest_plot(
        df,
        estimate_col='hr',
        ci_low_col='ci_low',
        ci_high_col='ci_high',
        label_col='variable',
        title=title,
        x_label='Hazard Ratio (95% CI)',
        ref_line=1.0,
    )


def create_forest_plot_from_rr(
    rr_or_dict: dict,
    title: str = "Risk/Odds Ratios",
    effect_type: str = 'RR'  # 'RR' or 'OR'
) -> go.Figure:
    """
    Convenience function for Risk Ratios or Odds Ratios from Chi-Square analysis.
    
    Parameters:
        rr_or_dict: Dictionary with format {group: {'rr': float, 'ci_low': float, 'ci_high': float}} or similar for OR
        title: Plot title
        effect_type: Type of effect ('RR', 'OR')
    
    Returns:
        go.Figure: Plotly figure
    """
    # Parse RR/OR results
    data = []
    metric_key = effect_type.lower()
    
    for group_name, result in rr_or_dict.items():
        estimate = result.get(metric_key, result.get(effect_type))
        ci_low = result.get('ci_low', result.get('CI Lower'))
        ci_high = result.get('ci_high', result.get('CI Upper'))
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        data.append({
            'variable': group_name,
            metric_key: float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        })
    
    if not data:
        logger.error(f"No valid {effect_type} values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    return create_forest_plot(
        df,
        estimate_col=metric_key,
        ci_low_col='ci_low',
        ci_high_col='ci_high',
        label_col='variable',
        title=title,
        x_label=f'{effect_type} (95% CI)',
        ref_line=1.0,
    )


def render_forest_plot_in_streamlit(
    data: pd.DataFrame,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str,
    title: str = "Forest Plot",
    x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0,
    allow_download: bool = True,
) -> None:
    """
    Display forest plot in Streamlit with optional download button.
    
    Parameters:
        data: DataFrame with results
        estimate_col: Column with point estimates
        ci_low_col: Column with CI lower bounds
        ci_high_col: Column with CI upper bounds
        label_col: Column with variable labels
        title: Plot title
        x_label: X-axis label
        ref_line: Reference line position
        allow_download: Show download button
    
    Example:
        >>> import streamlit as st
        >>> from forest_plot_lib import render_forest_plot_in_streamlit
        >>> 
        >>> render_forest_plot_in_streamlit(
        ...     results_df,
        ...     estimate_col='aor',
        ...     ci_low_col='ci_low',
        ...     ci_high_col='ci_high',
        ...     label_col='variable',
        ...     title='Adjusted Odds Ratios'
        ... )
    """
    try:
        fig = create_forest_plot(
            data,
            estimate_col=estimate_col,
            ci_low_col=ci_low_col,
            ci_high_col=ci_high_col,
            label_col=label_col,
            title=title,
            x_label=x_label,
            ref_line=ref_line,
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: Download button
        if allow_download:
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as HTML
                html_str = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label='📥 Download (HTML)',
                    data=html_str,
                    file_name=f'{title.lower().replace(" ", "_")}.html',
                    mime='text/html',
                )
            
            with col2:
                # Download data as CSV
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
