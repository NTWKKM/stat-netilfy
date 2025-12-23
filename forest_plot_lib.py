"""
Forest Plot Visualization Module

For displaying effect sizes (OR, HR, RR) with confidence intervals across multiple variables.
Supports logistic regression, survival analysis, and epidemiological studies.
Updated for Publication Quality Standards (Table + Plot layout).

Author: NTWKKM (Updated by Gemini)
License: MIT
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Added for table-plot layout
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
        pval_col (str): Column name for P-values (Optional)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        estimate_col: str,
        ci_low_col: str,
        ci_high_col: str,
        label_col: str,
        pval_col: str = None,  # Added p-value support
    ):
        """
        Initialize ForestPlot with data and column specifications.
        
        Parameters:
            data: DataFrame with results
            estimate_col: Column with point estimates
            ci_low_col: Column with CI lower bounds
            ci_high_col: Column with CI upper bounds
            label_col: Column with variable names/labels
            pval_col: Column with p-values (optional)
        
        Raises:
            ValueError: If required columns missing or data empty
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
        
        self.data = data.dropna(subset=[estimate_col, ci_low_col, ci_high_col]).copy()
        if self.data.empty:
            raise ValueError("No valid data after removing NaN values")
        
        # Reverse for top-down display (largest effect at top usually, but for table consistency strictly follows input order reversed)
        self.data = self.data.iloc[::-1].reset_index(drop=True)
        
        self.estimate_col = estimate_col
        self.ci_low_col = ci_low_col
        self.ci_high_col = ci_high_col
        self.label_col = label_col
        self.pval_col = pval_col
        
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
        height: int = None, # Auto-height if None
        show_values: bool = True,
        color: str = None,
    ) -> go.Figure:
        """
        Generate interactive Plotly forest plot (Publication Quality).
        
        Parameters:
            title: Plot title
            x_label: X-axis label
            ref_line: Position of reference line (e.g., 1.0 for OR/HR)
            show_ref_line: Whether to display reference line
            height: Plot height in pixels
            show_values: Display point estimates on plot (Deprecated: values are now in table)
            color: Custom marker color (uses COLORS['primary'] if None)
        
        Returns:
            go.Figure: Plotly figure object
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
                try:
                    p = float(p)
                    if p < 0.001: return "<0.001"
                    return f"{p:.3f}"
                except: return ""
            self.data['__display_p'] = self.data[self.pval_col].apply(fmt_p)
            # Determine color for p-values (Red if < 0.05)
            p_text_colors = self.data[self.pval_col].apply(
                lambda p: "red" if (isinstance(p, (int, float)) and p < 0.05) or (isinstance(p, str) and '<' in p) else "black"
            ).tolist()
        else:
            self.data['__display_p'] = ""
            p_text_colors = ["black"] * len(self.data)

        # --- Create Subplots ---
        # Layout: [Variable (25%)] [Est(CI) (20%)] [P-val (10%)] [Plot (45%)]
        fig = make_subplots(
            rows=1, cols=4,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            column_widths=[0.25, 0.20, 0.10, 0.45],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )

        y_pos = list(range(len(self.data)))
        
        # Column 1: Variable Labels
        fig.add_trace(go.Scatter(
            x=[0] * len(self.data),
            y=y_pos,
            text=self.data[self.label_col],
            mode="text",
            textposition="middle right",
            textfont=dict(size=13, color="black"),
            hoverinfo="none"
        ), row=1, col=1)

        # Column 2: Estimate (95% CI)
        fig.add_trace(go.Scatter(
            x=[0] * len(self.data),
            y=y_pos,
            text=self.data['__display_est'],
            mode="text",
            textposition="middle center",
            textfont=dict(size=13, color="black"),
            hoverinfo="none"
        ), row=1, col=2)

        # Column 3: P-value
        fig.add_trace(go.Scatter(
            x=[0] * len(self.data),
            y=y_pos,
            text=self.data['__display_p'],
            mode="text",
            textposition="middle center",
            textfont=dict(size=13, color=p_text_colors),
            hoverinfo="none"
        ), row=1, col=3)

        # Column 4: Forest Plot
        # Reference Line
        if show_ref_line:
            fig.add_vline(
                x=ref_line, line_dash='dash', line_color='gray', line_width=1, 
                row=1, col=4
            )

        # Markers & Error Bars
        fig.add_trace(go.Scatter(
            x=self.data[self.estimate_col],
            y=y_pos,
            error_x=dict(
                type='data',
                symmetric=False,
                array=self.data[self.ci_high_col] - self.data[self.estimate_col],
                arrayminus=self.data[self.estimate_col] - self.data[self.ci_low_col],
                color=color,
                thickness=2,
                width=4
            ),
            mode='markers',
            marker=dict(size=9, color=color, symbol='square'),
            text=self.data[self.label_col],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Estimate: %{x:.3f}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ), row=1, col=4)

        # --- Update Layout ---
        # Calculate dynamic height if not provided
        if height is None:
            height = max(400, len(self.data) * 35 + 120)

        fig.update_layout(
            title=dict(text=title, x=0.01, xanchor='left', font=dict(size=18)),
            height=height,
            showlegend=False,
            template='plotly_white',
            margin=dict(l=10, r=20, t=80, b=40),
            plot_bgcolor='white'
        )

        # Hide Axes for Text Columns (1, 2, 3)
        for c in [1, 2, 3]:
            fig.update_xaxes(visible=False, showgrid=False, zeroline=False, row=1, col=c)
            fig.update_yaxes(visible=False, showgrid=False, zeroline=False, row=1, col=c)

        # Set Axis for Plot Column (4)
        fig.update_yaxes(visible=False, range=[-0.5, len(self.data)-0.5], row=1, col=4)
        fig.update_xaxes(
            title_text=x_label, 
            row=1, col=4,
            gridcolor='rgba(200, 200, 200, 0.2)'
        )

        # --- Add Headers ---
        headers = ["Variable", "Estimate (95% CI)", "P-value", f"{x_label} Plot"]
        for i, h in enumerate(headers, 1):
            fig.add_annotation(
                x=0.5 if i != 1 else 1.0,  # Right align 'Variable' header to match text
                y=1.0, xref=f"x{i} domain", yref="paper",
                text=f"<b>{h}</b>", showarrow=False,
                yanchor="bottom",
                font=dict(size=14, color="black")
            )

        logger.info(f"Forest plot generated: {title}, {len(self.data)} variables")
        return fig


def create_forest_plot(
    data: pd.DataFrame,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str,
    pval_col: str = None, # Added
    title: str = "Forest Plot",
    x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0,
    height: int = None,
    **kwargs
) -> go.Figure:
    """
    Convenience function to create forest plot in one call.
    """
    try:
        fp = ForestPlot(data, estimate_col, ci_low_col, ci_high_col, label_col, pval_col)
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
        st.error(f"Could not create forest plot: {e}") # Fallback for UI
        return go.Figure()


def create_forest_plot_from_logit(aor_dict: dict, title: str = "Adjusted Odds Ratios") -> go.Figure:
    """
    Convenience function to create forest plot directly from logistic regression aOR results.
    """
    # Parse aOR results
    data = []
    
    for var, result in aor_dict.items():
        estimate = result.get('aor', result.get('or'))
        ci_low = result.get('ci_low')
        ci_high = result.get('ci_high')
        p_val = result.get('p_value', result.get('p')) # Try to extract P-value
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        row = {
            'variable': var,
            'aor': float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
        if p_val is not None:
            row['p_value'] = float(p_val)
            
        data.append(row)
    
    if not data:
        logger.error("No valid aOR values to plot")
        return go.Figure()
    
    df = pd.DataFrame(data)
    # Check if p_value column exists in df
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
    Convenience function for Cox regression hazard ratios.
    """
    # Parse HR results
    data = []
    
    for var, result in hr_dict.items():
        estimate = result.get('hr', result.get('HR'))
        ci_low = result.get('ci_low', result.get('CI Lower'))
        ci_high = result.get('ci_high', result.get('CI Upper'))
        p_val = result.get('p_value', result.get('p')) # Try to extract P-value

        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        row = {
            'variable': var,
            'hr': float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
        if p_val is not None:
            row['p_value'] = float(p_val)

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
    effect_type: str = 'RR'  # 'RR' or 'OR'
) -> go.Figure:
    """
    Convenience function for Risk Ratios or Odds Ratios from Chi-Square analysis.
    """
    # Parse RR/OR results
    data = []
    metric_key = effect_type.lower()
    
    for group_name, result in rr_or_dict.items():
        estimate = result.get(metric_key, result.get(effect_type))
        ci_low = result.get('ci_low', result.get('CI Lower'))
        ci_high = result.get('ci_high', result.get('CI Upper'))
        p_val = result.get('p_value', result.get('p')) # Try to extract P-value
        
        if estimate is None or ci_low is None or ci_high is None:
            continue
        
        row = {
            'variable': group_name,
            metric_key: float(estimate),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
        if p_val is not None:
            row['p_value'] = float(p_val)
            
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


def render_forest_plot_in_streamlit(
    data: pd.DataFrame,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str,
    pval_col: str = None, # Added
    title: str = "Forest Plot",
    x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0,
    allow_download: bool = True,
) -> None:
    """
    Display forest plot in Streamlit with optional download button.
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
