"""
Forest Plot Visualization Module

For displaying effect sizes (OR, HR, RR) with confidence intervals across multiple variables.
Supports logistic regression, survival analysis, and epidemiological studies.
Optimized for Multivariable Analysis (standard Regression output).

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
        Initialize ForestPlot with data and column specifications.
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
        
        # Reverse for top-down display
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
    
    def _add_significance_stars(self, p):
        """
        Convert p-value to significance stars (* ** ***).
        """
        if pd.isna(p) or not isinstance(p, (int, float)):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""
    
    def _get_ci_width_colors(self, base_color: str) -> list:
        """
        🎨 Generate marker colors based on CI width.
        CI width = precision indicator
        - Narrow CI (precise) = darker/more opaque
        - Wide CI (uncertain) = lighter/more transparent
        """
        ci_width = self.data[self.ci_high_col] - self.data[self.ci_low_col]
        
        # Normalize CI width to [0, 1]
        ci_min, ci_max = ci_width.min(), ci_width.max()
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
        📊 Return summary statistics for forest plot.
        
        Returns:
            dict: Contains n_variables, median_est, min/max_est, n_significant, pct_significant
        """
        # Count significant (p < 0.05)
        if self.pval_col and self.pval_col in self.data.columns:
            n_sig = (self.data[self.pval_col] < 0.05).sum()
            pct_sig = 100 * n_sig / len(self.data) if len(self.data) > 0 else 0
        else:
            n_sig = pct_sig = None
        
        # Count CI doesn't cross ref_line (graphical significance)
        ci_sig = (
            ((self.data[self.ci_low_col] > ref_line) | (self.data[self.ci_high_col] < ref_line))
            if ref_line > 0
            else ((self.data[self.ci_low_col] * self.data[self.ci_high_col]) > 0)
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
        Generate interactive Plotly forest plot (Publication Quality).
        Optimized for Multivariable Analysis (Logistic/Cox Regression).
        
        Features:
        - Dynamic column layout (adapts to presence of P-value)
        - Square markers with white border (standard Regression format)
        - Auto-detected log scale
        - Clear reference line with annotation
        - Complete hover information
        - Significance stars for p-values
        - 🎨 CI width color coding (narrow=opaque, wide=transparent)
        - ✂️ Significant/Non-significant split divider
        - 📊 Summary statistics in subtitle
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
            p_text_colors = self.data[self.pval_col].apply(
                lambda p: "red" if (isinstance(p, (int, float)) and p < 0.05) or (isinstance(p, str) and '<' in p) else "black"
            ).tolist()
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
        use_log_scale = est_min > 0 and (est_max / est_min > 5)

        if show_ref_line:
            fig.add_vline(x=ref_line, line_dash='dash', line_color='rgba(192, 21, 47, 0.6)', line_width=2, annotation_text=f'No Effect ({ref_line})', annotation_position='top', row=1, col=plot_col)
        
        # ✂️ Add horizontal divider between Significant and Non-significant
        if show_sig_divider:
            ci_sig = ((self.data[self.ci_low_col] > ref_line) | (self.data[self.ci_high_col] < ref_line)) if ref_line > 0 else ((self.data[self.ci_low_col] * self.data[self.ci_high_col]) > 0)
            if ci_sig.any() and (~ci_sig).any():
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
            self.data[self.pval_col] if has_pval else [None]*len(self.data),
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
    """Convenience function to create forest plot in one call."""
    try:
        fp = ForestPlot(data, estimate_col, ci_low_col, ci_high_col, label_col, pval_col)
        return fp.create(title=title, x_label=x_label, ref_line=ref_line, height=height, **kwargs)
    except ValueError as e:
        logger.error(f"Forest plot creation failed: {e}")
        st.error(f"Could not create forest plot: {e}")
        return go.Figure()


def create_forest_plot_from_logit(aor_dict: dict, title: str = "Adjusted Odds Ratios") -> go.Figure:
    """
    Convenience function to create forest plot directly from logistic regression aOR results.
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
            row['p_value'] = float(p_val)
            
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
    Convenience function for Cox regression hazard ratios.
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
    effect_type: str = 'RR'
) -> go.Figure:
    """
    Convenience function for Risk Ratios or Odds Ratios from Chi-Square analysis.
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
    pval_col: str = None,
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
