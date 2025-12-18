import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import io, base64
import streamlit as st
import html as _html
from tabs._common import get_color_palette

# Get unified color palette
COLORS = get_color_palette()

# 🔧 FIX: Line 314 now uses COLORS['text'] instead of COLORS['text_primary']

# 🌟 1. IMPORT STREAMLIT

@st.cache_data(show_spinner=False)
def calculate_chi2(df, col1, col2, method='Pearson (Standard)', v1_pos=None, v2_pos=None):
    """ 
    Compute a contingency table and perform a Chi-square or Fisher's Exact test between two categorical dataframe columns.
    Constructs crosstabs (counts, totals, and row percentages), optionally reorders rows/columns based on v1_pos/v2_pos, 
    and runs the selected statistical test. For 2x2 tables the function also computes common risk metrics 
    (risk, risk ratio, risk difference, NNT, and odds ratio) when possible.
    
    Parameters:
        df (pandas.DataFrame): Source dataframe containing the columns.
        col1 (str): Row (exposure) column name to analyze.
        col2 (str): Column (outcome) column name to analyze.
        method (str, optional): Test selection string. If it contains "Fisher" the function runs Fisher's Exact Test 
            (requires a 2x2 table). If it contains "Yates" a Yates-corrected chi-square is used; 
            otherwise Pearson chi-square is used. Defaults to 'Pearson (Standard)'.
        v1_pos (str | int, optional): If provided, that row label is moved to the first position in the displayed table 
            (useful for ordering exposure groups).
        v2_pos (str | int, optional): If provided, that column label is moved to the first position in the displayed table 
            (useful for ordering outcome categories).
    
    Returns:
        tuple: (display_tab, stats_res, msg, risk_df)
            display_tab (pandas.DataFrame): Formatted contingency table for display where each cell is "count (percentage%)", 
                including totals.
            stats_res (dict | None): Test results and metadata (e.g., {"Test": ..., "Statistic": ..., "P-value": ..., 
                "Degrees of Freedom": ..., "N": ...}) or Fisher-specific keys; None on error.
            msg (str): Human-readable summary of the test result and any warnings (e.g., expected count warnings 
                or Fisher requirement errors).
            risk_df (pandas.DataFrame | None): For 2x2 tables, a table of risk metrics (Risk in exposed/unexposed, RR, RD, NNT, OR); 
                None when not applicable or on failure.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None, None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    
    # 1. Crosstabs
    tab_chi2 = pd.crosstab(data[col1], data[col2])
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    tab_row_pct = pd.crosstab(data[col1], data[col2], normalize='index', margins=True, margins_name="Total") * 100
    
    # --- REORDERING LOGIC ---
    all_col_labels = tab_raw.columns.tolist()
    all_row_labels = tab_raw.index.tolist()
    base_col_labels = [col for col in all_col_labels if col != 'Total']
    base_row_labels = [row for row in all_row_labels if row != 'Total']
    
    # 🌟 Helper Functions (ประกาศครั้งเดียว)
    def get_original_label(label_str, df_labels):
        """ 
        Find the original label from a collection that matches a given string representation.
        """
        for lbl in df_labels:
            if str(lbl) == label_str:
                return lbl
        return label_str
    
    def custom_sort(label):
        """ 
        Produce a sort key for a label by converting numeric-like labels to floats and leaving others as strings.
        Using tuple (priority, value) to handle mixed types safely.
        """
        try:
            return (0, float(label))  # ให้ตัวเลขมาก่อนเสมอ
        except (ValueError, TypeError):
            return (1, str(label))  # ตามด้วยตัวหนังสือ
    
    # --- Reorder Cols ---
    final_col_order_base = base_col_labels[:]
    if v2_pos is not None:
        v2_pos_original = get_original_label(v2_pos, base_col_labels)
        if v2_pos_original in final_col_order_base:
            final_col_order_base.remove(v2_pos_original)
            final_col_order_base.insert(0, v2_pos_original)
    else:
        # Intentional: Sort ascending (smallest to largest) for deterministic order.
        final_col_order_base.sort(key=custom_sort)
    
    final_col_order = final_col_order_base + ['Total']
    
    # --- Reorder Rows ---
    final_row_order_base = base_row_labels[:]
    if v1_pos is not None:
        v1_pos_original = get_original_label(v1_pos, base_row_labels)
        if v1_pos_original in final_row_order_base:
            final_row_order_base.remove(v1_pos_original)
            final_row_order_base.insert(0, v1_pos_original)
    else:
        # Intentional: Sort ascending (smallest to largest) for deterministic order.
        final_row_order_base.sort(key=custom_sort)
    
    final_row_order = final_row_order_base + ['Total']
    
    # Reindex
    tab_raw = tab_raw.reindex(index=final_row_order, columns=final_col_order)
    tab_row_pct = tab_row_pct.reindex(index=final_row_order, columns=final_col_order)
    tab_chi2 = tab_chi2.reindex(index=final_row_order_base, columns=final_col_order_base)
    
    col_names = final_col_order
    index_names = final_row_order
    
    display_data = []
    for row_name in index_names:
        row_data = []
        for col_name in col_names:
            count = tab_raw.loc[row_name, col_name]
            if col_name == 'Total':
                pct = 100.0
            else:
                pct = tab_row_pct.loc[row_name, col_name]
            cell_content = f"{count} ({pct:.1f}%)"
            row_data.append(cell_content)
        display_data.append(row_data)
    
    display_tab = pd.DataFrame(display_data, columns=col_names, index=index_names)
    display_tab.index.name = col1
    
    # 3. Stats
    msg = "" # 🌟 NEW: Initialize msg for warnings only
    try:
        is_2x2 = (tab_chi2.shape == (2, 2))
        
        if "Fisher" in method:
            if not is_2x2:
                return display_tab, None, "Error: Fisher's Exact Test requires a 2x2 table.", None
            
            # Fisher's Exact Test
            odds_ratio, p_value = stats.fisher_exact(tab_chi2)
            method_name = "Fisher's Exact Test"
            
            stats_res = {
                "Test": method_name,
                "Statistic (OR)": f"{odds_ratio:.4f}", # Format Value here
                "P-value": f"{p_value:.4f}",          # Format Value here
                "Degrees of Freedom": "-",
                "N": len(data)
            }
        else:
            # Chi-Square Test
            use_correction = True if "Yates" in method else False
            chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=use_correction)
            method_name = "Chi-Square"
            if is_2x2:
                method_name += " (with Yates')" if use_correction else " (Pearson)"
            
            stats_res = {
                "Test": method_name,
                "Statistic": f"{chi2:.4f}",           # Format Value here
                "P-value": f"{p:.4f}",                # Format Value here
                "Degrees of Freedom": f"{dof}",
                "N": len(data)
            }
            
            # Warning check
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg += " ⚠️ Warning: Expected count < 5. Consider using Fisher's Exact Test."
        
        # 🌟 FIX: res is dict, convert to DataFrame for table display
        stats_df_for_report = pd.DataFrame(stats_res, index=[0]).T.reset_index()
        stats_df_for_report.columns = ['Statistic', 'Value']

        return display_tab, stats_df_for_report, msg, None # 🌟 RETURN DF
    
    except Exception as e:
        return display_tab, None, str(e), None


@st.cache_data(show_spinner=False)
def calculate_correlation(df, col1, col2, method='pearson'):
    """ 
    Compute a correlation between two dataframe columns and produce an interactive scatter plot with optional linear fit using Plotly.
    
    Parameters:
        df (pandas.DataFrame): Source dataframe containing the two columns.
        col1 (str): Column name to use for the x-axis.
        col2 (str): Column name to use for the y-axis.
        method (str): Correlation method: 'pearson' for Pearson (linear) or any other value for Spearman (monotonic).
    
    Returns:
        result (dict or None): If successful, a dictionary with keys "Method" (name), "Coefficient" (correlation value), 
            "P-value" (two-sided p-value), and "N" (number of paired observations); otherwise None.
        error (str or None): An error message when columns are missing or non-numeric; otherwise None.
        fig (plotly.graph_objects.Figure or None): Interactive scatter plot figure with regression line for Pearson; None on error.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None, "Columns not found", None
    
    # 1. Coerce to numeric, turning non-numeric into NaN (handles mixed types gracefully)
    v1_coerced = pd.to_numeric(df[col1], errors='coerce')
    v2_coerced = pd.to_numeric(df[col2], errors='coerce')
    
    # 2. Combine and drop all rows where either column is NaN (original NaN or coerced non-numeric)
    data_numeric = pd.DataFrame({col1: v1_coerced, col2: v2_coerced}).dropna()
    
    # Check if enough numeric data remains (need at least 2 points for correlation)
    if len(data_numeric) < 2:
        return None, "Error: Cannot compute correlation. Columns must contain at least two numeric values.", None
    
    v1 = data_numeric[col1]
    v2 = data_numeric[col2]
    
    if method == 'pearson':
        corr, p = stats.pearsonr(v1, v2)
        name = "Pearson"
        desc = "Linear"
    else:
        corr, p = stats.spearmanr(v1, v2)
        name = "Spearman"
        desc = "Monotonic"
    
    # 🌟 UPDATED: ใช้ Plotly แทน Matplotlib และ unified colors
    fig = go.Figure()
    
    # เพิ่ม scatter plot
    fig.add_trace(go.Scatter(
        x=v1,
        y=v2,
        mode='markers',
        marker={
            'size': 8,
            'color': COLORS['primary'],
            'line': {'color': 'white', 'width': 0.5},
            'opacity': 0.7
        },
        name='Data points',
        hovertemplate=f'{col1}: %{{x:.2f}}<br>{col2}: %{{y:.2f}}<extra></extra>'
    ))
    
    # เพิ่มเส้น regression (เฉพาะ Pearson)
    if method == 'pearson':
        try:
            m, b = np.polyfit(v1, v2, 1)
            x_line = np.array([v1.min(), v1.max()])
            y_line = m * x_line + b
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Linear fit',
                line={'color': COLORS['danger'], 'width': 2, 'dash': 'dash'},
                hovertemplate='Fitted line<extra></extra>'
            ))
        except Exception as e:
            fig.add_annotation(
                text=f"Fit line unavailable: {e}",
                xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
                font={'color': COLORS['danger'], 'size': 11},
            )
    
    # ปรับแต่งเค้าโครงหลาย
    fig.update_layout(
        title={
            'text': f'{col1} vs {col2}<br><sub>{name} correlation (r={corr:.3f}, p={p:.4f})</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=col1,
        yaxis_title=col2,
        hovermode='closest',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        height=500,
        width=700,
        font={'size': 12},
        showlegend=True
    )
    
    # เพิ่มกริด
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return {"Method": name, "Coefficient": corr, "P-value": p, "N": len(data_numeric)}, None, fig


def generate_report(title, elements):
    """ 
    Generate a complete HTML report containing a title and a sequence of report elements.
    Enhanced with unified navy color palette from _common.py.
    
    Parameters:
        title (str): Report title displayed at the top of the page.
        elements (list[dict]): Ordered list of report elements. Each element must include:
            - type (str): One of 'text', 'table', 'contingency_table', 'interpretation', or 'plot'.
            - data: Content for the element:
                - 'text': a string paragraph.
                - 'table': a pandas DataFrame rendered as an HTML table.
                - 'contingency_table': a pandas DataFrame used to build a custom two-row header contingency table.
                - 'interpretation': interpretation text with styling.
                - 'plot': a Plotly Figure to be embedded as interactive HTML.
            - header (str, optional): Section header placed above the element.
    
    Returns:
        str: Complete HTML document as a string, styled and containing the rendered elements.
    """
    
    # 🔧 FIX: Use COLORS['text'] instead of COLORS['text_primary']
    primary_color = COLORS['primary']
    primary_dark = COLORS['primary_dark']
    text_color = COLORS['text']  # 🔧 FIXED: was COLORS['text_primary']
    
    css_style = f""" 
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: {text_color};
            line-height: 1.6;
        }}
        h1 {{
            color: {primary_dark};
            border-bottom: 3px solid {primary_color};
            padding-bottom: 12px;
            font-size: 2em;
            margin-bottom: 20px;
        }}
        h2 {{
            color: {primary_dark};
            margin-top: 25px;
            font-size: 1.35em;
            border-left: 5px solid {primary_color};
            padding-left: 12px;
            margin-bottom: 15px;
        }}
        /* Professional Tables */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            border-radius: 6px;
            overflow: hidden;
        }}
        table th, table td {{
            border: 1px solid #ecf0f1;
            padding: 12px 15px;
            text-align: left;
        }}
        table th {{
            background-color: {primary_color};
            color: white;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        table tr:hover {{
            background-color: #f8f9fa;
        }}
        table tr:nth-child(even) {{
            background-color: #fcfcfc;
        }}
        /* Text formatting */
        p {{
            margin: 12px 0;
            color: {text_color};
        }}
        .metric-text {{
            font-size: 1.02em;
            margin: 10px 0;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}
        .metric-label {{
            font-weight: 600;
            color: {primary_dark};
            min-width: 160px;
        }}
        .metric-value {{
            color: {primary_color};
            font-weight: 600;
            font-family: 'Courier New', monospace;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 4px;
            letter-spacing: 0.3px;
        }}
        /* Interpretation boxes */
        .interpretation {{
            background: linear-gradient(135deg, #ecf0f1 0%, #f8f9fa 100%);
            border-left: 4px solid {primary_color};
            padding: 14px 15px;
            margin: 16px 0;
            border-radius: 5px;
            line-height: 1.7;
            color: {text_color};
        }}
        .interpretation::before {{
            content: "ℹ️ ";
            margin-right: 8px;
        }}
        /* Warning boxes */
        .warning {{
            background: linear-gradient(135deg, #fef5e7 0%, #f9f6f0 100%);
            border-left: 4px solid {COLORS['warning']};
            padding: 14px 15px;
            margin: 16px 0;
            border-radius: 5px;
            color: #7d6608;
            line-height: 1.7;
        }}
        /* Report footer */
        .report-table {{
            border: 1px solid #ecf0f1;
        }}
        .report-footer {{
            text-align: center;
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 40px;
            border-top: 1px solid #ecf0f1;
            padding-top: 20px;
        }}
        .report-footer a {{
            color: {primary_color};
            text-decoration: none;
            transition: color 0.3s ease;
        }}
        .report-footer a:hover {{
            color: {primary_dark};
            text-decoration: underline;
        }}
    </style>
    """
    
    html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{css_style}</head>\n<body>"
    html += f"<h1>{_html.escape(str(title))}</h1>"
    
    for element in elements:
        element_type = element.get('type')
        data = element.get('data')
        header = element.get('header')
        
        if header:
            html += f"<h2>{_html.escape(str(header))}</h2>"
        
        if element_type == 'text':
            text_str = str(data)
            # Check if it's a metric (has : separator)
            if ':' in text_str and len(text_str) < 150:  # Likely a metric, not a full sentence
                parts = text_str.split(':', 1)
                label = _html.escape(parts[0].strip())
                value = _html.escape(parts[1].strip())
                html += f"<p class='metric-text'><span class='metric-label'>{label}:</span> <span class='metric-value'>{value}</span></p>"
            else:
                html += f"<p>{_html.escape(text_str)}</p>"
        
        elif element_type == 'interpretation':
            html += f"<div class='interpretation'>{_html.escape(str(data))}</div>"
        
        elif element_type == 'warning':
            html += f"<div class='warning'>{_html.escape(str(data))}</div>"
        
        elif element_type == 'table':
            # Check if it's a statistics table
            is_stats_table = ('Statistic' in data.columns and 'Value' in data.columns 
                              and data.index.name is None)
            
            html += data.to_html(index=not is_stats_table, classes='report-table', escape=True)
        
        elif element_type == 'contingency_table':
            col_labels = data.columns.tolist()
            row_labels = data.index.tolist()
            exp_name = data.index.name or "Exposure"
            out_name = element.get('outcome_col', 'Outcome')
            
            html += "<table class='report-table'>"
            html += "<thead>"
            
            # First Header Row: Spanning Outcome Column
            html += f"<tr><th></th><th colspan='{len(col_labels)}'>{_html.escape(str(out_name))}</th></tr>"
            
            # Second Header Row: Exposure and all Column Labels
            html += "<tr>"
            html += f"<th>{_html.escape(str(exp_name))}</th>"
            for col_label in col_labels:
                html += f"<th>{_html.escape(str(col_label))}</th>"
            html += "</tr>"
            html += "</thead>"
            
            # Table Body
            html += "<tbody>"
            for idx_label in row_labels:
                html += "<tr>"
                # Row Header (Index name)
                html += f"<td>{_html.escape(str(idx_label))}</td>"
                # Data Cells
                for col_label in col_labels:
                    val = data.loc[idx_label, col_label]
                    html += f"<td>{_html.escape(str(val))}</td>" 
                html += "</tr>"
            html += "</tbody>"
            
            html += "</table>"
        
        elif element_type == 'plot':
            # Check if it's a Plotly Figure
            plot_obj = data
            
            if hasattr(plot_obj, 'to_html'):
                # Plotly Figure
                html += plot_obj.to_html(full_html=False, include_plotlyjs='cdn', div_id=f"plot_{id(plot_obj)}") 
            else:
                # Matplotlib Figure - convert to PNG and embed
                buf = io.BytesIO()
                plot_obj.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%; margin: 20px 0;" />'

    html += f"""<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM n donate</a> | Powered by GitHub, Gemini, Streamlit
    </div>"""
    
    html += "</body>\n</html>"
    return html
