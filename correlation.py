import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import io, base64
import streamlit as st
import html as _html

# 🟢 1. IMPORT STREAMLIT

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
    
    # 🟢 Helper Functions (ประกาศครั้งเดียว)
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
    try:
        is_2x2 = (tab_chi2.shape == (2, 2))
        
        if "Fisher" in method:
            if not is_2x2:
                return display_tab, None, "Error: Fisher's Exact Test requires a 2x2 table.", None
            odds_ratio, p_value = stats.fisher_exact(tab_chi2)
            method_name = "Fisher's Exact Test"
            msg = f"{method_name}: P-value={p_value:.4f}, OR={odds_ratio:.4f}"
            stats_res = {
                "Test": method_name,
                "Statistic (OR)": odds_ratio,
                "P-value": p_value,
                "Degrees of Freedom": "-",
                "N": len(data)
            }
        else:
            use_correction = True if "Yates" in method else False
            chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=use_correction)
            method_name = "Chi-Square"
            if is_2x2:
                method_name += " (with Yates')" if use_correction else " (Pearson)"
            msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
            stats_res = {
                "Test": method_name,
                "Statistic": chi2,
                "P-value": p,
                "Degrees of Freedom": dof,
                "N": len(data)
            }
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg += " ⚠️ Warning: Expected count < 5. Consider using Fisher's Exact Test."
        
        # 4. Risk
        risk_df = None
        if is_2x2:
            try:
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                row_labels = tab_chi2.index.tolist()
                col_labels = tab_chi2.columns.tolist()
                label_exp = str(row_labels[0])
                label_unexp = str(row_labels[1])
                label_event = str(col_labels[0])
                
                risk_exp = a/(a+b) if (a+b)>0 else 0
                risk_unexp = c/(c+d) if (c+d)>0 else 0
                rr = risk_exp/risk_unexp if risk_unexp>0 else np.nan
                rd = risk_exp - risk_unexp
                odd_ratio, _ = stats.fisher_exact(tab_chi2)
                
                # 🟢 MODIFIED: Compute NNT or NNH based on Risk Difference sign
                nnt_abs = abs(1/rd) if rd!=0 else np.inf
                nnt_value = f"{nnt_abs:.1f}" if np.isfinite(nnt_abs) else str(np.inf)
                
                if rd < 0: # Risk in exposed is lower -> BENEFIT (Number Needed to Treat)
                    nnt_label = "Number Needed to Treat (NNT)"
                    nnt_interp = f"Treat {nnt_value} patients with {label_exp} to prevent 1 outcome ('{label_event}')"
                elif rd > 0: # Risk in exposed is higher -> HARM (Number Needed to Harm)
                    nnt_label = "Number Needed to Harm (NNH)"
                    nnt_interp = f"Expose {nnt_value} patients to {label_exp} to cause 1 additional outcome ('{label_event}')"
                else: # rd == 0
                    nnt_label = "NNT/NNH"
                    nnt_value = str(np.inf)
                    nnt_interp = "Risk Difference is zero (No absolute effect)"
                # ---------------------
                
                risk_data = [
                    {"Statistic": f"Risk in {label_exp} (R1)", "Value": f"{risk_exp:.4f}", 
                     "Interpretation": f"Risk of '{label_event}' in group {label_exp}"},
                    {"Statistic": f"Risk in {label_unexp} (R0)", "Value": f"{risk_unexp:.4f}", 
                     "Interpretation": f"Baseline Risk of '{label_event}' in group {label_unexp}"},
                    {"Statistic": "Risk Ratio (RR)", "Value": f"{rr:.4f}", 
                     "Interpretation": f"Risk in {label_exp} is {rr:.2f} times that of {label_unexp}"},
                    {"Statistic": "Risk Difference (RD)", "Value": f"{rd:.4f}", 
                     "Interpretation": "Absolute difference (R1 - R0)"},
                    {"Statistic": nnt_label, "Value": nnt_value, 
                     "Interpretation": nnt_interp}, # <-- Updated NNT/NNH presentation
                    {"Statistic": "Odds Ratio (OR)", "Value": f"{odd_ratio:.4f}", 
                     "Interpretation": "Odds of Event (Exp vs Unexp)"}
                ]
                risk_df = pd.DataFrame(risk_data)
            except Exception as e: # 🟢 MODIFIED: Catch specific error for visibility
                msg += f"\n⚠️ Warning: 2x2 risk-metric computation failed: {e}"
                risk_df = None
        
        return display_tab, stats_res, msg, risk_df
    
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
    
    # 🟢 UPDATED: ใช้ Plotly แทน Matplotlib
    fig = go.Figure()
    
    # เพิ่ม scatter plot
    fig.add_trace(go.Scatter(
        x=v1,
        y=v2,
        mode='markers',
        marker=dict(
            size=8,
            color='rgba(0, 100, 200, 0.6)',
            line=dict(color='white', width=0.5),
            opacity=0.7
        ),
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
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Fitted line<extra></extra>'
            ))
        except Exception as e:
            fig.add_annotation(
                text=f"Fit line unavailable: {e}",
                xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
                font=dict(color="darkred", size=11),
            )
    
    # ปรับแต่งเค้าโครง
    fig.update_layout(
        title=dict(
            text=f'{col1} vs {col2}<br><sub>{name} correlation (r={corr:.3f}, p={p:.4f})</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=col1,
        yaxis_title=col2,
        hovermode='closest',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        height=500,
        width=700,
        font=dict(size=12),
        showlegend=True
    )
    
    # เพิ่มกริด
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return {"Method": name, "Coefficient": corr, "P-value": p, "N": len(data_numeric)}, None, fig


def generate_report(title, elements):
    """ 
    Generate a complete HTML report containing a title and a sequence of report elements.
    
    Parameters:
        title (str): Report title displayed at the top of the page.
        elements (list[dict]): Ordered list of report elements. Each element must include:
            - type (str): One of 'text', 'table', 'contingency_table', or 'plot'.
            - data: Content for the element:
                - 'text': a string paragraph.
                - 'table': a pandas DataFrame rendered as an HTML table.
                - 'contingency_table': a pandas DataFrame used to build a custom two-row header contingency table 
                  (index.name used as exposure label).
                - 'plot': a Plotly Figure to be embedded as interactive HTML.
            - header (str, optional): Section header placed above the element.
            - outcome_col (str, optional, only for 'contingency_table'): label for the outcome header (defaults to "Outcome").
    
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
        .report-table {
            border: 1px solid #ddd;
        }
        .report-footer {
            text-align: right;
            font-size: 0.75em;
            color: var(--text-color);
            margin-top: 20px;
            border-top: 1px dashed var(--border-color);
            padding-top: 10px;
        }
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
            html += f"<p>{_html.escape(str(data))}</p>"
        
        elif element_type == 'table':
            idx = 'Interpretation' not in data.columns
            html += data.to_html(index=idx, classes='report-table')
        
        elif element_type == 'contingency_table':
            col_labels = data.columns.tolist()
            row_labels = data.index.tolist()
            exp_name = data.index.name or "Exposure"
            out_name = element.get('outcome_col', 'Outcome')
            
            # 🟢 MODIFIED: Replaced Markdown pipe table with proper HTML table structure
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
                    # Ensure value is treated as string and escaped
                    html += f"<td>{_html.escape(str(val))}</td>" 
                html += "</tr>"
            html += "</tbody>"
            
            html += "</table>"
        
        elif element_type == 'plot':
            # ตรวจสอบว่าเป็น Plotly Figure หรือ Matplotlib Figure
            plot_obj = data
            
            # ถ้าเป็น Plotly Figure
            if hasattr(plot_obj, 'to_html'):
                # 🟢 MODIFIED: Use include_plotlyjs='cdn' for portability 
                # and remove global CDN injection at the bottom.
                html += plot_obj.to_html(full_html=False, include_plotlyjs='cdn', div_id=f"plot_{id(plot_obj)}") 
            else:
                # ถ้าเป็น Matplotlib Figure - แปลงเป็น PNG และ embed
                buf = io.BytesIO()
                plot_obj.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{b64}" />'
    
    # 🟢 REMOVED: Removed the duplicate global CDN script injection
    # html += """ 
    # <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
    # """

    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div>"""
    
    html += "</body>\n</html>"
    return html
