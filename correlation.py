import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import io, base64
import streamlit as st # 🟢 1. IMPORT STREAMLIT

@st.cache_data(show_spinner=False)
def calculate_chi2(df, col1, col2, method='Pearson (Standard)', v1_pos=None, v2_pos=None): # 👈 แก้บรรทัดนี้
    """
    Compute and format a contingency table and association statistics between two categorical columns.
    
    Parameters:
        df (pd.DataFrame): Source data frame containing the columns to analyze.
        col1 (str): Name of the row (exposure) column.
        col2 (str): Name of the column (outcome) column.
        method (str, optional): Test selection string; contains "Fisher" to use Fisher's exact test for 2x2 tables,
            contains "Yates" to apply Yates' continuity correction for chi-square in 2x2 tables, otherwise uses Pearson-style chi-square. Defaults to 'Pearson (Standard)'.
        v1_pos (str|int, optional): Label or string representation of a row value to move to the first row position in the displayed table.
        v2_pos (str|int, optional): Label or string representation of a column value to move to the first column position in the displayed table.
    
    Returns:
        tuple: (display_tab, stats_res, msg, risk_df)
            - display_tab (pd.DataFrame): Formatted table of counts with row percentages as strings "count (pct%)", indexed by col1 with a 'Total' row/column.
            - stats_res (dict or None): Dictionary summarizing the test performed and key statistics (e.g., Test name, Statistic or OR, P-value, Degrees of Freedom, N). None if test could not be computed.
            - msg (str): Human-readable summary of the chosen test and results, or an error/warning message when applicable.
            - risk_df (pd.DataFrame or None): For 2x2 tables, a table of risk metrics (Risk in exposed/unexposed, RR, RD, NNT, OR). None when not applicable or on calculation error.
    
    Notes:
        - If either column name is not present in df, the function returns (None, None, "Columns not found", None).
        - Fisher's exact test is only performed when the contingency table is 2x2; otherwise an explanatory error message is returned in `msg`.
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

    def get_original_label(label_str, df_labels):
        """
        Map a string representation of a label back to the original label object from a collection.
        
        Parameters:
            label_str (str): String representation to match against the labels.
            df_labels (iterable): Collection of labels to search; each label is compared using str(label).
        
        Returns:
            The matching label from `df_labels` whose string form equals `label_str`, or `label_str` if no match is found.
        """
        for lbl in df_labels:
            if str(lbl) == label_str:
                return lbl
        return label_str 
    
    # Reorder Cols
    final_col_order_base = base_col_labels[:]
    if v2_pos is not None: 
        v2_pos_original = get_original_label(v2_pos, base_col_labels)
        if v2_pos_original in final_col_order_base:
            final_col_order_base.remove(v2_pos_original)
            final_col_order_base.insert(0, v2_pos_original)
    else:
        def custom_sort(label):
            """
            Return a sorting key by converting the input to a float when possible.
            
            Parameters:
                label: A value intended as a sort key; may be numeric or non-numeric.
            
            Returns:
                A float if `label` can be converted to float, otherwise the string form of `label`.
            """
            try: return float(label)
            except (ValueError, TypeError): return str(label)
        final_col_order_base.sort(key=custom_sort, reverse=True)
    final_col_order = final_col_order_base + ['Total'] 

    # Reorder Rows
    final_row_order_base = base_row_labels[:]
    if v1_pos is not None: 
        v1_pos_original = get_original_label(v1_pos, base_row_labels)
        if v1_pos_original in final_row_order_base:
            final_row_order_base.remove(v1_pos_original)
            final_row_order_base.insert(0, v1_pos_original)
    else:
        def custom_sort(label):
            """
            Return a sorting key by converting the input to a float when possible.
            
            Parameters:
                label: A value intended as a sort key; may be numeric or non-numeric.
            
            Returns:
                A float if `label` can be converted to float, otherwise the string form of `label`.
            """
            try: return float(label)
            except (ValueError, TypeError): return str(label)
        final_row_order_base.sort(key=custom_sort, reverse=True)
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
            if col_name == 'Total': pct = 100.0
            else: pct = tab_row_pct.loc[row_name, col_name]
            cell_content = f"{count} ({pct:.1f}%)"
            row_data.append(cell_content)
        display_data.append(row_data)
    
    display_tab = pd.DataFrame(display_data, columns=col_names, index=index_names)
    display_tab.index.name = col1
    
    # 3. Stats
    try:
        is_2x2 = (tab_chi2.shape == (2, 2))
        
        # ✅ ตอนนี้ method จะมีค่าแล้ว ไม่ Error
        if "Fisher" in method:
            if not is_2x2:
                return display_tab, None, "Error: Fisher's Exact Test requires a 2x2 table.", None
            
            odds_ratio, p_value = stats.fisher_exact(tab_chi2)
            method_name = "Fisher's Exact Test"
            msg = f"{method_name}: P-value={p_value:.4f}, OR={odds_ratio:.4f}"
            stats_res = {"Test": method_name, "Statistic (OR)": odds_ratio, "P-value": p_value, "Degrees of Freedom": "-", "N": len(data)}
            
        else:
            use_correction = True if "Yates" in method else False
            chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=use_correction)
            
            method_name = "Chi-Square"
            if is_2x2:
                method_name += " (with Yates')" if use_correction else " (Pearson)"
            
            msg = f"{method_name}: Chi2={chi2:.4f}, p={p:.4f}"
            stats_res = {"Test": method_name, "Statistic": chi2, "P-value": p, "Degrees of Freedom": dof, "N": len(data)}
            
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg += " ⚠️ Warning: Expected count < 5. Consider using Fisher's Exact Test."
                
        # 4. Risk
        risk_df = None
        if is_2x2:
            try:
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]
                c, d = vals[1, 0], vals[1, 1]
                row_labels = tab_chi2.index.tolist(); col_labels = tab_chi2.columns.tolist()
                label_exp = str(row_labels[0]); label_unexp = str(row_labels[1]); label_event = str(col_labels[0])
                
                risk_exp = a/(a+b) if (a+b)>0 else 0; risk_unexp = c/(c+d) if (c+d)>0 else 0
                rr = risk_exp/risk_unexp if risk_unexp>0 else np.nan
                rd = risk_exp - risk_unexp; nnt = abs(1/rd) if rd!=0 else np.inf
                odd_ratio, _ = stats.fisher_exact(tab_chi2)
                
                risk_data = [
                    {"Statistic": f"Risk in {label_exp} (R1)", "Value": f"{risk_exp:.4f}", "Interpretation": f"Risk of '{label_event}' in group {label_exp}"},
                    {"Statistic": f"Risk in {label_unexp} (R0)", "Value": f"{risk_unexp:.4f}", "Interpretation": f"Baseline Risk of '{label_event}' in group {label_unexp}"},
                    {"Statistic": "Risk Ratio (RR)", "Value": f"{rr:.4f}", "Interpretation": f"Risk in {label_exp} is {rr:.2f} times that of {label_unexp}"},
                    {"Statistic": "Risk Difference (RD)", "Value": f"{rd:.4f}", "Interpretation": f"Absolute difference (R1 - R0)"},
                    {"Statistic": "Number Needed to Treat (NNT)", "Value": f"{nnt:.1f}", "Interpretation": "Patients to treat to prevent/cause 1 outcome"},
                    {"Statistic": "Odds Ratio (OR)", "Value": f"{odd_ratio:.4f}", "Interpretation": "Odds of Event (Exp vs Unexp)"}
                ]
                risk_df = pd.DataFrame(risk_data)
            except: pass

        return display_tab, stats_res, msg, risk_df
    except Exception as e:
        return display_tab, None, str(e), None

@st.cache_data(show_spinner=False) # 🟢 2. ADD CACHE
def calculate_correlation(df, col1, col2, method='pearson'):
    """
    Compute the correlation between two DataFrame columns and return summary metrics plus a scatter plot.
    
    The function requires that both columns exist in the DataFrame and be numeric (or convertible to numeric). It computes either Pearson (linear) or Spearman (monotonic) correlation, returns the coefficient, p-value, sample size, and a Matplotlib figure showing the scatter and an optional fitted regression line for Pearson.
    
    Parameters:
        method (str): 'pearson' to compute Pearson correlation (linear); any other value computes Spearman correlation (monotonic).
    
    Returns:
        tuple: (metrics, error, figure)
            - metrics (dict): {"Method": str, "Coefficient": float, "P-value": float, "N": int}
            - error (str or None): Error message when columns are missing or non-numeric; otherwise None.
            - figure (matplotlib.figure.Figure): Scatter plot of the two variables with axis labels and title.
    """
    if col1 not in df.columns or col2 not in df.columns: return None, "Columns not found", None
    data = df[[col1, col2]].dropna()
    try:
        v1 = pd.to_numeric(data[col1], errors='raise'); v2 = pd.to_numeric(data[col2], errors='raise')
    except: return None, f"Error: Numeric required.", None
    if method == 'pearson': corr, p = stats.pearsonr(v1, v2); name="Pearson"; desc="Linear"
    else: corr, p = stats.spearmanr(v1, v2); name="Spearman"; desc="Monotonic"
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(v1, v2, alpha=0.6, edgecolors='w', s=50)
    if method == 'pearson':
        try: m, b = np.polyfit(v1, v2, 1); ax.plot(v1, m*v1 + b, 'r--', alpha=0.8)
        except: pass
    ax.set_xlabel(col1); ax.set_ylabel(col2); ax.set_title(f"{col1} vs {col2}"); ax.grid(True, alpha=0.3)
    return {"Method": name, "Coefficient": corr, "P-value": p, "N": len(data)}, None, fig

def generate_report(title, elements):
    # (คงเดิม)
    """
    Generate a complete HTML report containing headings, text, tables, contingency tables, and embedded plots.
    
    Parameters:
        title (str): The report title displayed at the top of the document.
        elements (list[dict]): Ordered list of elements to include in the report. Each element must include:
            - type (str): One of 'text', 'table', 'contingency_table', or 'plot'.
            - data: Content for the element:
                * 'text': a string.
                * 'table': a pandas DataFrame (rendered via DataFrame.to_html()).
                * 'contingency_table': a pandas DataFrame where the index name is used as the exposure label and columns are outcomes; values are rendered into a custom HTML table.
                * 'plot': a matplotlib.figure.Figure, which will be embedded as a PNG image.
            - header (str, optional): Section header displayed above the element.
            - outcome_col (str, optional, contingency_table only): Label to display above the outcome columns (defaults to 'Outcome').
        
    Returns:
        str: A complete HTML document (string) styled and ready for display or saving.
    """
    css_style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; margin: 0; color: #333; }
        .report-container { background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); padding: 20px; width: 100%; box-sizing: border-box; margin-bottom: 20px; }
        h2 { color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        h4 { color: #34495e; margin-top: 25px; margin-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 0.9em; }
        th, td { padding: 10px 15px; border: 1px solid #e0e0e0; vertical-align: middle; text-align: center; } 
        th { background-color: #f0f2f6; font-weight: 600; }
        tr:nth-child(even) td { background-color: #f9f9f9; }
        .report-footer { text-align: right; font-size: 0.75em; color: #666; margin-top: 20px; border-top: 1px dashed #ddd; padding-top: 10px; }
        .th-exposure { text-align: left; background-color: #e8ecf1; } 
        .th-outcome { background-color: #e8ecf1; }
        .td-label { text-align: left; font-weight: bold; background-color: #fcfcfc; }
    </style>
    """
    html = f"<!DOCTYPE html><html><head>{css_style}</head><body>"
    html += f"<div class='report-container'><h2>{title}</h2>"
    
    for element in elements:
        element_type = element['type']
        data = element['data']
        header = element.get('header', '')
        if header: html += f"<h4>{header}</h4>"
        if element_type == 'text': html += f"<p>{data}</p>"
        elif element_type == 'table': 
            idx = not ('Interpretation' in data.columns)
            html += data.to_html(index=idx, classes='report-table')
        elif element_type == 'contingency_table':
            col_labels = data.columns.tolist() 
            row_labels = data.index.tolist()   
            exp_name = data.index.name         
            out_name = element.get('outcome_col', 'Outcome')
            html_tab = "<table>"
            html_tab += "<thead><tr>"
            html_tab += "<th style='background-color: white; border: none;'></th>" 
            html_tab += f"<th colspan='{len(col_labels)}' class='th-outcome'>{out_name}</th>"
            html_tab += "</tr>"
            html_tab += "<tr>"
            html_tab += f"<th class='th-exposure'>{exp_name}</th>"
            for label in col_labels: html_tab += f"<th>{label}</th>"
            html_tab += "</tr></thead>"
            html_tab += "<tbody>"
            for idx_label in row_labels:
                html_tab += "<tr>"
                html_tab += f"<td class='td-label'>{idx_label}</td>"
                for col_label in col_labels:
                    val = data.loc[idx_label, col_label]
                    html_tab += f"<td>{val}</td>"
                html_tab += "</tr>"
            html_tab += "</tbody></table>"
            html += html_tab
        elif element_type == 'plot':
            buf = io.BytesIO()
            if isinstance(data, plt.Figure):
                data.savefig(buf, format='png', bbox_inches='tight'); plt.close(data)
                uri = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{uri}" style="max-width: 100%;"/>'
            buf.close()
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank" style="text-decoration:none; color:inherit;">NTWKKM n Donate</a>. All Rights Reserved. | Powered by GitHub, Gemini, Streamlit
    </div></body></html>"""
    return html