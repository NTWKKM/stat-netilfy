import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score
import plotly.graph_objects as go
import plotly.express as px
import io, base64
import streamlit as st
import html as _html
import warnings


# ========================================
# 1. DESCRIPTIVE STATISTICS
# ========================================

@st.cache_data(show_spinner=False)
def calculate_descriptive(df, col):
    """คำนวณสถิติพื้นฐาน"""
    if col not in df.columns:
        return None
    
    data = df[col].dropna()
    
    try:
        num_data = pd.to_numeric(data, errors='raise')
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False
    
    if is_numeric:
        desc = num_data.describe()
        return pd.DataFrame({
            "Statistic": ["Count", "Mean", "SD", "Median", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
            "Value": [
                f"{desc['count']:.0f}",
                f"{desc['mean']:.4f}",
                f"{desc['std']:.4f}",
                f"{desc['50%']:.4f}",
                f"{desc['min']:.4f}",
                f"{desc['max']:.4f}",
                f"{desc['25%']:.4f}",
                f"{desc['75%']:.4f}"
            ]
        })
    else:
        counts = data.value_counts()
        percent = data.value_counts(normalize=True) * 100
        return pd.DataFrame({
            "Category": counts.index,
            "Count": counts.values,
            "Percentage (%)": percent.values
        }).sort_values("Count", ascending=False)


# ========================================
# 2. CI CALCULATION FUNCTIONS
# ========================================

def calculate_ci_wilson_score(successes, n, ci=0.95):
    """
    Wilson Score Interval for binomial proportion
    More accurate than Wald interval, especially for extreme proportions
    """
    if n <= 0:
        return np.nan, np.nan
    
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    p_hat = successes / n if n > 0 else 0
    denominator = 1 + (z**2 / n)
    centre_adjusted_probability = (p_hat + (z**2 / (2 * n))) / denominator
    adjusted_standard_error = np.sqrt((p_hat * (1 - p_hat) + (z**2 / (4 * n))) / n) / denominator
    lower = max(0, centre_adjusted_probability - z * adjusted_standard_error)
    upper = min(1, centre_adjusted_probability + z * adjusted_standard_error)
    return lower, upper


def calculate_ci_log_odds(or_value, se_log_or, ci=0.95):
    """
    Confidence Interval for Odds Ratio using log scale
    """
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    log_or = np.log(or_value) if or_value > 0 else np.nan
    if np.isnan(log_or) or np.isnan(se_log_or) or se_log_or <= 0:
        return np.nan, np.nan
    lower_log = log_or - z * se_log_or
    upper_log = log_or + z * se_log_or
    return np.exp(lower_log), np.exp(upper_log)


def calculate_ci_rr(risk_exp, n_exp, risk_unexp, n_unexp, ci=0.95):
    """
    Confidence Interval for Risk Ratio using log scale
    """
    rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
    if np.isnan(rr) or rr <= 0:
        return np.nan, np.nan
    
    # Robins standard error
    se_log_rr = np.sqrt((1 - risk_exp) / (risk_exp * n_exp) + 
                        (1 - risk_unexp) / (risk_unexp * n_unexp)) if (risk_exp > 0 and risk_unexp > 0) else np.nan
    
    if np.isnan(se_log_rr) or se_log_rr <= 0:
        return np.nan, np.nan
    
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    log_rr = np.log(rr)
    lower_log = log_rr - z * se_log_rr
    upper_log = log_rr + z * se_log_rr
    return np.exp(lower_log), np.exp(upper_log)


def calculate_ci_nnt(rd, rd_se, ci=0.95):
    """
    Confidence Interval for NNT
    Based on CI of Risk Difference
    """
    if abs(rd) < 0.001:
        return np.nan, np.nan

    if np.isnan(rd_se) or rd_se <= 0:
        return np.nan, np.nan

    z = stats.norm.ppf(1 - (1 - ci) / 2)
    rd_lower = rd - z * rd_se
    rd_upper = rd + z * rd_se

    # NNT CI is inverse of RD CI (bounds swap)
    # Handle case where CI crosses zero
    if rd_lower * rd_upper <= 0:
        return np.nan, np.nan

    nnt_lower = abs(1 / rd_upper)
    nnt_upper = abs(1 / rd_lower)
    return min(nnt_lower, nnt_upper), max(nnt_lower, nnt_upper)

# ========================================
# 2. CHI-SQUARE & FISHER'S EXACT TEST + DIAGNOSTIC METRICS
# ========================================

@st.cache_data(show_spinner=False)
def calculate_chi2(df, col1, col2, method='Pearson (Standard)', v1_pos=None, v2_pos=None):
    """
    Comprehensive 2x2 contingency table analysis with:
    - Chi-Square / Fisher's Exact Test
    - Risk metrics (OR, RR, NNT) with 95% CI
    - Diagnostic metrics (Sensitivity, Specificity, PPV, NPV, LR+, LR-)
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
    
    def get_original_label(label_str: str, df_labels: list) -> any:
        for lbl in df_labels:
            if str(lbl) == label_str:
                return lbl
        return label_str
    
    def custom_sort(label) -> tuple:
        try:
            return (0, float(label))
        except (ValueError, TypeError):
            return (1, str(label))
    
    # --- Reorder Cols ---
    final_col_order_base = base_col_labels[:]
    if v2_pos is not None:
        v2_pos_original = get_original_label(v2_pos, base_col_labels)
        if v2_pos_original in final_col_order_base:
            final_col_order_base.remove(v2_pos_original)
            final_col_order_base.insert(0, v2_pos_original)
    else:
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
            cell_content = f"{int(count)} ({pct:.1f}%)"
            row_data.append(cell_content)
        display_data.append(row_data)
    
    display_tab = pd.DataFrame(display_data, columns=col_names, index=index_names)
    display_tab.index.name = col1
    
    # 2. Stats Test
    msg = ""
    try:
        is_2x2 = (tab_chi2.shape == (2, 2))
        
        if "Fisher" in method:
            if not is_2x2:
                return display_tab, None, "Error: Fisher's Exact Test requires a 2x2 table.", None
            
            odds_ratio, p_value = stats.fisher_exact(tab_chi2)
            method_name = "Fisher's Exact Test"
            
            stats_res = {
                "Test": method_name,
                "Statistic (OR)": f"{odds_ratio:.4f}",
                "P-value": f"{p_value:.4f}",
                "Degrees of Freedom": "-",
                "N": len(data)
            }
        else:
            use_correction = True if "Yates" in method else False
            chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=use_correction)
            method_name = "Chi-Square"
            if is_2x2:
                method_name += " (with Yates')" if use_correction else " (Pearson)"
            
            stats_res = {
                "Test": method_name,
                "Statistic": f"{chi2:.4f}",
                "P-value": f"{p:.4f}",
                "Degrees of Freedom": f"{dof}",
                "N": len(data)
            }
            
            # Warning check
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg = "⚠️ Warning: Expected count < 5. Consider using Fisher's Exact Test."
        
        stats_df_for_report = pd.DataFrame(stats_res, index=[0]).T.reset_index()
        stats_df_for_report.columns = ['Statistic', 'Value']

        # 3. 2x2 Risk & Diagnostic Metrics
        risk_df = None
        if is_2x2:
            try:
                vals = tab_chi2.values
                a, b = vals[0, 0], vals[0, 1]  # Exposed: Event, No Event
                c, d = vals[1, 0], vals[1, 1]  # Unexposed: Event, No Event
                
                row_labels = tab_chi2.index.tolist()
                col_labels = tab_chi2.columns.tolist()
                label_exp = str(row_labels[0])
                label_unexp = str(row_labels[1])
                label_event = str(col_labels[0])
                
                # RISK METRICS
                risk_exp = a / (a + b) if (a + b) > 0 else 0
                risk_unexp = c / (c + d) if (c + d) > 0 else 0
                rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
                rd = risk_exp - risk_unexp
                nnt_abs = abs(1 / rd) if abs(rd) > 0.001 else np.inf
                
                # Odds Ratio with CI
                or_value = (a * d) / (b * c) if (b * c) != 0 else np.nan
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if (a > 0 and b > 0 and c > 0 and d > 0) else np.nan
                or_ci_lower, or_ci_upper = calculate_ci_log_odds(or_value, se_log_or)
                
                # Risk Ratio with CI
                if (a+b) > 0 and (c+d) > 0 and risk_exp > 0 and risk_unexp > 0:
                    rr_ci_lower, rr_ci_upper = calculate_ci_rr(risk_exp, a+b, risk_unexp, c+d)
                else:
                    rr_ci_lower, rr_ci_upper = np.nan, np.nan
                
                # NNT with CI
                rd_se = np.sqrt(
                    (risk_exp * (1 - risk_exp)) / (a + b) + 
                    (risk_unexp * (1 - risk_unexp)) / (c + d)
                ) if (a + b) > 0 and (c + d) > 0 else np.nan
                nnt_ci_lower, nnt_ci_upper = calculate_ci_nnt(rd, rd_se)
                
                # DIAGNOSTIC METRICS
                sensitivity = a / (a + c) if (a + c) > 0 else 0
                se_ci_lower, se_ci_upper = calculate_ci_wilson_score(a, a + c)
                
                specificity = d / (b + d) if (b + d) > 0 else 0
                sp_ci_lower, sp_ci_upper = calculate_ci_wilson_score(d, b + d)
                
                ppv = a / (a + b) if (a + b) > 0 else 0
                ppv_ci_lower, ppv_ci_upper = calculate_ci_wilson_score(a, a + b)
                
                npv = d / (c + d) if (c + d) > 0 else 0
                npv_ci_lower, npv_ci_upper = calculate_ci_wilson_score(d, c + d)
                
                lr_plus = sensitivity / (1 - specificity) if (1 - specificity) > 0 else np.nan
                lr_minus = (1 - sensitivity) / specificity if specificity > 0 else np.nan
                
                if rd < 0:
                    nnt_label = "Number Needed to Treat (NNT)"
                elif rd > 0:
                    nnt_label = "Number Needed to Harm (NNH)"
                else:
                    nnt_label = "NNT/NNH"
                
                # Build Risk Metrics Table with SECTION HEADERS
                risk_data = [
                    {"Metric": "RISK METRICS (Assumes: Rows=Exposure Status, Cols=Outcome Status)", 
                     "Value": "", "95% CI": "", "Interpretation": "Use for cohort/case-control studies"},
                    {"Metric": "Risk in Exposed (R1)", "Value": f"{risk_exp:.4f}", 
                     "95% CI": "-", "Interpretation": f"Risk of '{label_event}' in {label_exp}"},
                    {"Metric": "Risk in Unexposed (R0)", "Value": f"{risk_unexp:.4f}", 
                     "95% CI": "-", "Interpretation": f"Baseline risk of '{label_event}' in {label_unexp}"},
                    {"Metric": "Risk Ratio (RR)", "Value": f"{rr:.4f}", 
                     "95% CI": f"({rr_ci_lower:.4f} - {rr_ci_upper:.4f})" if np.isfinite(rr_ci_lower) else "N/A",
                     "Interpretation": f"Risk in {label_exp} is {rr:.2f}x that of {label_unexp}"},
                    {"Metric": "Risk Difference (RD)", "Value": f"{rd:.4f}", 
                     "95% CI": "-", "Interpretation": "Absolute risk difference (R1 - R0)"},
                    {"Metric": nnt_label, "Value": f"{nnt_abs:.1f}", 
                     "95% CI": f"({nnt_ci_lower:.1f} - {nnt_ci_upper:.1f})" if np.isfinite(nnt_ci_lower) else "N/A",
                     "Interpretation": "Patients to treat to prevent/cause 1 outcome"},
                    {"Metric": "Odds Ratio (OR)", "Value": f"{or_value:.4f}", 
                     "95% CI": f"({or_ci_lower:.4f} - {or_ci_upper:.4f})" if np.isfinite(or_ci_lower) else "N/A",
                     "Interpretation": f"Odds of '{label_event}' ({label_exp} vs {label_unexp})"},
                    {"Metric": "DIAGNOSTIC METRICS (Assumes: Rows=Test Result, Cols=Disease Status)", 
                     "Value": "", "95% CI": "", "Interpretation": "Use for diagnostic/screening studies"},
                    {"Metric": "Sensitivity", "Value": f"{sensitivity:.4f}", 
                     "95% CI": f"({se_ci_lower:.4f} - {se_ci_upper:.4f})",
                     "Interpretation": "P(Test+ | Disease+) - True Positive Rate"},
                    {"Metric": "Specificity", "Value": f"{specificity:.4f}", 
                     "95% CI": f"({sp_ci_lower:.4f} - {sp_ci_upper:.4f})",
                     "Interpretation": "P(Test- | Disease-) - True Negative Rate"},
                    {"Metric": "PPV (Positive Predictive Value)", "Value": f"{ppv:.4f}", 
                     "95% CI": f"({ppv_ci_lower:.4f} - {ppv_ci_upper:.4f})",
                     "Interpretation": "P(Disease+ | Test+) - Precision"},
                    {"Metric": "NPV (Negative Predictive Value)", "Value": f"{npv:.4f}", 
                     "95% CI": f"({npv_ci_lower:.4f} - {npv_ci_upper:.4f})",
                     "Interpretation": "P(Disease- | Test-) - Negative Precision"},
                    {"Metric": "LR+ (Likelihood Ratio +)", "Value": f"{lr_plus:.4f}", 
                     "95% CI": "-", "Interpretation": "Sensitivity / (1 - Specificity)"},
                    {"Metric": "LR- (Likelihood Ratio -)", "Value": f"{lr_minus:.4f}", 
                     "95% CI": "-", "Interpretation": "(1 - Sensitivity) / Specificity"},
                ]
                risk_df = pd.DataFrame(risk_data)
            except (ZeroDivisionError, ValueError, KeyError) as e:
                risk_df = None
                msg += f" (Risk metrics unavailable: {e!s})"
        
        return display_tab, stats_df_for_report, msg, risk_df
            
    except Exception as e:
        return display_tab, None, str(e), None


# ========================================
# 3. COHEN'S KAPPA
# ========================================

@st.cache_data(show_spinner=False)
def calculate_kappa(df, col1, col2):
    """คำนวณ Cohen's Kappa ระหว่าง 2 raters"""
    if col1 not in df.columns or col2 not in df.columns:
        return None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    if data.empty:
        return None, "No data after dropping NAs", None
    
    y1 = data[col1].astype(str)
    y2 = data[col2].astype(str)
    
    try:
        kappa = cohen_kappa_score(y1, y2)
        
        # Interpretation (Landis & Koch, 1977)
        if kappa < 0:
            interp = "Poor agreement"
        elif kappa <= 0.20:
            interp = "Slight agreement"
        elif kappa <= 0.40:
            interp = "Fair agreement"
        elif kappa <= 0.60:
            interp = "Moderate agreement"
        elif kappa <= 0.80:
            interp = "Substantial agreement"
        else:
            interp = "Perfect/Almost perfect agreement"
        
        res_df = pd.DataFrame({
            "Statistic": ["Cohen's Kappa", "N (Pairs)", "Interpretation"],
            "Value": [f"{kappa:.4f}", f"{len(data)}", interp]
        })
        
        # Confusion Matrix
        conf_matrix = pd.crosstab(y1, y2, rownames=[f"{col1} (Obs 1)"], colnames=[f"{col2} (Obs 2)"])
        
    except ValueError as e:
        return None, str(e), None
    
    return res_df, None, conf_matrix


# ========================================
# 4. ROC CURVE
# ========================================

def auc_ci_hanley_mcneil(auc, n1, n2):
    """Calculate 95% CI for AUC using Hanley & McNeil method"""
    q1 = auc / (2 - auc)
    q2 = 2 * (auc**2) / (1 + auc)
    se_auc = np.sqrt(((auc * (1 - auc)) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1 * n2))
    return auc - 1.96 * se_auc, auc + 1.96 * se_auc, se_auc


def auc_ci_delong(y_true, y_scores):
    """Calculate 95% CI for AUC using DeLong method (Robust Version)"""
    try:
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        distinct_value_indices = np.where(np.diff(y_scores))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        
        n_pos = tps[-1]
        n_neg = fps[-1]
        
        if n_pos <= 1 or n_neg <= 1:
            return np.nan, np.nan, np.nan
        
        auc = roc_auc_score(y_true, y_scores)
        
        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]
        
        v10 = []
        v01 = []
        
        for p in pos_scores:
            v10.append((np.sum(p > neg_scores) + 0.5*np.sum(p == neg_scores)) / n_neg)
        
        for n in neg_scores:
            v01.append((np.sum(pos_scores > n) + 0.5*np.sum(pos_scores == n)) / n_pos)
        
        s10 = np.var(v10, ddof=1) if len(v10) > 1 else 0
        s01 = np.var(v01, ddof=1) if len(v01) > 1 else 0
        se_auc = np.sqrt((s10 / n_pos) + (s01 / n_neg))
        
        return auc - 1.96*se_auc, auc + 1.96*se_auc, se_auc
    except Exception as e:
        warnings.warn(f"DeLong CI calculation failed: {e}", stacklevel=2)
        return np.nan, np.nan, np.nan


@st.cache_data(show_spinner=False)
def analyze_roc(df, truth_col, score_col, method='delong', pos_label_user=None):
    """
    Analyze ROC curve using Plotly for interactive visualization.
    """
    data = df[[truth_col, score_col]].dropna()
    y_true_raw = data[truth_col]
    y_score = pd.to_numeric(data[score_col], errors='coerce').dropna()
    y_true_raw = y_true_raw.loc[y_score.index]
    
    if y_true_raw.nunique() != 2 or pos_label_user is None:
        return None, "Error: Binary outcome required (must have exactly 2 unique classes).", None, None
    
    y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)
    
    if y_score.nunique() < 2:
        return None, "Error: Prediction score is constant (single value). Cannot compute ROC.", None, None

    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    
    if n1 == 0 or n0 == 0:
        return None, "Error: Need both Positive and Negative cases after dropping NA scores.", None, None
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    
    if method == 'delong':
        ci_lower, ci_upper, se = auc_ci_delong(y_true, y_score.values)
        m_name = "DeLong"
    else:
        ci_lower, ci_upper, se = auc_ci_hanley_mcneil(auc_val, n1, n0)
        m_name = "Hanley"
    
    p_val_auc = (stats.norm.sf(abs((auc_val - 0.5) / se)) * 2 
                 if (se is not None and np.isfinite(se) and se > 0) 
                 else np.nan)
    
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    ci_lower_f = ci_lower if np.isfinite(ci_lower) else np.nan
    ci_upper_f = ci_upper if np.isfinite(ci_upper) else np.nan
    stats_res = {
        "AUC": auc_val,
        "SE": se,
        "95% CI Lower": max(0, ci_lower_f) if np.isfinite(ci_lower_f) else np.nan,
        "95% CI Upper": min(1, ci_upper_f) if np.isfinite(ci_upper_f) else np.nan,
        "Method": m_name,
        "P-value": p_val_auc,
        "Youden J": j_scores[best_idx],
        "Best Cut-off": thresholds[best_idx],
        "Sensitivity": tpr[best_idx],
        "Specificity": 1-fpr[best_idx],
        "N(+)": n1,
        "N(-)": n0,
        "Positive Label": pos_label_user
    }
    
    # Create Plotly ROC curve
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC={auc_val:.3f})',
        line={'color': 'darkorange', 'width': 2},
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Diagonal (chance line)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Chance (AUC=0.5)',
        line={'color': 'gray', 'width': 1, 'dash': 'dash'},
        hoverinfo='skip'
    ))
    
    # Optimal point (Youden J)
    fig.add_trace(go.Scatter(
        x=[fpr[best_idx]],
        y=[tpr[best_idx]],
        mode='markers',
        name=f'Optimal (Sens={tpr[best_idx]:.3f}, Spec={1-fpr[best_idx]:.3f})',
        marker={'size': 10, 'color': 'red'},
        hovertemplate='Sensitivity: %{y:.3f}<br>Specificity: %{customdata:.3f}<extra></extra>',
        customdata=[1 - fpr[best_idx]],
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': f'ROC Curve<br><sub>AUC = {auc_val:.4f} (95% CI: {stats_res["95% CI Lower"]:.4f}-{stats_res["95% CI Upper"]:.4f})</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='1 - Specificity (False Positive Rate)',
        yaxis_title='Sensitivity (True Positive Rate)',
        hovermode='closest',
        template='plotly_white',
        width=700,
        height=600,
        font={'size': 12}
    )
    
    # Set axis ranges
    fig.update_xaxes(range=[-0.05, 1.05])
    fig.update_yaxes(range=[-0.05, 1.05])
    
    coords_df = pd.DataFrame({
        'Threshold': thresholds,
        'Sensitivity': tpr,
        'Specificity': 1-fpr,
        'Youden J': tpr - fpr
    }).round(4)
    
    return stats_res, None, fig, coords_df


# ========================================
# 5. ICC (Intraclass Correlation)
# ========================================

@st.cache_data(show_spinner=False)
def calculate_icc(df, cols):
    """
    คำนวณ ICC(2,1) และ ICC(3,1) โดยใช้ Two-way ANOVA Formula
    """
    if len(cols) < 2:
        return None, "Please select at least 2 variables (raters/methods).", None
    
    data = df[cols].dropna()
    n, k = data.shape  # n=subjects, k=raters
    
    if n < 2:
        return None, "Insufficient data (need at least 2 rows).", None
    
    if k < 2:
        return None, "Insufficient raters (need at least 2 columns).", None
    
    # ANOVA Calculations
    grand_mean = data.values.mean()
    
    # Sum of Squares
    SStotal = ((data.values - grand_mean)**2).sum()
    
    # Between-subjects (Rows)
    row_means = data.mean(axis=1)
    SSrow = k * ((row_means - grand_mean)**2).sum()
    
    # Between-raters (Cols)
    col_means = data.mean(axis=0)
    SScol = n * ((col_means - grand_mean)**2).sum()
    
    # Residual (Error)
    SSres = SStotal - SSrow - SScol
    
    # Degrees of Freedom
    df_row = n - 1
    df_col = k - 1
    df_res = df_row * df_col
    
    # Mean Squares
    MSrow = SSrow / df_row
    MScol = SScol / df_col
    MSres = SSres / df_res
    
    # Guard against zero denominators
    denom_icc3 = MSrow + (k - 1) * MSres
    denom_icc2 = MSrow + (k - 1) * MSres + (k / n) * (MScol - MSres)
    
    if denom_icc3 == 0 or denom_icc2 == 0:
        return None, "Insufficient variance to compute ICC (denominator = 0).", None
    
    # Calculate ICCs
    icc3_1 = (MSrow - MSres) / denom_icc3
    icc2_1 = (MSrow - MSres) / denom_icc2
    
    def interpret_icc(v):
        if not np.isfinite(v):
            return "Undefined"
        if v < 0.5:
            return "Poor"
        elif v < 0.75:
            return "Moderate"
        elif v < 0.9:
            return "Good"
        else:
            return "Excellent"
    
    res_df = pd.DataFrame({
        "Model": ["ICC(2,1) - Absolute Agreement", "ICC(3,1) - Consistency"],
        "Description": [
            "Use when raters are random & agreement matters (e.g. 2 different machines)",
            "Use when raters are fixed & consistency matters (e.g. ranking consistency)"
        ],
        "ICC Value": [f"{icc2_1:.4f}", f"{icc3_1:.4f}"],
        "Interpretation": [interpret_icc(icc2_1), interpret_icc(icc3_1)]
    })
    
    # ANOVA Table
    anova_df = pd.DataFrame({
        "Source": ["Between Subjects (Rows)", "Between Raters (Cols)", "Residual (Error)"],
        "SS": [f"{SSrow:.2f}", f"{SScol:.2f}", f"{SSres:.2f}"],
        "df": [df_row, df_col, df_res],
        "MS": [f"{MSrow:.2f}", f"{MScol:.2f}", f"{MSres:.2f}"]
    })
    
    return res_df, None, anova_df


# ========================================
# 6. REPORT GENERATION
# ========================================

def generate_report(title, elements):
    """Generate HTML report with professional color palette (synchronized with correlation.py)"""
    css_style = """ 
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: #2c3e50;
            line-height: 1.6;
        }
        h1 {
            color: #1a5276;
            border-bottom: 3px solid #2980b9;
            padding-bottom: 12px;
            font-size: 2em;
            margin-bottom: 20px;
        }
        h2 {
            color: #1a5276;
            margin-top: 25px;
            font-size: 1.35em;
            border-left: 5px solid #3498db;
            padding-left: 12px;
            margin-bottom: 15px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            border-radius: 6px;
            overflow: hidden;
        }
        table th, table td {
            border: 1px solid #ecf0f1;
            padding: 12px 15px;
            text-align: left;
        }
        table th {
            background-color: #34495e;
            color: white;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        table tr:hover {
            background-color: #f8f9fa;
        }
        table tr:nth-child(even) {
            background-color: #fcfcfc;
        }
        p {
            margin: 12px 0;
            color: #2c3e50;
        }
        .metric-text {
            font-size: 1.02em;
            margin: 10px 0;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }
        .metric-label {
            font-weight: 600;
            color: #34495e;
            min-width: 160px;
        }
        .metric-value {
            color: #2980b9;
            font-weight: 600;
            font-family: 'Courier New', monospace;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 4px;
            letter-spacing: 0.3px;
        }
        .interpretation {
            background: linear-gradient(135deg, #ecf0f1 0%, #f8f9fa 100%);
            border-left: 4px solid #3498db;
            padding: 14px 15px;
            margin: 16px 0;
            border-radius: 5px;
            line-height: 1.7;
            color: #2c3e50;
        }
        .interpretation::before {
            content: "ℹ️ ";
            margin-right: 8px;
        }
        .warning {
            background: linear-gradient(135deg, #fef5e7 0%, #f9f6f0 100%);
            border-left: 4px solid #f39c12;
            padding: 14px 15px;
            margin: 16px 0;
            border-radius: 5px;
            color: #7d6608;
            line-height: 1.7;
        }
        .report-table {
            border: 1px solid #ecf0f1;
        }
        .report-footer {
            text-align: center;
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 40px;
            border-top: 1px solid #ecf0f1;
            padding-top: 20px;
        }
        .report-footer a {
            color: #3498db;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .report-footer a:hover {
            color: #2980b9;
            text-decoration: underline;
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
            text_str = str(data)
            if ':' in text_str and len(text_str) < 150:
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
            is_stats_table = ('Statistic' in data.columns and 'Value' in data.columns 
                              and data.index.name is None)
            html_table = data.to_html(index=not is_stats_table, classes='report-table', escape=True)
            # Add section-header styling
            html_table = html_table.replace(
                '<td>RISK METRICS (Assumes: Rows=Exposure Status, Cols=Outcome Status)</td>',
                '<td style="background-color: #34495e; color: white; font-weight: bold;">RISK METRICS (Assumes: Rows=Exposure Status, Cols=Outcome Status)</td>'
            )
            html_table = html_table.replace(
                '<td>DIAGNOSTIC METRICS (Assumes: Rows=Test Result, Cols=Disease Status)</td>',
                '<td style="background-color: #34495e; color: white; font-weight: bold;">DIAGNOSTIC METRICS (Assumes: Rows=Test Result, Cols=Disease Status)</td>'
            )
            html += html_table
        
        elif element_type == 'contingency_table':
            col_labels = data.columns.tolist()
            row_labels = data.index.tolist()
            exp_name = data.index.name or "Exposure"
            out_name = element.get('outcome_col', 'Outcome')
            
            html += "<table class='report-table'>"
            html += "<thead>"
            html += f"<tr><th></th><th colspan='{len(col_labels)}'>{_html.escape(str(out_name))}</th></tr>"
            html += "<tr>"
            html += f"<th>{_html.escape(str(exp_name))}</th>"
            for col_label in col_labels:
                html += f"<th>{_html.escape(str(col_label))}</th>"
            html += "</tr>"
            html += "</thead>"
            
            html += "<tbody>"
            for idx_label in row_labels:
                html += "<tr>"
                html += f"<td>{_html.escape(str(idx_label))}</td>"
                for col_label in col_labels:
                    val = data.loc[idx_label, col_label]
                    html += f"<td>{_html.escape(str(val))}</td>" 
                html += "</tr>"
            html += "</tbody>"
            html += "</table>"
        
        elif element_type == 'plot':
            plot_obj = data
            
            if hasattr(plot_obj, 'to_html'):
                html += plot_obj.to_html(full_html=False, include_plotlyjs='cdn', div_id=f"plot_{id(plot_obj)}")
            else:
                buf = io.BytesIO()
                plot_obj.savefig(buf, format='png', bbox_inches='tight')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%; margin: 20px 0;" />'
    
    html += """<div class='report-footer'>
    &copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM n donate</a> | Powered by GitHub, Gemini, Streamlit
    </div>"""
    
    html += "</body>\n</html>"
    
    return html
