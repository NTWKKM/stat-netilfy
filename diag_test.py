# diag_test.py
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def calculate_descriptive(df, col):
    """คำนวณสถิติพื้นฐาน"""
    if col not in df.columns: return "Column not found"
    
    data = df[col].dropna()
    try:
        # พยายามแปลงเป็นตัวเลข
        num_data = pd.to_numeric(data, errors='raise')
        is_numeric = True
    except:
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
        # Categorical
        counts = data.value_counts()
        percent = data.value_counts(normalize=True) * 100
        return pd.DataFrame({
            "Category": counts.index,
            "Count": counts.values,
            "Percentage (%)": percent.values
        }).sort_values("Count", ascending=False)

def calculate_chi2(df, col1, col2):
    """คำนวณ Chi-square"""
    if col1 not in df.columns or col2 not in df.columns: return None, "Columns not found"
    
    # Drop missing
    data = df[[col1, col2]].dropna()
    
    # Contingency Table
    tab = pd.crosstab(data[col1], data[col2])
    
    try:
        chi2, p, dof, ex = stats.chi2_contingency(tab)
        msg = f"Chi-square statistic: {chi2:.4f}, p-value: {p:.4f}"
        return tab, msg
    except Exception as e:
        return tab, str(e)

# --- ROC & AUC FUNCTIONS ---

def auc_ci_hanley_mcneil(auc, n1, n2):
    """
    Hanley & McNeil (1982) method for AUC Variance (Parametric/Binomial assumption)
    n1: positive cases, n2: negative cases
    """
    q1 = auc / (2 - auc)
    q2 = 2 * (auc**2) / (1 + auc)
    
    se_auc = np.sqrt(((auc * (1 - auc)) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1 * n2))
    lower = auc - 1.96 * se_auc
    upper = auc + 1.96 * se_auc
    return lower, upper, se_auc

def auc_ci_delong(y_true, y_scores):
    """
    DeLong et al. (1988) method for AUC Variance (Non-parametric)
    Ref: Fast implementation logic
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Sort by score
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    n_pos = tps[-1]
    n_neg = fps[-1]
    
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan # Cannot calc
    
    auc = roc_auc_score(y_true, y_scores)
    
    # DeLong Covariance Calculation
    # Compute V10 (X) and V01 (Y)
    # Using simple loop for clarity (optimized versions exist but this is fine for N < 10k)
    
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    
    # Mann-Whitney statistic elements
    # V10[i] = mean( I(pos_score[i] > neg_scores) + 0.5 * I(pos_score[i] == neg_scores) )
    
    def compute_mid_rank(x):
        """Helper to get mid-ranks"""
        argsort = np.argsort(x)
        ranks = np.empty_like(argsort)
        ranks[argsort] = np.arange(len(x))
        return (ranks + 1) # dummy, actual logic below
        
    # Faster vectorization for V10, V01
    # Concatenate all scores to rank them
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    
    # Rank (with average for ties)
    order = np.argsort(all_scores)
    ranks = stats.rankdata(all_scores) # 1-based, average ties
    
    # Sum of ranks for positives
    # AUC = (Sum(R_pos) - n_pos(n_pos+1)/2 ) / (n_pos * n_neg)
    
    # DeLong variance components
    # We need empirical Probabilities
    # This is complex to code from scratch without bug.
    # Alternative: Use simpler bootstrap if DeLong is too heavy, 
    # BUT user asked for DeLong. Let's use a known robust snippet logic.
    
    # V10: For each positive, what fraction of negatives is it greater than?
    v10 = []
    for p in pos_scores:
        v10.append( (np.sum(p > neg_scores) + 0.5*np.sum(p == neg_scores)) / n_neg )
    v10 = np.array(v10)
    
    # V01: For each negative, what fraction of positives is it smaller than?
    v01 = []
    for n in neg_scores:
        v01.append( (np.sum(pos_scores > n) + 0.5*np.sum(pos_scores == n)) / n_pos )
    v01 = np.array(v01)
    
    # Variance
    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    
    var_auc = (s10 / n_pos) + (s01 / n_neg)
    se_auc = np.sqrt(var_auc)
    
    return auc - 1.96*se_auc, auc + 1.96*se_auc, se_auc


def analyze_roc(df, truth_col, score_col, method='delong'):
    """Main ROC Analysis"""
    data = df[[truth_col, score_col]].dropna()
    y_true = data[truth_col]
    y_score = pd.to_numeric(data[score_col], errors='coerce').dropna()
    # Align indices
    y_true = y_true.loc[y_score.index]
    
    if y_true.nunique() < 2:
        return None, "Outcome must have 2 classes (0 and 1)", None
        
    # 1. Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    
    n1 = sum(y_true == 1)
    n0 = sum(y_true == 0)
    
    # 2. Calculate CI
    if method == 'delong':
        ci_lower, ci_upper, se = auc_ci_delong(y_true.values, y_score.values)
        method_name = "DeLong et al."
    else:
        # Binomial Exact / Hanley McNeil
        ci_lower, ci_upper, se = auc_ci_hanley_mcneil(auc_val, n1, n0)
        method_name = "Hanley & McNeil (Parametric/Binomial)"
        
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)
    
    # 3. Youden Index
    # J = Sensitivity + Specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    youden_j = j_scores[best_idx]
    best_thresh = thresholds[best_idx]
    best_sens = tpr[best_idx]
    best_spec = 1 - fpr[best_idx]
    
    stats_res = {
        "AUC": auc_val,
        "SE": se,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "Method": method_name,
        "Youden Index (J)": youden_j,
        "Best Cut-off": best_thresh,
        "Sensitivity": best_sens,
        "Specificity": best_spec,
        "N (Positive)": n1,
        "N (Negative)": n0
    }
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark Youden point
    ax.plot(1-best_spec, best_sens, 'ro', label=f'Best Cut-off ({best_thresh:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(f'ROC Curve: {score_col} vs {truth_col}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    return stats_res, None, fig
