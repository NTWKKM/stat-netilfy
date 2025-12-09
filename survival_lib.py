import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import streamlit as st  # ✅ IMPORT STREAMLIT

# ✅ ADD CACHE: คำนวณ KM และ Logrank หนักๆ ให้จำค่าไว้
@st.cache_data(show_spinner=False)
def fit_km_logrank(df, time_col, event_col, group_col):
    """
    Fits Kaplan-Meier curves and performs Log-rank test.
    Returns the figure and a statistics dataframe.
    """
    try:
        T = df[time_col].astype(float)
        E = df[event_col].astype(float)
        groups = df[group_col].astype(str)
        
        kmf = KaplanMeierFitter()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        results_list = []
        
        # Fit Loop
        for name in sorted(groups.unique()):
            mask = (groups == name)
            kmf.fit(T[mask], event_observed=E[mask], label=str(name))
            kmf.plot_survival_function(ax=ax, ci_show=False)
            
            # Median Survival
            median_surv = kmf.median_survival_time_
            results_list.append({
                "Group": name,
                "N": mask.sum(),
                "Events": E[mask].sum(),
                "Median Survival": median_surv
            })
            
        # Log-rank Test
        results = logrank_test(T, groups, event_observed=E)
        p_value = results.p_value
        
        ax.set_title(f"Kaplan-Meier Survival Curve\nLog-rank p-value = {p_value:.4f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.grid(alpha=0.3)
        
        stats_df = pd.DataFrame(results_list)
        stats_df['Log-rank P'] = p_value
        
        return fig, stats_df
        
    except Exception as e:
        return None, str(e)

# ✅ ADD CACHE: Cox Regression ก็หนัก ควรจำไว้
@st.cache_data(show_spinner=False)
def fit_cox_multivariate(df, time_col, event_col, features):
    try:
        data = df[[time_col, event_col] + features].dropna()
        cph = CoxPHFitter()
        cph.fit(data, duration_col=time_col, event_col=event_col)
        
        summary = cph.summary.reset_index()
        summary = summary.rename(columns={'index': 'Variable'})
        
        return summary, cph
    except Exception as e:
        return None, str(e)
