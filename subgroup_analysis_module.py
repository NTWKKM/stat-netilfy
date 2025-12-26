"""
Professional Subgroup Analysis Module

Comprehensive subgroup analysis for publication-grade statistical reporting.
Supports Logistic Regression (binary outcomes) and Cox Regression (survival data).

Standard: ICMJE, CONSORT, and Cochrane guidelines
Author: NTWKKM (Adapted by Gemini)
Version: 1.4 (Robust Type Checking Fix)
License: MIT
"""

import pandas as pd
import numpy as np
import streamlit as st
from logger import get_logger
from forest_plot_lib import create_forest_plot
import warnings
from contextlib import contextmanager

@contextmanager
def suppress_convergence_warnings():
    """
    Context manager that suppresses convergence-related warnings during model fitting.
    
    While active, ignores RuntimeWarning and any warnings whose message contains the substring "convergence", allowing model-fitting code to run without emitting those warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*convergence.*')
        yield

logger = get_logger(__name__)


class SubgroupAnalysisLogit:
    """
    Professional Subgroup Analysis for Logistic Regression
    
    Publication-ready implementation with:
    - Interaction testing (Likelihood Ratio Test)
    - Multiple effect measures (OR, log-OR, SE)
    - Comprehensive statistics
    - Publication-grade forest plots
    - Sensitivity analyses
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Set up the SubgroupAnalysisLogit instance with a copy of the provided dataset.
        
        Copies the input DataFrame and initializes internal containers for results, summary statistics,
        interaction test results, and plotting figure storage.
        
        Parameters:
            df (pd.DataFrame): Input dataset to analyze; a shallow copy is stored internally.
        """
        self.df = df.copy()
        self.results = None
        self.stats = None
        self.interaction_result = None
        self.figure = None
        logger.info(f"SubgroupAnalysisLogit initialized with {len(df)} observations")
    
    def validate_inputs(
        self,
        outcome_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list = None
    ) -> bool:
        """
        Validate that required columns exist and meet basic requirements for subgroup analysis.
        
        Parameters:
            outcome_col (str): Name of the outcome column; must contain exactly two unique values.
            treatment_col (str): Name of the treatment/exposure column.
            subgroup_col (str): Name of the subgroup column; must contain two or more categories.
            adjustment_cols (list, optional): Additional column names required for adjusted models.
        
        Returns:
            bool: True if all validations pass.
        
        Raises:
            ValueError: If any required column is missing, the outcome is not binary, the subgroup has fewer than two categories, or the dataframe has fewer than 10 observations.
        """
        # Check columns exist
        required_cols = {outcome_col, treatment_col, subgroup_col}
        if adjustment_cols:
            required_cols.update(adjustment_cols)
        
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check outcome is binary
        if self.df[outcome_col].nunique() != 2:
            raise ValueError(f"Outcome '{outcome_col}' must be binary (2 unique values)")
        
        # Check subgroup has 2+ categories
        if self.df[subgroup_col].nunique() < 2:
            raise ValueError(f"Subgroup '{subgroup_col}' must have ≥2 categories")
        
        # Check sample size
        if len(self.df) < 10:
            raise ValueError(f"Minimum 10 observations required, got {len(self.df)}")
        
        logger.info("All input validations passed")
        return True
    
    def analyze(
        self,
        outcome_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list = None,
        min_subgroup_n: int = 5,
    ) -> dict:
        """
        Perform a complete subgroup analysis for a binary outcome, including overall and stratum-specific logistic models and an interaction test.
        
        Parameters:
            outcome_col (str): Column name of the binary outcome (encoded as 0/1 or two distinct values).
            treatment_col (str): Column name of the binary treatment/exposure (will be auto-encoded to 0/1 if necessary).
            subgroup_col (str): Column name of the categorical stratification variable.
            adjustment_cols (list, optional): List of covariate column names to include as adjustment terms; defaults to None.
            min_subgroup_n (int): Minimum number of observations required to fit a subgroup model; subgroups with fewer observations are skipped.
        
        Returns:
            dict: Structured results containing:
                - overall: overall model estimate (odds ratio), confidence interval, p-value, n, and events;
                - subgroups: list of subgroup result records (odds ratio, CI, p-value, n, events, etc.);
                - interaction: interaction test results including `p_value` and `significant` flag;
                - summary: computed summary statistics (counts, ranges, heterogeneity indicator);
                - results_df: DataFrame of raw result rows used to build the output.
        """
        try:
            from statsmodels.formula.api import logit
            from scipy import stats # For Chi2 test (LRT)
            
            # Validate inputs
            self.validate_inputs(outcome_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            # Clean data
            cols_to_use = [outcome_col, treatment_col, subgroup_col] + adjustment_cols
            df_clean = self.df[cols_to_use].dropna().copy()
            
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data after removing NaN: {len(df_clean)} rows")
            
            # 🟢 NEW: Auto-encode Treatment to 0/1 if it's categorical
            # This prevents KeyError when accessing model.params[treatment_col]
            if not pd.api.types.is_numeric_dtype(df_clean[treatment_col]) or df_clean[treatment_col].nunique() == 2:
                 unique_treats = sorted(df_clean[treatment_col].unique())
                 if len(unique_treats) == 2:
                     ref_val, risk_val = unique_treats[0], unique_treats[1]
                     # Check if it needs mapping (not already 0/1)
                     if not (ref_val == 0 and risk_val == 1):
                         df_clean[treatment_col] = df_clean[treatment_col].map({ref_val: 0, risk_val: 1})
                         st.info(f"ℹ️ **Treatment Encoding:** {ref_val} (Ref) vs {risk_val} (Risk)")
                 elif len(unique_treats) > 2:
                     if not pd.api.types.is_numeric_dtype(df_clean[treatment_col]):
                        raise ValueError(f"Treatment variable '{treatment_col}' has >2 levels. Subgroup analysis requires binary treatment.")

            # Build formula
            formula_base = f'{outcome_col} ~ {treatment_col}'
            if adjustment_cols:
                formula_base += ' + ' + ' + '.join(adjustment_cols)
            
            results_list = []
            
            # === OVERALL ANALYSIS ===
            st.info("📊 Computing Overall Model...")
            try:
                model_overall = logit(formula_base, data=df_clean).fit(disp=0)
                
                # ✅ Robust Parameter Access
                if treatment_col not in model_overall.params:
                    # Fallback for formula-renamed variables (e.g. if column starts with number)
                    matched_col = [c for c in model_overall.params.index if treatment_col in c]
                    target_var = matched_col[0] if matched_col else treatment_col
                else:
                    target_var = treatment_col

                or_overall = np.exp(model_overall.params[target_var])
                se_logit = model_overall.bse[target_var]
                ci_overall = np.exp(model_overall.conf_int().loc[target_var])
                p_overall = model_overall.pvalues[target_var]
                z_overall = model_overall.tvalues[target_var]
                
                results_list.append({
                    'group': f'Overall (N={len(df_clean)})',
                    'n': len(df_clean),
                    'events': int(df_clean[outcome_col].sum()),
                    'or': or_overall,
                    'log_or': model_overall.params[target_var],
                    'se': se_logit,
                    'ci_low': ci_overall[0],
                    'ci_high': ci_overall[1],
                    'z_stat': z_overall,
                    'p_value': p_overall,
                    'type': 'overall'
                })
                
                logger.info(f"Overall: OR={or_overall:.3f}, P={p_overall:.4f}")
            except Exception as e:
                st.error(f"❌ Overall model error: {e}")
                logger.exception("Overall model fitting failed")
                raise
            
            # === SUBGROUP ANALYSES ===
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            
            st.info(f"📊 Computing {len(subgroups)} Subgroup Models...")
            
            for _i, subgroup_val in enumerate(subgroups, 1):
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
                
                # Check N and Treatment variation
                if len(df_sub) < min_subgroup_n:
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: N={len(df_sub)} < {min_subgroup_n}, skipping")
                    continue
                
                if df_sub[treatment_col].nunique() < 2:
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: No variation in treatment (all same value), skipping")
                    continue
                
                if df_sub[outcome_col].nunique() < 2:
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: No variation in outcome (all 0 or all 1), skipping")
                    continue

                try:
                    model_sub = logit(formula_base, data=df_sub).fit(disp=0)
                    
                    # ✅ Robust Parameter Access
                    if treatment_col not in model_sub.params:
                        matched_col = [c for c in model_sub.params.index if treatment_col in c]
                        target_var = matched_col[0] if matched_col else treatment_col
                    else:
                        target_var = treatment_col

                    or_sub = np.exp(model_sub.params[target_var])
                    se_logit_sub = model_sub.bse[target_var]
                    ci_sub = np.exp(model_sub.conf_int().loc[target_var])
                    p_sub = model_sub.pvalues[target_var]
                    z_sub = model_sub.tvalues[target_var]
                    
                    results_list.append({
                        'group': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                        'subgroup': subgroup_val,
                        'n': len(df_sub),
                        'events': int(df_sub[outcome_col].sum()),
                        'or': or_sub,
                        'log_or': model_sub.params[target_var],
                        'se': se_logit_sub,
                        'ci_low': ci_sub[0],
                        'ci_high': ci_sub[1],
                        'z_stat': z_sub,
                        'p_value': p_sub,
                        'type': 'subgroup'
                    })
                    
                    logger.info(f"Subgroup {subgroup_val}: OR={or_sub:.3f}, P={p_sub:.4f}")
                
                except Exception as e:
                    st.warning(f"  ⚠️ Model failed for {subgroup_val}: {e}")
                    logger.warning(f"Subgroup {subgroup_val} model failed: {e}")
                    continue
            
            # === INTERACTION TEST (Likelihood Ratio Test - LRT) ===
            st.info("📊 Computing Interaction Test (LRT)...")
            try:
                # Model WITHOUT interaction (Reduced)
                formula_reduced = f'{outcome_col} ~ {treatment_col} + C({subgroup_col})'
                if adjustment_cols:
                    formula_reduced += ' + ' + ' + '.join(adjustment_cols)
                
                # Model WITH interaction (Full)
                formula_full = f'{outcome_col} ~ {treatment_col} * C({subgroup_col})'
                if adjustment_cols:
                    formula_full += ' + ' + ' + '.join(adjustment_cols)
                
                model_reduced = logit(formula_reduced, data=df_clean).fit(disp=0)
                model_full = logit(formula_full, data=df_clean).fit(disp=0)
                
                # Calculate LRT
                lr_stat = -2 * (model_reduced.llf - model_full.llf)
                df_diff = model_full.df_model - model_reduced.df_model
                
                if df_diff > 0:
                    p_interaction = stats.chi2.sf(lr_stat, df_diff)
                else:
                    p_interaction = np.nan
                
                # ✅ FIX: Type Check for significance comparison
                is_sig = False
                if isinstance(p_interaction, (int, float)) and pd.notna(p_interaction):
                    is_sig = p_interaction < 0.05

                self.interaction_result = {
                    'p_value': p_interaction,
                    'coefficient': None,
                    'se': None,
                    'significant': is_sig
                }
                
                logger.info(f"Interaction (LRT) P={p_interaction:.4f}")
            
            except Exception as e:
                st.warning(f"⚠️ Interaction test failed: {e}")
                logger.warning(f"Interaction test error: {e}")
                self.interaction_result = {
                    'p_value': np.nan,
                    'coefficient': np.nan,
                    'se': np.nan,
                    'significant': False
                }
            
            # Store results
            self.results = pd.DataFrame(results_list)
            self.stats = self._compute_summary_statistics()
            
            st.success("✅ Analysis complete!")
            return self._format_output()
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {e!s}")
            logger.exception("Subgroup analysis error")
            raise
    
    def _compute_summary_statistics(self) -> dict:
        """
        Compute high-level summary statistics from the stored results DataFrame.
        
        Returns:
            summary (dict): Aggregated statistics with keys:
                - n_overall (int): Total sample size for the overall analysis.
                - events_overall (int): Number of events in the overall analysis.
                - n_subgroups (int): Count of subgroup rows included in results.
                - or_overall (float): Overall odds ratio (OR) for the treatment.
                - ci_overall (tuple[float, float]): 95% confidence interval for the overall OR as (low, high).
                - p_overall (float or NaN): P-value for the overall treatment effect.
                - p_interaction (float or NaN): P-value from the interaction (heterogeneity) test.
                - heterogeneous (bool): True if the interaction test indicates significant heterogeneity.
                - or_range (tuple[float, float]): Minimum and maximum subgroup ORs as (min, max); (0, 0) if no subgroups.
                - or_range_ratio (float): Ratio of max OR to min OR across subgroups; np.inf if undefined (no subgroups or min <= 0).
        """
        if self.results is None or self.results.empty:
            return {}

        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        # Heterogeneity assessment
        het_significant = self.interaction_result.get('significant', False)
        
        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'or_overall': overall['or'],
            'ci_overall': (overall['ci_low'], overall['ci_high']),
            'p_overall': overall['p_value'],
            'p_interaction': self.interaction_result['p_value'],
            'heterogeneous': het_significant,
            'or_range': (subgroups['or'].min(), subgroups['or'].max()) if not subgroups.empty else (0,0),
            'or_range_ratio': (subgroups['or'].max() / subgroups['or'].min()) if not subgroups.empty and subgroups['or'].min() > 0 else np.inf
        }
    
    def _format_output(self) -> dict:
        """
        Prepare a JSON-serializable summary dictionary of analysis results for reporting.
        
        Returns an empty dict if no results are available or no overall row exists.
        
        Returns:
            output (dict): A dictionary with the following keys:
                overall (dict): Overall effect estimate with keys:
                    - or (float): Odds ratio for the overall population.
                    - ci (tuple[float, float]): Lower and upper 95% confidence interval.
                    - p_value (float): P-value for the overall effect (may be NaN if model produced NaN).
                    - n (int): Sample size used for the overall estimate.
                    - events (int): Number of outcome events for the overall estimate.
                subgroups (list[dict]): List of subgroup result records (one dict per subgroup) taken from the results DataFrame.
                interaction (dict):
                    - p_value (float|None): Interaction p-value, or None if not numeric/NaN.
                    - significant (bool): Whether the interaction was flagged as significant.
                summary (dict): Summary statistics computed by _compute_summary_statistics.
                results_df (pandas.DataFrame): Original results DataFrame (kept for downstream use).
        """
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results['type'] == 'overall']
        if overall_rows.empty:
            return {}
            
        overall = overall_rows.iloc[0]
        
        # ✅ FIX: Handle NaN P-value for JSON serialization
        p_int = self.interaction_result['p_value']
        p_int_val = float(p_int) if isinstance(p_int, (int, float)) and pd.notna(p_int) else None

        return {
            'overall': {
                'or': float(overall['or']),
                'ci': (float(overall['ci_low']), float(overall['ci_high'])),
                'p_value': float(overall['p_value']),
                'n': int(overall['n']),
                'events': int(overall['events'])
            },
            'subgroups': self.results[self.results['type'] == 'subgroup'].to_dict('records'),
            'interaction': {
                'p_value': p_int_val,
                'significant': bool(self.interaction_result['significant'])
            },
            'summary': self.stats,
            'results_df': self.results
        }
    
    def create_forest_plot(
        self,
        title: str = "Subgroup Analysis: Logistic Regression",
        color: str = "#2180BE"
    ):
        """
        Builds and stores a publication-grade forest plot of subgroup odds ratios.
        
        This requires that analyze() has been run and self.results populated; the plot includes subgroup estimates (OR and 95% CI) and appends the interaction p-value and heterogeneity cue to the title when available.
        
        Parameters:
            title (str): Plot title to display above the figure.
            color (str): Hex color used for the plot elements.
        
        Returns:
            figure: The generated plot figure object.
        
        Raises:
            ValueError: If analyze() has not been run and self.results is None.
        """
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        # Prepare data
        plot_data = self.results[['group', 'or', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'or', 'ci_low', 'ci_high', 'p_value']
        
        # Add interaction info to title
        p_int = self.interaction_result['p_value']
        # ✅ FIX: Robust type check
        is_het = self.interaction_result.get('significant', False)
        het_text = "Heterogeneous ⚠️" if is_het else "Homogeneous ✓"
        
        if isinstance(p_int, (int, float)) and pd.notna(p_int):
            title_final = f"{title}<br><span style='font-size: 12px; color: #666;'>P for Interaction = {p_int:.4f} ({het_text})</span>"
        else:
            title_final = title
        
        # Create forest plot
        self.figure = create_forest_plot(
            data=plot_data,
            estimate_col='or',
            ci_low_col='ci_low',
            ci_high_col='ci_high',
            label_col='variable',
            pval_col='p_value',
            title=title_final,
            x_label='Odds Ratio (95% CI)',
            ref_line=1.0,
            color=color,
            height=max(400, len(plot_data) * 60 + 150)
        )
        return self.figure
    
    def get_interpretation(self) -> str:
        """
        Produce a concise clinical interpretation based on the stored interaction test result.
        
        Returns one of three short, user-facing strings:
        - If the interaction p-value is missing or non-numeric, a question-mark message indicating the interaction test could not be performed.
        - If the interaction is statistically significant, a warning-style message noting significant heterogeneity with the formatted p-value and recommending separate subgroup reporting and discussion of mechanisms.
        - If not significant, a confirmation-style message noting no significant heterogeneity with the formatted p-value and that the overall effect is appropriate across subgroups.
        
        Returns:
            str: A formatted interpretation message (includes an emoji and the interaction p-value when available).
        """
        p_int = self.interaction_result['p_value']
        het = self.interaction_result.get('significant', False)
        
        if not isinstance(p_int, (int, float)) or pd.isna(p_int):
            return "❓ Interaction test could not be performed."
        
        if het:
            return (
                f"⚠️ **Significant heterogeneity detected** (P = {p_int:.4f})\n\n"
                f"The treatment effect varies significantly across subgroups. "
                f"Consider reporting results separately for each subgroup and discuss "
                f"possible mechanisms for differential treatment response."
            )
        else:
            return (
                f"✅ **No significant heterogeneity** (P = {p_int:.4f})\n\n"
                f"The treatment effect is consistent across all subgroups. "
                f"The overall effect estimate is appropriate for all population segments."
            )


class SubgroupAnalysisCox:
    """
    Professional Subgroup Analysis for Cox Regression
    
    Publication-ready implementation for survival analysis with:
    - Interaction testing (Likelihood Ratio Test)
    - Multiple effect measures (HR, log-HR, SE)
    - Comprehensive statistics
    - Publication-grade forest plots
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Create a SubgroupAnalysisCox instance bound to the provided dataset.
        
        Parameters:
            df (pd.DataFrame): Input dataset containing the columns required for Cox subgroup analysis
                (e.g., time, event, treatment, subgroup, and any adjustment covariates). The dataframe
                is copied and stored internally.
        
        Attributes set:
            df (pd.DataFrame): A copy of the provided dataframe.
            results: Placeholder for the analysis results DataFrame (None until analyze() is run).
            stats: Placeholder for computed summary statistics (None until analyze() is run).
            interaction_result: Placeholder for interaction test results (None until analyze() is run).
            figure: Placeholder for a generated forest plot (None until create_forest_plot() is run).
        """
        self.df = df.copy()
        self.results = None
        self.stats = None
        self.interaction_result = None
        self.figure = None
        logger.info(f"SubgroupAnalysisCox initialized with {len(df)} observations")
    
    def validate_inputs(
        self,
        time_col: str,
        event_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list = None
    ) -> bool:
        """
        Validate that the DataFrame contains the required columns and that those columns meet Cox model input requirements.
        
        Checks performed:
        - All required columns (time_col, event_col, treatment_col, subgroup_col, plus any adjustment_cols) are present.
        - time_col is numeric and all values are greater than zero.
        - event_col has exactly two distinct values.
        - subgroup_col has at least two distinct categories.
        
        Parameters:
            time_col (str): Name of the follow-up time/duration column.
            event_col (str): Name of the binary event indicator column.
            treatment_col (str): Name of the treatment/exposure column.
            subgroup_col (str): Name of the subgroup/stratification column.
            adjustment_cols (list, optional): Additional covariate column names required for the model.
        
        Returns:
            bool: `True` if all validations pass.
        
        Raises:
            ValueError: If any required column is missing, if time_col is non-numeric or contains non-positive values, if event_col is not binary, or if subgroup_col has fewer than two categories.
        """
        required_cols = {time_col, event_col, treatment_col, subgroup_col}
        if adjustment_cols:
            required_cols.update(adjustment_cols)
        
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check time is numeric and positive
        if not pd.api.types.is_numeric_dtype(self.df[time_col]):
            raise ValueError(f"Time column '{time_col}' must be numeric")
        if (self.df[time_col] <= 0).any():
            raise ValueError(f"Time column '{time_col}' must have positive values")
        
        # Check event is binary
        if self.df[event_col].nunique() != 2:
            raise ValueError(f"Event column '{event_col}' must be binary")
        
        # Check subgroup has 2+ categories
        if self.df[subgroup_col].nunique() < 2:
            raise ValueError(f"Subgroup '{subgroup_col}' must have ≥2 categories")
        
        logger.info("All Cox input validations passed")
        return True
    
    def analyze(
        self,
        time_col: str,
        event_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list = None,
        min_subgroup_n: int = 5,
        min_events: int = 2
    ) -> dict:
        """
        Perform a complete subgroup analysis using Cox proportional hazards models, producing overall and stratum-specific hazard ratios, an interaction test, and formatted results for reporting.
        
        Parameters:
            time_col (str): Column name for follow-up time (duration).
            event_col (str): Column name for the event indicator (binary).
            treatment_col (str): Column name for the treatment/exposure variable (auto-encoded to 0/1 if binary non-numeric).
            subgroup_col (str): Column name for the stratification variable used to define subgroups.
            adjustment_cols (list, optional): List of covariate column names to adjust for in the models. Defaults to None.
            min_subgroup_n (int, optional): Minimum number of observations required in a subgroup to attempt model fitting. Defaults to 5.
            min_events (int, optional): Minimum number of events required in a subgroup to attempt model fitting. Defaults to 2.
        
        Returns:
            dict: A report dictionary containing:
                - overall: dict with overall hazard ratio (`hr`), confidence interval (`ci` tuple), `p_value`, `n`, and `events`.
                - subgroups: list of subgroup result dicts (each includes `group`, `subgroup`, `n`, `events`, `hr`, `ci_low`, `ci_high`, `p_value`, and related statistics).
                - interaction: dict with `p_value` from the likelihood-ratio test and `significant` boolean.
                - summary: computed summary statistics (counts, ranges, and heterogeneity indicator).
                - results_df: pandas DataFrame of the raw collected results suitable for further inspection or plotting.
        """
        try:
            from lifelines import CoxPHFitter
            from scipy import stats # For Chi2 test
            
            # Validate inputs
            self.validate_inputs(time_col, event_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            # Clean data
            cols_for_clean = [time_col, event_col, treatment_col, subgroup_col] + adjustment_cols
            df_clean = self.df[cols_for_clean].dropna().copy()
            
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")
            
            # 🟢 NEW: Auto-encode Treatment to 0/1 for Cox
            if not pd.api.types.is_numeric_dtype(df_clean[treatment_col]) or df_clean[treatment_col].nunique() == 2:
                 unique_treats = sorted(df_clean[treatment_col].unique())
                 if len(unique_treats) == 2:
                     ref_val, risk_val = unique_treats[0], unique_treats[1]
                     if not (ref_val == 0 and risk_val == 1):
                         df_clean[treatment_col] = df_clean[treatment_col].map({ref_val: 0, risk_val: 1})
                         st.info(f"ℹ️ **Treatment Encoding (Cox):** {ref_val} (Ref) vs {risk_val} (Risk)")
                 elif len(unique_treats) > 2:
                     if not pd.api.types.is_numeric_dtype(df_clean[treatment_col]):
                        raise ValueError(f"Treatment variable '{treatment_col}' has >2 levels. Subgroup analysis requires binary treatment.")
            
            results_list = []
            
            # === OVERALL ANALYSIS ===
            st.info("📊 Computing Overall Cox Model...")
            try:
                cph_overall = CoxPHFitter()
                cph_overall.fit(
                    df_clean[[time_col, event_col, treatment_col] + adjustment_cols],
                    duration_col=time_col,
                    event_col=event_col,
                    show_progress=False
                )
                
                hr_overall = np.exp(cph_overall.params_[treatment_col])
                se_loghr = cph_overall.standard_errors_[treatment_col]
                ci_overall = np.exp(cph_overall.confidence_intervals_.loc[treatment_col])
                p_overall = cph_overall.summary.loc[treatment_col, 'p']
                
                results_list.append({
                    'group': f'Overall (N={len(df_clean)})',
                    'n': len(df_clean),
                    'events': int(df_clean[event_col].sum()),
                    'hr': hr_overall,
                    'log_hr': cph_overall.params_[treatment_col],
                    'se': se_loghr,
                    'ci_low': ci_overall[0],
                    'ci_high': ci_overall[1],
                    'p_value': p_overall,
                    'type': 'overall'
                })
                
                logger.info(f"Overall Cox: HR={hr_overall:.3f}, P={p_overall:.4f}")
            except Exception as e:
                st.error(f"❌ Overall Cox model error: {e}")
                logger.exception("Overall Cox model fitting failed")
                raise
            
            # === SUBGROUP ANALYSES ===
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            st.info(f"📊 Computing {len(subgroups)} Subgroup Cox Models...")
            
            for _i, subgroup_val in enumerate(subgroups, 1):
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
                n_events = int(df_sub[event_col].sum())
                
                # Enhanced Stability Checks
                if len(df_sub) < min_subgroup_n or n_events < min_events:
                    msg = f"N={len(df_sub)}, events={n_events}"
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: {msg}, skipping")
                    continue
                
                if df_sub[treatment_col].nunique() < 2:
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: No variation in treatment (all same value), skipping")
                    continue
                
                try:
                    cph_sub = CoxPHFitter()
                    cph_sub.fit(
                        df_sub[[time_col, event_col, treatment_col] + adjustment_cols],
                        duration_col=time_col,
                        event_col=event_col,
                        show_progress=False
                    )
                    
                    hr_sub = np.exp(cph_sub.params_[treatment_col])
                    se_loghr_sub = cph_sub.standard_errors_[treatment_col]
                    ci_sub = np.exp(cph_sub.confidence_intervals_.loc[treatment_col])
                    p_sub = cph_sub.summary.loc[treatment_col, 'p']
                    
                    results_list.append({
                        'group': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                        'subgroup': subgroup_val,
                        'n': len(df_sub),
                        'events': n_events,
                        'hr': hr_sub,
                        'log_hr': cph_sub.params_[treatment_col],
                        'se': se_loghr_sub,
                        'ci_low': ci_sub[0],
                        'ci_high': ci_sub[1],
                        'p_value': p_sub,
                        'type': 'subgroup'
                    })
                    
                    logger.info(f"Subgroup {subgroup_val}: HR={hr_sub:.3f}, P={p_sub:.4f}")
                
                except Exception as e:
                    st.warning(f"  ⚠️ Cox model failed for {subgroup_val}: {e}")
                    logger.warning(f"Cox model failed for {subgroup_val}: {e}")
                    continue
            
            # === INTERACTION TEST (Likelihood Ratio Test) ===
            st.info("📊 Computing Interaction Test (LRT)...")
            try:
                base_formula_parts = [treatment_col, f"C({subgroup_col})"] + adjustment_cols
                formula_reduced = " + ".join(base_formula_parts)
                formula_full = f"{treatment_col} * C({subgroup_col})"
                if adjustment_cols:
                    formula_full += " + " + " + ".join(adjustment_cols)

                cph_reduced = CoxPHFitter()
                cph_reduced.fit(df_clean, duration_col=time_col, event_col=event_col, formula=formula_reduced, show_progress=False)
                
                cph_full = CoxPHFitter()
                cph_full.fit(df_clean, duration_col=time_col, event_col=event_col, formula=formula_full, show_progress=False)
                
                lr_stat = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
                df_diff = len(cph_full.params_) - len(cph_reduced.params_)
                
                if df_diff > 0:
                    p_interaction = stats.chi2.sf(lr_stat, df_diff)
                else:
                    p_interaction = np.nan
                
                # ✅ FIX: Robust type checking
                is_sig = False
                if isinstance(p_interaction, (int, float)) and pd.notna(p_interaction):
                    is_sig = p_interaction < 0.05

                self.interaction_result = {
                    'p_value': p_interaction,
                    'significant': is_sig
                }
                
                logger.info(f"Cox Interaction (LRT) P={p_interaction:.4f}")
            
            except Exception as e:
                st.warning(f"⚠️ Interaction test failed: {e}")
                logger.warning(f"Cox interaction test error: {e}")
                self.interaction_result = {
                    'p_value': np.nan,
                    'significant': False
                }
            
            # Store results
            self.results = pd.DataFrame(results_list)
            self.stats = self._compute_summary_statistics()
            
            st.success("✅ Cox analysis complete!")
            return self._format_output()
        
        except ImportError:
            st.error("❌ Lifelines library required. Install: pip install lifelines")
            raise
        except Exception as e:
            st.error(f"❌ Cox analysis failed: {e!s}")
            logger.exception("Cox subgroup analysis error")
            raise
    
    def _compute_summary_statistics(self) -> dict:
        """
        Assemble high-level summary statistics from self.results and the stored interaction result for reporting.
        
        Returns:
            dict: Mapping with the following keys:
                - n_overall (int): total sample size from the overall analysis row.
                - events_overall (int): number of events from the overall analysis row.
                - n_subgroups (int): count of subgroup rows included in results.
                - hr_overall (float): hazard ratio from the overall analysis row.
                - ci_overall (tuple[float, float]): lower and upper 95% confidence limits for the overall HR.
                - p_overall (float): p-value for the overall treatment effect.
                - p_interaction (float or NaN): p-value from the interaction (heterogeneity) test.
                - heterogeneous (bool): whether the interaction was flagged as statistically significant.
                - hr_range (tuple[float, float]): (min_hr, max_hr) across subgroup HRs; returns (0, 0) if no subgroup rows.
        """
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results['type'] == 'overall']
        if overall_rows.empty:
            return {}

        overall = overall_rows.iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'hr_overall': overall['hr'],
            'ci_overall': (overall['ci_low'], overall['ci_high']),
            'p_overall': overall['p_value'],
            'p_interaction': self.interaction_result['p_value'],
            'heterogeneous': self.interaction_result.get('significant', False),
            'hr_range': (subgroups['hr'].min(), subgroups['hr'].max()) if not subgroups.empty else (0,0)
        }
    
    def _format_output(self) -> dict:
        """
        Format the analysis results into a JSON-serializable report dictionary.
        
        Returns:
            report (dict): Structured output with keys:
                - overall (dict): overall estimate with keys:
                    - hr (float): hazard ratio for the overall cohort.
                    - ci (tuple[float, float]): (lower, upper) 95% confidence interval for the HR.
                    - p_value (float): p-value for the overall treatment effect.
                    - n (int): sample size used for the overall estimate.
                    - events (int): number of events for the overall estimate.
                - subgroups (list[dict]): list of subgroup result records (each row from the results table serialized to dict).
                - interaction (dict):
                    - p_value (float|None): interaction test p-value, or None if not numeric/available.
                    - significant (bool): flag indicating whether the interaction was marked significant.
                - summary (dict): precomputed summary statistics from _compute_summary_statistics.
                - results_df (pandas.DataFrame): the original results DataFrame.
        
            Returns an empty dict if no results are available.
        """
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results['type'] == 'overall']
        if overall_rows.empty:
            return {}
            
        overall = overall_rows.iloc[0]
        
        # ✅ FIX: Handle NaN for JSON serialization
        p_int = self.interaction_result['p_value']
        p_int_val = float(p_int) if isinstance(p_int, (int, float)) and pd.notna(p_int) else None

        return {
            'overall': {
                'hr': float(overall['hr']),
                'ci': (float(overall['ci_low']), float(overall['ci_high'])),
                'p_value': float(overall['p_value']),
                'n': int(overall['n']),
                'events': int(overall['events'])
            },
            'subgroups': self.results[self.results['type'] == 'subgroup'].to_dict('records'),
            'interaction': {
                'p_value': p_int_val,
                'significant': bool(self.interaction_result['significant'])
            },
            'summary': self.stats,
            'results_df': self.results
        }
    
    def create_forest_plot(
        self,
        title: str = "Subgroup Analysis: Cox Regression",
        color: str = "#2180BE"
    ):
        """
        Generate a publication-quality forest plot of subgroup Cox regression results.
        
        Parameters:
            title (str): Main plot title. If an interaction p-value is available, a formatted interaction line is appended to the title.
            color (str): Hex color used for plot elements (e.g., "#2180BE").
        
        Returns:
            figure: The figure object containing the forest plot (plot rendered by the plotting backend).
        
        Raises:
            ValueError: If analysis results are missing (run analyze() before calling this method).
        """
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        # Prepare data
        plot_data = self.results[['group', 'hr', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'hr', 'ci_low', 'ci_high', 'p_value']
        
        # Add interaction info
        p_int = self.interaction_result['p_value']
        is_het = self.interaction_result.get('significant', False)
        het_text = "Heterogeneous ⚠️" if is_het else "Homogeneous ✓"
        
        if isinstance(p_int, (int, float)) and pd.notna(p_int):
            title_final = f"{title}<br><span style='font-size: 12px; color: #666;'>P for Interaction = {p_int:.4f} ({het_text})</span>"
        else:
            title_final = title
        
        # Create forest plot
        self.figure = create_forest_plot(
            data=plot_data,
            estimate_col='hr',
            ci_low_col='ci_low',
            ci_high_col='ci_high',
            label_col='variable',
            pval_col='p_value',
            title=title_final,
            x_label='Hazard Ratio (95% CI)',
            ref_line=1.0,
            color=color,
            height=max(400, len(plot_data) * 60 + 150)
        )
        return self.figure
    
    def get_interpretation(self) -> str:
        """
        Produce a concise, human-readable interpretation of the interaction test result.
        
        Returns:
            interpretation (str): A short text summarizing the interaction test outcome. Possible forms:
                - A message indicating the interaction test could not be performed.
                - A warning stating significant heterogeneity with the formatted P-value.
                - A confirmation stating no significant heterogeneity with the formatted P-value.
        """
        p_int = self.interaction_result['p_value']
        het = self.interaction_result.get('significant', False)
        
        if not isinstance(p_int, (int, float)) or pd.isna(p_int):
            return "❓ Interaction test could not be performed."
        
        if het:
            return (
                f"⚠️ **Significant heterogeneity detected** (P = {p_int:.4f})\n\n"
                f"The treatment effect on survival varies significantly across subgroups. "
                f"Consider separately reporting and interpreting results for each stratum."
            )
        else:
            return (
                f"✅ **No significant heterogeneity** (P = {p_int:.4f})\n\n"
                f"The treatment effect on survival is consistent across all subgroups. "
                f"The overall hazard ratio applies uniformly to all population segments."
            )