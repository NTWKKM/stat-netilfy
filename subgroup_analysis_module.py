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
    
    Temporarily ignores RuntimeWarning and any warnings whose message matches 'convergence' so that model-fitting routines do not emit noisy convergence warnings while running inside the context.
    
    Yields:
        None
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
        Create a SubgroupAnalysisLogit instance backed by a copy of the provided dataframe.
        
        Stores a copy of `df` on the instance and initializes placeholders for analysis outputs (`results`, `stats`, `interaction_result`, `figure`). Logs the number of observations at initialization.
        
        Parameters:
            df (pd.DataFrame): Input dataset containing the variables to be used for subgroup analyses.
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
        Validate that the dataframe contains the required columns and basic data requirements for subgroup logistic analysis.
        
        Parameters:
            outcome_col (str): Column name for the binary outcome (must have exactly two unique values).
            treatment_col (str): Column name for the treatment/exposure variable.
            subgroup_col (str): Column name defining subgroup membership (must have at least two categories).
            adjustment_cols (list, optional): List of covariate column names to include; each must exist in the dataframe.
        
        Returns:
            bool: `True` if all validations pass.
        
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
        Perform subgroup logistic regression with subgroup-specific fits and an interaction test.
        
        Parameters:
            outcome_col (str): Binary outcome column name.
            treatment_col (str): Binary treatment/exposure column name (auto-encoded to 0/1 when appropriate).
            subgroup_col (str): Column name used to stratify the dataset into subgroups.
            adjustment_cols (list, optional): List of covariate column names to adjust for in all models. Defaults to None.
            min_subgroup_n (int): Minimum number of observations required to fit a subgroup model; subgroups with fewer rows are skipped. Defaults to 5.
        
        Returns:
            dict: A structured results dictionary containing:
                - overall: summary of the overall model (OR, CI, p-value, n, events).
                - subgroups: list of subgroup result entries (group label, n, events, OR, CI, p-value, etc.).
                - interaction_result: interaction test output with `p_value` and `significant` boolean.
                - summary: compact summary statistics derived from results.
                - results_df: the raw results table as a pandas-compatible structure.
        
        Raises:
            ValueError: If input validation fails or there are insufficient rows after cleaning.
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
                logger.error(f"Overall model fitting failed: {e}")
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
            st.error(f"❌ Analysis failed: {e}")
            logger.error(f"Subgroup analysis error: {e}")
            raise
    
    def _compute_summary_statistics(self) -> dict:
        """
        Summarizes key numeric results and heterogeneity indicators from the stored results table.
        
        Returns:
            summary (dict): A dictionary with the following keys:
                - n_overall (int): Total sample size used in the overall model.
                - events_overall (int): Number of events (outcomes) in the overall model.
                - n_subgroups (int): Count of subgroup rows included in the results.
                - or_overall (float): Overall effect estimate (odds ratio or hazard ratio as stored).
                - ci_overall (tuple[float, float]): Lower and upper bounds of the overall confidence interval.
                - p_overall (float): P-value for the overall treatment effect.
                - p_interaction (float): P-value from the interaction (likelihood ratio) test.
                - heterogeneous (bool): True if the interaction test indicates heterogeneity, False otherwise.
                - or_range (tuple[float, float]): Minimum and maximum subgroup effect estimates; (0, 0) if no subgroup rows.
                - or_range_ratio (float): Ratio of max to min subgroup effect estimates; returns infinity if min is zero or no valid subgroup estimates.
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
        Prepare a JSON-serializable summary of the analysis results including overall effect, subgroup rows, interaction test, and computed summary statistics.
        
        Returns:
            output (dict): Dictionary with keys:
                - overall: Dict with overall effect summary containing:
                    - 'or': overall odds ratio as a float.
                    - 'ci': tuple(float, float) with the lower and upper 95% confidence bounds.
                    - 'p_value': overall p-value as a float.
                    - 'n': total sample size as an int.
                    - 'events': number of outcome events as an int.
                - subgroups: List[dict] of subgroup result rows (one dict per subgroup) suitable for JSON serialization.
                - interaction: Dict with interaction test result containing:
                    - 'p_value': interaction p-value as a float, or None if not available.
                    - 'significant': boolean flag indicating whether the interaction is statistically significant.
                - summary: Precomputed summary statistics (self.stats) as-is.
                - results_df: Original results DataFrame (self.results) included for completeness.
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
        Generate a publication-grade forest plot of subgroup odds ratios with interaction annotation.
        
        Creates a forest plot from the analysis results stored on the instance and annotates the title with the interaction p-value and a heterogeneity indicator when available.
        
        Parameters:
            title (str): Plot title shown above the figure. If an interaction p-value is available, it is appended to the title.
            color (str): Hex color used for plot elements.
        
        Returns:
            figure: The generated plot/figure object returned by the plotting utility.
        
        Raises:
            ValueError: If analysis results are not available (run analyze() first).
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
        Return a concise clinical interpretation based on the stored interaction test result.
        
        If the interaction p-value is missing or invalid, returns a placeholder indicating the test could not be performed. If the interaction is significant, returns a heterogeneity-focused interpretation advising separate subgroup reporting; otherwise returns a conclusion that the treatment effect is consistent across subgroups.
        
        Returns:
            interpretation (str): A human-readable message summarizing the interaction test outcome and its clinical implication.
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
        Create a SubgroupAnalysisCox instance for performing Cox regression subgroup analyses.
        
        Parameters:
            df (pd.DataFrame): Input dataset; a copy is stored internally to avoid modifying the caller's DataFrame.
        
        Initializes:
            self.df (pd.DataFrame): Copied input data.
            self.results (Optional[pd.DataFrame]): Placeholder for analysis results, initialized to None.
            self.stats (Optional[dict]): Placeholder for summary statistics, initialized to None.
            self.interaction_result (Optional[dict]): Placeholder for interaction test result, initialized to None.
            self.figure (Optional[object]): Placeholder for a generated figure (forest plot), initialized to None.
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
        Validate that required columns exist and meet type/value requirements for Cox regression.
        
        Parameters:
            time_col (str): Name of the follow-up time column; must be numeric and contain values > 0.
            event_col (str): Name of the event indicator column; must be binary (exactly two unique values).
            treatment_col (str): Name of the treatment/exposure column expected to be present in the dataframe.
            subgroup_col (str): Name of the subgroup column; must have at least two distinct categories.
            adjustment_cols (list, optional): Additional column names required for adjustment; each must exist in the dataframe.
        
        Raises:
            ValueError: If any required column is missing.
            ValueError: If `time_col` is not numeric or contains non‑positive values.
            ValueError: If `event_col` does not have exactly two unique values.
            ValueError: If `subgroup_col` has fewer than two categories.
        
        Returns:
            bool: `True` when all validations pass.
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
        Perform a complete Cox proportional hazards subgroup analysis and return a structured results dictionary.
        
        Parameters:
            time_col (str): Name of the time-to-event column.
            event_col (str): Name of the binary event indicator column.
            treatment_col (str): Name of the binary treatment/exposure column (auto-encoded to 0/1 if necessary).
            subgroup_col (str): Name of the subgroup (stratification) column to iterate over.
            adjustment_cols (list, optional): List of covariate column names to adjust for. Defaults to None.
            min_subgroup_n (int, optional): Minimum number of observations required in a subgroup to run its model. Defaults to 5.
            min_events (int, optional): Minimum number of events required in a subgroup to run its model. Defaults to 2.
        
        Returns:
            dict: Formatted results produced by _format_output, containing:
                - overall effect (hazard ratio, CI, p-value, n, events),
                - list of subgroup results (HR, CI, p-value, n, events),
                - interaction test result (LRT p-value and significance flag),
                - summary statistics, and
                - the raw results dataframe.
        
        Side effects:
            Updates instance attributes self.results (DataFrame), self.stats (summary dict), and self.interaction_result (dict).
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
                raise
            
            # === SUBGROUP ANALYSES ===
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            st.info(f"📊 Computing {len(subgroups)} Subgroup Cox Models...")
            
            for i, subgroup_val in enumerate(subgroups, 1):
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
            st.error(f"❌ Cox analysis failed: {e}")
            logger.error(f"Cox subgroup analysis error: {e}")
            raise
    
    def _compute_summary_statistics(self) -> dict:
        """
        Produce a compact summary of overall and subgroup effect estimates and the interaction test result.
        
        Returns:
            summary (dict): A dictionary with the following keys:
                - n_overall (int): Total sample size used in the overall model.
                - events_overall (int): Number of events (outcomes) in the overall model.
                - n_subgroups (int): Number of subgroup rows included in results.
                - hr_overall (float): Overall hazard ratio (or effect estimate) from the overall model.
                - ci_overall (tuple[float, float]): 95% confidence interval for the overall effect (lower, upper).
                - p_overall (float): P-value for the overall treatment effect.
                - p_interaction (float): P-value from the interaction (heterogeneity) test.
                - heterogeneous (bool): True when the interaction test indicates significant heterogeneity, False otherwise.
                - hr_range (tuple[float, float]): Minimum and maximum subgroup effect estimates (hr) across subgroups; (min, max). 
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
        Prepare a JSON-serializable summary of analysis results.
        
        Returns:
            output (dict): Dictionary with keys:
                - overall: dict with overall effect estimate (`hr`), confidence interval tuple (`ci`),
                  `p_value`, sample size `n`, and `events`.
                - subgroups: list of subgroup result records (one dict per subgroup).
                - interaction: dict with `p_value` (float or `None` if not available) and
                  `significant` (bool).
                - summary: computed summary statistics dictionary.
                - results_df: the full results DataFrame (kept for downstream inspection).
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
        Generate a publication-quality forest plot of the stored Cox subgroup analysis results.
        
        Parameters:
            title (str): Plot title. If an interaction test result is available, the plot title will include the interaction p-value and a heterogeneity indicator.
            color (str): Hex color used for the plot accents (e.g., estimate markers and CI lines).
        
        Returns:
            figure: The created figure object (also stored on `self.figure`).
        
        Raises:
            ValueError: If analysis results are not available (run `analyze()` first).
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
        Return a concise clinical interpretation based on the stored interaction test result.
        
        If the interaction p-value is missing or invalid, returns a placeholder stating the interaction test could not be performed.
        If the interaction is significant, returns a heterogeneity warning that includes the p-value.
        If the interaction is not significant, returns a statement that the treatment effect is consistent across subgroups and includes the p-value.
        
        Returns:
            interpretation (str): A short, human-readable interpretation suitable for reports or UI.
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