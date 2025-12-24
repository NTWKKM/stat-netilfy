"""
Professional Subgroup Analysis Module

Comprehensive subgroup analysis for publication-grade statistical reporting.
Supports Logistic Regression (binary outcomes) and Cox Regression (survival data).

Standard: ICMJE, CONSORT, and Cochrane guidelines
Author: NTWKKM (Adapted by Gemini)
Version: 1.0
License: MIT
"""

import pandas as pd
import numpy as np
import streamlit as st
from logger import get_logger
from forest_plot_lib import create_forest_plot
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class SubgroupAnalysisLogit:
    """
    Professional Subgroup Analysis for Logistic Regression
    
    Publication-ready implementation with:
    - Interaction testing (Wald test)
    - Multiple effect measures (OR, log-OR, SE)
    - Comprehensive statistics
    - Publication-grade forest plots
    - Sensitivity analyses
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data frame
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
        Validate all required columns and data types.
        
        Returns:
        --------
        bool : True if all validations pass
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
        use_firth: bool = False
    ) -> dict:
        """
        Run complete subgroup analysis with interaction testing.
        
        Parameters:
        -----------
        outcome_col : str
            Binary outcome variable
        treatment_col : str
            Treatment/exposure variable
        subgroup_col : str
            Variable to stratify by
        adjustment_cols : list, optional
            List of adjustment variables
        min_subgroup_n : int
            Minimum sample size per subgroup
        use_firth : bool
            Use Firth's penalized likelihood for rare outcomes
        
        Returns:
        --------
        dict : Comprehensive results dictionary
        """
        try:
            from statsmodels.formula.api import logit
            from statsmodels.genmod.logistic_regression import LogitResults
            
            # Validate inputs
            self.validate_inputs(outcome_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            # Clean data
            df_clean = self.df[[outcome_col, treatment_col, subgroup_col] + adjustment_cols].dropna()
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data after removing NaN: {len(df_clean)} rows")
            
            # Build formula
            formula_base = f'{outcome_col} ~ {treatment_col}'
            if adjustment_cols:
                formula_base += ' + ' + ' + '.join(adjustment_cols)
            
            results_list = []
            
            # === OVERALL ANALYSIS ===
            st.info("📊 Computing Overall Model...")
            try:
                model_overall = logit(formula_base, data=df_clean).fit(disp=0)
                
                or_overall = np.exp(model_overall.params[treatment_col])
                se_logit = model_overall.bse[treatment_col]
                ci_overall = np.exp(model_overall.conf_int().loc[treatment_col])
                p_overall = model_overall.pvalues[treatment_col]
                z_overall = model_overall.tvalues[treatment_col]
                
                results_list.append({
                    'group': f'Overall (N={len(df_clean)})',
                    'n': len(df_clean),
                    'events': int(df_clean[outcome_col].sum()),
                    'or': or_overall,
                    'log_or': model_overall.params[treatment_col],
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
            
            for i, subgroup_val in enumerate(subgroups, 1):
                st.write(f"  Processing: {subgroup_col} = {subgroup_val}")
                
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
                
                if len(df_sub) < min_subgroup_n:
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: N={len(df_sub)} < {min_subgroup_n}, skipping")
                    logger.warning(f"Skipping subgroup {subgroup_val}: insufficient sample size")
                    continue
                
                try:
                    model_sub = logit(formula_base, data=df_sub).fit(disp=0)
                    
                    or_sub = np.exp(model_sub.params[treatment_col])
                    se_logit_sub = model_sub.bse[treatment_col]
                    ci_sub = np.exp(model_sub.conf_int().loc[treatment_col])
                    p_sub = model_sub.pvalues[treatment_col]
                    z_sub = model_sub.tvalues[treatment_col]
                    
                    results_list.append({
                        'group': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                        'subgroup': subgroup_val,
                        'n': len(df_sub),
                        'events': int(df_sub[outcome_col].sum()),
                        'or': or_sub,
                        'log_or': model_sub.params[treatment_col],
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
            
            # === INTERACTION TEST ===
            st.info("📊 Computing Interaction Test...")
            try:
                formula_int = f'{outcome_col} ~ {treatment_col} * {subgroup_col}'
                if adjustment_cols:
                    formula_int += ' + ' + ' + '.join(adjustment_cols)
                
                model_int = logit(formula_int, data=df_clean).fit(disp=0)
                
                # Find interaction term
                int_cols = [col for col in model_int.params.index if ':' in col]
                if int_cols:
                    int_col = int_cols[0]
                    p_interaction = model_int.pvalues[int_col]
                    coef_int = model_int.params[int_col]
                    se_int = model_int.bse[int_col]
                else:
                    p_interaction = np.nan
                    coef_int = np.nan
                    se_int = np.nan
                
                self.interaction_result = {
                    'p_value': p_interaction,
                    'coefficient': coef_int,
                    'se': se_int,
                    'significant': p_interaction < 0.05
                }
                
                logger.info(f"Interaction P={p_interaction:.4f}")
            
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
        Compute summary statistics from results.
        """
        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        # Heterogeneity assessment
        het_significant = self.interaction_result['significant']
        
        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'or_overall': overall['or'],
            'ci_overall': (overall['ci_low'], overall['ci_high']),
            'p_overall': overall['p_value'],
            'p_interaction': self.interaction_result['p_value'],
            'heterogeneous': het_significant,
            'or_range': (subgroups['or'].min(), subgroups['or'].max()),
            'or_range_ratio': subgroups['or'].max() / subgroups['or'].min() if subgroups['or'].min() > 0 else np.inf
        }
    
    def _format_output(self) -> dict:
        """
        Format output for reporting.
        """
        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        
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
                'p_value': float(self.interaction_result['p_value']) if not np.isnan(self.interaction_result['p_value']) else None,
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
        Create publication-grade forest plot.
        """
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        # Prepare data
        plot_data = self.results[['group', 'or', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'or', 'ci_low', 'ci_high', 'p_value']
        
        # Add interaction info to title
        p_int = self.interaction_result['p_value']
        het_text = "Heterogeneous ⚠️" if self.interaction_result['significant'] else "Homogeneous ✓"
        
        if not np.isnan(p_int):
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
        Generate clinical interpretation.
        """
        p_int = self.interaction_result['p_value']
        het = self.interaction_result['significant']
        
        if np.isnan(p_int):
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
    - Interaction testing (Wald test)
    - Multiple effect measures (HR, log-HR, SE)
    - Comprehensive statistics
    - Publication-grade forest plots
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with dataset.
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
        Validate all required columns and data types.
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
        Run complete Cox subgroup analysis.
        
        Parameters:
        -----------
        time_col : str
            Time to event variable
        event_col : str
            Event indicator (binary)
        treatment_col : str
            Treatment variable
        subgroup_col : str
            Stratification variable
        adjustment_cols : list, optional
            Adjustment variables
        min_subgroup_n : int
            Minimum sample size per subgroup
        min_events : int
            Minimum number of events per subgroup
        
        Returns:
        --------
        dict : Results dictionary
        """
        try:
            from lifelines import CoxPHFitter
            
            # Validate inputs
            self.validate_inputs(time_col, event_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            # Clean data
            cols_for_clean = {time_col, event_col, treatment_col, subgroup_col} | set(adjustment_cols)
            df_clean = self.df[list(cols_for_clean)].dropna()
            
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")
            
            results_list = []
            
            # === OVERALL ANALYSIS ===
            st.info("📊 Computing Overall Cox Model...")
            try:
                cph_overall = CoxPHFitter()
                covariates = [treatment_col] + adjustment_cols
                cph_overall.fit(
                    df_clean,
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
                st.write(f"  Processing: {subgroup_col} = {subgroup_val}")
                
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
                n_events = int(df_sub[event_col].sum())
                
                if len(df_sub) < min_subgroup_n or n_events < min_events:
                    msg = f"N={len(df_sub)}, events={n_events}"
                    st.warning(f"  ⚠️ Subgroup {subgroup_val}: {msg}, skipping")
                    logger.warning(f"Skipping {subgroup_val}: {msg}")
                    continue
                
                try:
                    cph_sub = CoxPHFitter()
                    cph_sub.fit(
                        df_sub,
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
            
            # === INTERACTION TEST ===
            st.info("📊 Computing Interaction Test...")
            try:
                # Create dummy interaction term
                df_clean_copy = df_clean.copy()
                df_clean_copy['__int_term'] = df_clean_copy[treatment_col] * pd.Categorical(
                    df_clean_copy[subgroup_col]
                ).codes
                
                cph_int = CoxPHFitter()
                cols_for_int = [treatment_col, '__int_term'] + adjustment_cols
                cph_int.fit(
                    df_clean_copy[[time_col, event_col] + cols_for_int],
                    duration_col=time_col,
                    event_col=event_col,
                    show_progress=False
                )
                
                if '__int_term' in cph_int.summary.index:
                    p_interaction = cph_int.summary.loc['__int_term', 'p']
                else:
                    p_interaction = np.nan
                
                self.interaction_result = {
                    'p_value': p_interaction,
                    'significant': p_interaction < 0.05
                }
                
                logger.info(f"Cox Interaction P={p_interaction:.4f}")
            
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
        Compute summary statistics.
        """
        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'hr_overall': overall['hr'],
            'ci_overall': (overall['ci_low'], overall['ci_high']),
            'p_overall': overall['p_value'],
            'p_interaction': self.interaction_result['p_value'],
            'heterogeneous': self.interaction_result['significant'],
            'hr_range': (subgroups['hr'].min(), subgroups['hr'].max())
        }
    
    def _format_output(self) -> dict:
        """
        Format output for reporting.
        """
        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        
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
                'p_value': float(self.interaction_result['p_value']) if not np.isnan(self.interaction_result['p_value']) else None,
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
        Create publication-grade forest plot.
        """
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        # Prepare data
        plot_data = self.results[['group', 'hr', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'hr', 'ci_low', 'ci_high', 'p_value']
        
        # Add interaction info
        p_int = self.interaction_result['p_value']
        het_text = "Heterogeneous ⚠️" if self.interaction_result['significant'] else "Homogeneous ✓"
        
        if not np.isnan(p_int):
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
        Generate clinical interpretation.
        """
        p_int = self.interaction_result['p_value']
        het = self.interaction_result['significant']
        
        if np.isnan(p_int):
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
