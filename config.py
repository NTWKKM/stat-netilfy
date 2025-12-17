"""
Configuration Management System for Medical Statistical Tool

This module provides centralized configuration management for the application,
including analysis parameters, UI settings, logging configuration, and runtime options.

Usage:
    from config import CONFIG
    
    # Access config
    print(CONFIG['analysis']['logit_method'])
    
    # Update config (runtime)
    CONFIG.update('analysis.logit_method', 'firth')
    
    # Get with default
    value = CONFIG.get('some.nested.key', default='default_value')
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Dict
import warnings


class ConfigManager:
    """
    Centralized configuration management with hierarchical key access.
    
    Supports:
    - Nested dictionary access with dot notation
    - Default values and fallbacks
    - Environment variable overrides
    - Config validation
    - Runtime updates
    """
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Create a ConfigManager populated with the given configuration or the module defaults and apply environment variable overrides.
        
        Parameters:
            config_dict (dict | None): Optional initial configuration dictionary to use instead of the built-in defaults. If None, the manager is initialized from the default configuration.
        """
        self._config = config_dict or self._get_default_config()
        self._env_prefix = "MEDSTAT_"
        self._load_env_overrides()
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Provide the default nested configuration used by the application.
        
        Returns:
            Dict[str, Any]: A dictionary with the default configuration sections
            ('analysis', 'ui', 'logging', 'performance', 'validation', 'debug') and
            their corresponding default settings.
        """
        return {
            # ========== ANALYSIS SETTINGS ==========
            "analysis": {
                # Logistic Regression
                "logit_method": "auto",  # 'auto', 'firth', 'bfgs', 'default'
                "logit_max_iter": 100,
                "logit_screening_p": 0.20,  # Variables with p < this get into multivariate
                "logit_min_cases": 10,  # Minimum cases for multivariate analysis
                
                # Variable Detection
                "var_detect_threshold": 10,  # Unique values threshold for categorical/continuous
                "var_detect_decimal_pct": 0.30,  # Decimal % for continuous classification
                
                # P-value Handling
                "pvalue_bounds_lower": 0.0,
                "pvalue_bounds_upper": 1.0,
                "pvalue_clip_tolerance": 0.0001,  # Allow +/- this much
                "pvalue_format_small": "<0.001",
                "pvalue_format_large": ">0.999",
                
                # Survival Analysis
                "survival_method": "kaplan-meier",  # 'kaplan-meier', 'weibull'
                "cox_method": "efron",  # 'efron', 'breslow'
                
                # Missing Data
                "missing_strategy": "complete-case",  # 'complete-case', 'drop'
                "missing_threshold_pct": 50,  # Flag if >X% missing in a column
            },
            
            # ========== UI & DISPLAY SETTINGS ==========
            "ui": {
                # Page Setup
                "page_title": "Medical Stat Tool",
                "layout": "wide",
                "theme": "light",  # 'light', 'dark', 'auto'
                
                # Sidebar
                "sidebar_width": 300,
                "show_sidebar_logo": True,
                
                # Tables
                "table_max_rows": 1000,  # Max rows to display in data table
                "table_pagination": True,
                "table_decimal_places": 3,
                
                # Plots
                "plot_width": 10,
                "plot_height": 6,
                "plot_dpi": 100,
                "plot_style": "seaborn",
            },
            
            # ========== LOGGING SETTINGS ==========
            "logging": {
                "enabled": True, # ลองเปลี่ยนเป็น False เพื่อทดสอบ
                "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
                "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                
                # File Logging
                "file_enabled": True,  # 🔴 เปลี่ยนเป็น False
                "log_dir": "logs",
                "log_file": "app.log",
                "max_log_size": 10485760,  # 10MB in bytes
                "backup_count": 5,
                
                # Console Logging
                "console_enabled": True,
                "console_level": "INFO",
                
                # Streamlit Logging
                "streamlit_enabled": True,
                "streamlit_level": "WARNING",
                
                # What to Log
                "log_file_operations": True,
                "log_data_operations": True,
                "log_analysis_operations": True,
                "log_ui_events": False,  # Can be verbose
                "log_performance": True,  # Timing information
            },
            
            # ========== PERFORMANCE SETTINGS ==========
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,  # seconds
                "enable_compression": False,
                "num_threads": 4,
            },
            
            # ========== VALIDATION SETTINGS ==========
            "validation": {
                "strict_mode": False,  # Warn vs Error on validation failures
                "validate_inputs": True,
                "validate_outputs": True,
                "auto_fix_errors": True,  # Try to fix issues automatically
            },
            
            # ========== DEVELOPER SETTINGS ==========
            "debug": {
                "enabled": False,
                "verbose": False,
                "profile_performance": False,
                "show_timings": False,
            },
        }
    
    def _load_env_overrides(self) -> None:
        """
        Apply configuration overrides from environment variables that start with the MEDSTAT_ prefix.
        
        Environment variables must follow the form MEDSTAT_<SECTION>_<KEY>=value; the portion after the prefix is lowercased and split on underscores, where the first segment is treated as the section and the remaining segments are joined with underscores to form the dot-notated key within that section (e.g., MEDSTAT_LOGGING_LEVEL -> logging.level). Variables without at least a section and key are ignored. If applying an override fails (e.g., type or key errors), a warning is emitted and the override is skipped.
        """
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                # Parse environment variable
                # MEDSTAT_LOGGING_LEVEL -> ['logging', 'level']
                parts = key[len(self._env_prefix):].lower().split('_')
                
                if len(parts) < 2:
                    continue
                
                section = parts[0]
                key_name = '_'.join(parts[1:])
                
                # Try to set the value
                try:
                    self.update(f"{section}.{key_name}", value)
                except (KeyError, ValueError, TypeError) as e:
                    warnings.warn(f"Failed to set env override {key}={value}: {e}", stacklevel=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using a dot-separated key path.
        
        Parameters:
            key (str): Dot-separated path to a nested configuration value (e.g., "logging.level").
            default: Value to return if the specified path does not exist.
        
        Returns:
            The configuration value at the given path, or `default` if the path is not found.
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """
        Set an existing configuration value identified by a dot-separated path.
        
        Parameters:
            key (str): Dot-separated path to an existing configuration entry (e.g., "logging.level").
            value (Any): Value to assign to the configuration entry.
        
        Raises:
            KeyError: If any intermediate path segment or the final key does not exist in the configuration.
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent key
        for k in keys[:-1]:
            if k not in config:
                raise KeyError(f"Config path '{'.'.join(keys[:-1])}' does not exist")
            config = config[k]
        
        # Set final key
        final_key = keys[-1]
        if final_key not in config:
            raise KeyError(f"Config key '{key}' does not exist")
        
        config[final_key] = value
    
    def set_nested(self, key: str, value: Any, create: bool = False) -> None:
        """
        Set a value in the configuration using a dot-separated path, optionally creating missing intermediate dictionaries.
        
        Parameters:
            key (str): Dot-separated path to the configuration key (e.g., "section.sub.key").
            value (Any): Value to assign to the final key.
            create (bool): If True, create missing intermediate dictionaries along the path; if False and a path segment is missing, a KeyError is raised.
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate/create path to parent
        for k in keys[:-1]:
            if k not in config:
                if create:
                    config[k] = {}
                else:
                    raise KeyError(f"Config path '{k}' does not exist")
            config = config[k]
        
        # Set final key
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Return a deep copy of a top-level configuration section.
        
        Parameters:
            section (str): Top-level section name (e.g., "logging").
        
        Returns:
            dict or Any: A deep copy of the section dictionary if the section is a dict; otherwise the section value as-is.
        """
        import copy
        result = self.get(section, {})
        return copy.deepcopy(result) if isinstance(result, dict) else result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get a deep copy of the entire configuration dictionary.
        
        Returns:
            dict: A deep copy of the full configuration that can be modified without affecting the manager's internal state.
        """
        import copy
        return copy.deepcopy(self._config)
    
    def to_json(self, filepath: Optional[str] = None, pretty: bool = True) -> str:
        """
        Serialize the current configuration to a JSON string.
        
        Parameters:
            filepath (str | None): Optional filesystem path to write the JSON output; when provided, the file is overwritten.
            pretty (bool): If True, format the JSON with indentation for readability; if False, produce compact JSON.
        
        Returns:
            str: The configuration serialized as a JSON-formatted string.
        """
        json_str = json.dumps(self._config, indent=2 if pretty else None)
        
        if filepath:
            Path(filepath).write_text(json_str)
        
        return json_str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate key configuration constraints and collect any violations.
        
        Performs a set of sanity checks on configuration values and records any problems found:
        - Ensures `analysis.logit_screening_p` is greater than 0 and less than 1.
        - Ensures `analysis.pvalue_bounds_lower` is less than `analysis.pvalue_bounds_upper`.
        - Ensures `logging.level` is one of `['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']`.
        - Ensures `analysis.logit_method` is one of `['auto', 'firth', 'bfgs', 'default']`.
        
        Returns:
            tuple: (is_valid, errors) where `is_valid` is `True` if no validation errors were found, `False` otherwise; `errors` is a list of human-readable error messages.
        """
        errors = []
        
        # Validate analysis settings
        screening_p = self.get('analysis.logit_screening_p')
        if screening_p is None or not (0 < screening_p < 1):
            errors.append("analysis.logit_screening_p must be between 0 and 1")
        
        # Validate p-value bounds
        lower = self.get('analysis.pvalue_bounds_lower')
        upper = self.get('analysis.pvalue_bounds_upper')
        if lower is None or upper is None or not (lower < upper):
            errors.append("pvalue_bounds_lower must be < pvalue_bounds_upper")
        
        # Validate logging
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.get('logging.level') not in valid_levels:
            errors.append(f"logging.level must be one of {valid_levels}")
        
        # Validate analysis method
        valid_methods = ['auto', 'firth', 'bfgs', 'default']
        if self.get('analysis.logit_method') not in valid_methods:
            errors.append(f"analysis.logit_method must be one of {valid_methods}")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        """
        Return a concise representation of the ConfigManager showing how many top-level sections it contains.
        
        Returns:
            str: A string in the form "ConfigManager(<N> sections)" where <N> is the number of top-level configuration sections.
        """
        return f"ConfigManager({len(self._config)} sections)"


# Global config instance
CONFIG = ConfigManager()


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("\n" + "="*60)
    print("Configuration Management System - Test")
    print("="*60)
    
    # Test 1: Get values
    print("\n[Test 1] Getting configuration values:")
    print(f"  Logit method: {CONFIG.get('analysis.logit_method')}")
    print(f"  Logging level: {CONFIG.get('logging.level')}")
    print(f"  Log file enabled: {CONFIG.get('logging.file_enabled')}")
    
    # Test 2: Get with default
    print("\n[Test 2] Getting with defaults:")
    print(f"  Nonexistent key: {CONFIG.get('some.fake.key', 'default_value')}")
    
    # Test 3: Update config
    print("\n[Test 3] Updating configuration:")
    try:
        CONFIG.update('logging.level', 'DEBUG')
        print(f"  ✓ Updated logging.level to: {CONFIG.get('logging.level')}")
    except KeyError as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Get section
    print("\n[Test 4] Getting section:")
    logging_section = CONFIG.get_section('logging')
    print(f"  Logging section keys: {list(logging_section.keys())}")
    
    # Test 5: Validate
    print("\n[Test 5] Validating configuration:")
    is_valid, errors = CONFIG.validate()
    print(f"  Valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"    ✗ {err}")
    else:
        print("    ✓ No errors found")
    
    # Test 6: Export to JSON
    print("\n[Test 6] Exporting configuration:")
    json_str = CONFIG.to_json(pretty=False)
    print(f"  JSON length: {len(json_str)} characters")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")
