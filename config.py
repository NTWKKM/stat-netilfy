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
        Initialize ConfigManager.
        
        Parameters:
            config_dict (dict, optional): Initial configuration dictionary.
                If None, loads from default config.
        """
        self._config = config_dict or self._get_default_config()
        self._env_prefix = "MEDSTAT_"
        self._load_env_overrides()
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            dict: Default configuration with all settings.
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
                "enabled": True,
                "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
                "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                
                # File Logging
                "file_enabled": False,  # 🔴 เปลี่ยนเป็น False
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
        Load configuration overrides from environment variables.
        
        Environment variable naming convention:
            MEDSTAT_<SECTION>_<KEY>=value
            Example: MEDSTAT_LOGGING_LEVEL=DEBUG
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
        Get configuration value using dot notation.
        
        Parameters:
            key (str): Dot-separated key path (e.g., 'logging.level')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Examples:
            CONFIG.get('analysis.logit_method')  # Returns 'auto'
            CONFIG.get('nonexistent.key', 'default')  # Returns 'default'
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
        Update configuration value using dot notation.
        
        Parameters:
            key (str): Dot-separated key path (e.g., 'logging.level')
            value: New value to set
        
        Raises:
            KeyError: If path doesn't exist in config
        
        Examples:
            CONFIG.update('logging.level', 'DEBUG')
            CONFIG.update('analysis.logit_method', 'firth')
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
        Set configuration value, optionally creating intermediate keys.
        
        Parameters:
            key (str): Dot-separated key path
            value: Value to set
            create (bool): If True, create missing intermediate keys
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
        Get entire configuration section.
        
        Parameters:
            section (str): Section name (e.g., 'logging')
        
        Returns:
            dict: Section configuration
        """
        import copy
        result = self.get(section, {})
        return copy.deepcopy(result) if isinstance(result, dict) else result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            dict: Complete configuration
        """
        import copy
        return copy.deepcopy(self._config)
    
    def to_json(self, filepath: Optional[str] = None, pretty: bool = True) -> str:
        """
        Export configuration as JSON.
        
        Parameters:
            filepath (str, optional): If provided, save to file
            pretty (bool): Pretty-print JSON
        
        Returns:
            str: JSON string
        """
        json_str = json.dumps(self._config, indent=2 if pretty else None)
        
        if filepath:
            Path(filepath).write_text(json_str)
        
        return json_str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration values.
        
        Returns:
            tuple: (is_valid, [list of error messages])
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
        """String representation of ConfigManager."""
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
