"""
Logging Framework for Medical Statistical Tool

This module provides comprehensive logging infrastructure with:
- Multiple output targets (file, console, Streamlit)
- Configurable log levels and formats
- Automatic log rotation
- Performance tracking
- Context tracking for debugging

Usage:
    from logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Application started")
    logger.warning("Data quality issue")
    logger.error("Analysis failed")
    
    # Performance tracking
    with logger.track_time("data_load"):
        df = pd.read_csv("data.csv")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
import time
import traceback
from datetime import datetime

from config import CONFIG


class PerformanceLogger:
    """
    Track and log performance metrics.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize PerformanceLogger."""
        self.logger = logger
        self.timings: Dict[str, list] = {}
    
    @contextmanager
    def track_time(self, operation: str, log_level: str = "DEBUG"):
        """
        Context manager to track operation timing.
        
        Parameters:
            operation (str): Operation name for logging
            log_level (str): Logging level for output
        
        Example:
            with logger.track_time("data_processing"):
                process_data()
        """
        if not CONFIG.get('logging.log_performance'):
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            # Store timing
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(elapsed)
            
            # Log timing
            log_method = getattr(self.logger, log_level.lower(), self.logger.debug)
            log_method(f"⏱️ {operation} completed in {elapsed:.3f}s")
    
    def get_timings(self, operation: Optional[str] = None) -> Dict[str, list]:
        """
        Get recorded timings.
        
        Parameters:
            operation (str, optional): Specific operation to get timings for
        
        Returns:
            dict: Timing data
        """
        if operation:
            return {operation: self.timings.get(operation, [])}
        return self.timings
    
    def print_summary(self) -> None:
        """
        Print timing summary.
        """
        if not self.timings:
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Performance Summary")
        self.logger.info("="*60)
        
        for operation, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                self.logger.info(
                    f"  {operation}: "
                    f"avg={avg:.3f}s, min={min_t:.3f}s, max={max_t:.3f}s (n={len(times)})"
                )
        
        self.logger.info("="*60)


class ContextFilter(logging.Filter):
    """
    Add context information to log records.
    """
    
    def __init__(self):
        """Initialize ContextFilter."""
        super().__init__()
        self.context: Dict[str, Any] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context to log record.
        
        Parameters:
            record: Log record
        
        Returns:
            bool: True to allow log record
        """
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, **kwargs) -> None:
        """
        Set context variables.
        
        Example:
            context_filter.set_context(user_id="123", session="abc")
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """
        Clear all context.
        """
        self.context.clear()


class LoggerFactory:
    """
    Factory for creating and managing loggers.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _context_filter: Optional[ContextFilter] = None
    _perf_logger: Optional[PerformanceLogger] = None
    _configured = False
    
    @classmethod
    def configure(cls) -> None:
        """
        Configure logging system based on configuration.
        
        Should be called once at application startup.
        🟢 FIX #9: Made safe with try-catch to prevent infinite loading on errors
        """
        if cls._configured:
            return
        
        try:
            # Check if logging enabled
            if not CONFIG.get('logging.enabled'):
                logging.disable(logging.CRITICAL)
                cls._configured = True
                return
            
            # Get logging config
            log_level = CONFIG.get('logging.level', 'INFO')
            log_format = CONFIG.get('logging.format')
            date_format = CONFIG.get('logging.date_format')
            
            # Create formatter
            formatter = logging.Formatter(log_format, datefmt=date_format)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, log_level))
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Create context filter
            cls._context_filter = ContextFilter()
            
            # File logging
            if CONFIG.get('logging.file_enabled'):
                cls._setup_file_logging(root_logger, formatter)
            
            # Console logging
            if CONFIG.get('logging.console_enabled'):
                cls._setup_console_logging(root_logger, formatter)
            
            # Streamlit logging (suppress some warnings)
            if CONFIG.get('logging.streamlit_enabled'):
                cls._setup_streamlit_logging()
            
            cls._configured = True
        
        except Exception as e:
            # 🟢 FIX #9: Catch errors and set configured flag to prevent retry loops
            print(f"[WARNING] Logging configuration failed: {e}", file=sys.stderr)
            cls._configured = True  # Mark as configured to prevent retry
    
    @classmethod
    def _setup_file_logging(cls, root_logger: logging.Logger, formatter: logging.Formatter) -> None:
        """
        Setup file logging with rotation.
        
        Parameters:
            root_logger: Root logger to configure
            formatter: Log formatter
        
        🟢 FIX #9: Added try-catch to handle disk/permission issues gracefully
        """
        try:
            log_dir = Path(CONFIG.get('logging.log_dir', 'logs'))
            log_dir.mkdir(exist_ok=True, parents=True)
            
            log_file = log_dir / CONFIG.get('logging.log_file', 'app.log')
            max_size = CONFIG.get('logging.max_log_size', 10485760)
            backup_count = CONFIG.get('logging.backup_count', 5)
            
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count
            )
            handler.setFormatter(formatter)
            handler.addFilter(cls._context_filter)
            root_logger.addHandler(handler)
        
        except Exception as e:
            # 🟢 FIX #9: Log error but don't crash
            print(f"[WARNING] Failed to setup file logging: {e}", file=sys.stderr)
    
    @classmethod
    def _setup_console_logging(cls, root_logger: logging.Logger, formatter: logging.Formatter) -> None:
        """
        Setup console logging.
        
        Parameters:
            root_logger: Root logger to configure
            formatter: Log formatter
        """
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_level = CONFIG.get('logging.console_level', 'INFO')
            console_handler.setLevel(getattr(logging, console_level))
            console_handler.setFormatter(formatter)
            console_handler.addFilter(cls._context_filter)
            root_logger.addHandler(console_handler)
        
        except Exception as e:
            # 🟢 FIX #9: Log error but don't crash
            print(f"[WARNING] Failed to setup console logging: {e}", file=sys.stderr)
    
    @classmethod
    def _setup_streamlit_logging(cls) -> None:
        """
        Configure Streamlit logger to reduce noise.
        """
        try:
            streamlit_level = CONFIG.get('logging.streamlit_level', 'WARNING')
            logging.getLogger('streamlit').setLevel(getattr(logging, streamlit_level))
            logging.getLogger('altair').setLevel(getattr(logging, streamlit_level))
        
        except Exception as e:
            # 🟢 FIX #9: Silently fail for Streamlit config
            pass
    
    @classmethod
    def get_logger(cls, name: str) -> 'Logger':
        """
        Get or create logger.
        
        Parameters:
            name (str): Logger name (usually __name__)
        
        Returns:
            Logger: Custom logger instance
        """
        # Configure if not done
        if not cls._configured:
            cls.configure()
        
        # Create or return existing logger
        if name not in cls._loggers:
            standard_logger = logging.getLogger(name)
            logger = Logger(standard_logger, cls._context_filter)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def get_performance_logger(cls) -> PerformanceLogger:
        """
        Get performance logger.
        
        Returns:
            PerformanceLogger: Performance tracking logger
        """
        if cls._perf_logger is None:
            perf_std_logger = logging.getLogger('performance')
            cls._perf_logger = PerformanceLogger(perf_std_logger)
        return cls._perf_logger


class Logger:
    """
    Wrapper around standard logger with additional features.
    """
    
    def __init__(self, standard_logger: logging.Logger, context_filter: Optional[ContextFilter] = None):
        """
        Initialize Logger.
        
        Parameters:
            standard_logger: Standard Python logger
            context_filter: Optional context filter
        """
        self._logger = standard_logger
        self._context_filter = context_filter
        self._perf_logger = LoggerFactory.get_performance_logger()
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)
    
    def log_operation(self, operation: str, status: str = "started", **details) -> None:
        """
        Log operation event.
        
        Parameters:
            operation (str): Operation name
            status (str): Operation status (started, completed, failed)
            **details: Additional details
        """
        msg_parts = [f"[{operation}]"]
        
        if status:
            msg_parts.append(f"{status.upper()}")
        
        if details:
            detail_str = " | ".join([f"{k}={v}" for k, v in details.items()])
            msg_parts.append(detail_str)
        
        msg = " ".join(msg_parts)
        
        if status.lower() == "failed":
            self.error(msg)
        elif status.lower() == "completed":
            self.info(msg)
        else:
            self.info(msg)
    
    def log_data_summary(self, df_name: str, shape: tuple, dtypes: Dict[str, str]) -> None:
        """
        Log data frame summary.
        
        Parameters:
            df_name (str): DataFrame name
            shape (tuple): DataFrame shape (rows, cols)
            dtypes (dict): Column dtypes
        """
        if CONFIG.get('logging.log_data_operations'):
            self.info(
                f"📊 {df_name}: shape={shape}, "
                f"numeric={sum(1 for t in dtypes.values() if 'int' in t or 'float' in t)}, "
                f"object={sum(1 for t in dtypes.values() if 'object' in t)}"
            )
    
    def log_analysis(self, analysis_type: str, outcome: str, n_vars: int, n_samples: int) -> None:
        """
        Log analysis execution.
        
        Parameters:
            analysis_type (str): Type of analysis
            outcome (str): Outcome variable
            n_vars (int): Number of variables
            n_samples (int): Number of samples
        """
        if CONFIG.get('logging.log_analysis_operations'):
            self.info(
                f"📈 {analysis_type}: outcome='{outcome}', "
                f"predictors={n_vars}, n={n_samples}"
            )
    
    @contextmanager
    def track_time(self, operation: str, log_level: str = "DEBUG"):
        """
        Context manager for tracking operation timing.
        
        Parameters:
            operation (str): Operation name
            log_level (str): Logging level
        
        Example:
            with logger.track_time("data_loading"):
                df = load_data()
        """
        with self._perf_logger.track_time(operation, log_level):
            yield
    
    def get_timings(self) -> Dict[str, list]:
        """
        Get performance timings.
        
        Returns:
            dict: Timing data
        """
        return self._perf_logger.get_timings()
    
    def set_context(self, **kwargs) -> None:
        """
        Set context for logging.
        
        Parameters:
            **kwargs: Context key-value pairs
        
        Example:
            logger.set_context(user_id="123", session="abc")
        """
        if self._context_filter:
            self._context_filter.set_context(**kwargs)
    
    def clear_context(self) -> None:
        """
        Clear logging context.
        """
        if self._context_filter:
            self._context_filter.clear_context()


# Convenience function
def get_logger(name: str) -> Logger:
    """
    Get a logger instance.
    
    Parameters:
        name (str): Logger name (usually __name__)
    
    Returns:
        Logger: Custom logger instance
    
    Example:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello world")
    """
    return LoggerFactory.get_logger(name)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("\n" + "="*70)
    print("Logging Framework - Test")
    print("="*70)
    
    logger = get_logger(__name__)
    
    # Test 1: Basic logging
    print("\n[Test 1] Basic logging:")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Test 2: Operation logging
    print("\n[Test 2] Operation logging:")
    logger.log_operation("file_upload", "started", filename="data.csv", size="5MB")
    logger.log_operation("file_upload", "completed", rows=10000, columns=50)
    
    # Test 3: Performance tracking
    print("\n[Test 3] Performance tracking:")
    with logger.track_time("data_processing"):
        time.sleep(0.1)  # Simulate work
    
    with logger.track_time("analysis", log_level="info"):
        time.sleep(0.05)  # Simulate work
    
    # Test 4: Context
    print("\n[Test 4] Context tracking:")
    logger.set_context(user_id="user123", session="sess456")
    logger.info("Message with context")
    logger.clear_context()
    logger.info("Message without context")
    
    # Test 5: Data summary
    print("\n[Test 5] Data summary:")
    logger.log_data_summary(
        "patients_df",
        shape=(1000, 15),
        dtypes={"age": "int64", "name": "object", "bmi": "float64"}
    )
    
    # Test 6: Analysis logging
    print("\n[Test 6] Analysis logging:")
    logger.log_analysis(
        "Logistic Regression",
        outcome="disease_status",
        n_vars=12,
        n_samples=500
    )
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")
