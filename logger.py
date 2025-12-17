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
from typing import Optional, Any, Dict, List
import time
import traceback
import threading
from datetime import datetime

from config import CONFIG


class PerformanceLogger:
    """
    Track and log performance metrics.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Create a PerformanceLogger bound to a standard logger and prepare storage for timing data.
        
        Initializes the instance with the given `logging.Logger` and an empty dict that maps operation names to lists of elapsed times (in seconds).
        """
        self.logger = logger
        self.timings: Dict[str, list] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def track_time(self, operation: str, log_level: str = "DEBUG"):
        """
        Context manager that measures and logs the elapsed time of a named operation.
        
        If CONFIG['logging.log_performance'] is falsy, the context yields without measuring or logging. When enabled, the elapsed time is appended to self.timings[operation] (creating the list if needed) and a timestamped message is emitted on the wrapped logger at the requested log level.
        
        Parameters:
            operation (str): Name of the operation to record and log.
            log_level (str): Name of the logger method to call (e.g., "DEBUG", "INFO"); falls back to debug if unavailable.
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
            with self._lock:
                if operation not in self.timings:
                    self.timings[operation] = []
                self.timings[operation].append(elapsed)
            
            # Log timing
            log_method = getattr(self.logger, log_level.lower(), self.logger.debug)
            log_method(f"⏱️ {operation} completed in {elapsed:.3f}s")
    
    def get_timings(self, operation: Optional[str] = None) -> Dict[str, list]:
        """
        Return recorded performance timings.
        
        If `operation` is provided, return a dict containing only that operation mapped to its list of timings (empty list if no timings recorded for that operation). Otherwise return the full timings mapping.
        
        Parameters:
            operation (str, optional): Operation name to filter timings for.
        
        Returns:
            dict: Mapping of operation names to lists of elapsed times in seconds.
        """
        if operation:
            return {operation: self.timings.get(operation, [])}
        return self.timings
    
    def print_summary(self) -> None:
        """
        Log a human-readable summary of recorded operation timings.
        
        If no timings are present, the method does nothing. For each tracked operation it logs the average, minimum, maximum, and count of recorded durations using the instance's logger.
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
        """
        Create a ContextFilter instance and initialize its internal context storage.
        
        Initializes an empty dictionary used to store contextual key-value pairs that will be attached to log records.
        """
        super().__init__()
        self.context: Dict[str, Any] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Attach stored context key/value pairs as attributes on the given LogRecord.
        
        Parameters:
            record (logging.LogRecord): The log record to augment.
        
        Returns:
            bool: `True` to allow the record to be processed by logging handlers.
        """
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, **kwargs) -> None:
        """
        Add or update contextual key-value pairs that will be attached to subsequent log records.
        
        Parameters:
            **kwargs: Arbitrary keyword arguments whose keys and values will be stored as context attributes and made available on log records processed by this filter.
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """
        Remove all stored context key/value pairs used to enrich log records.
        
        After calling this method, subsequent log records will no longer include any previously set context values.
        """
        self.context.clear()

from typing import ClassVar

class LoggerFactory:
    """
    Factory for creating and managing loggers.
    """
    
    _loggers: ClassVar[Dict[str, 'Logger']] = {}
    _context_filter: Optional[ContextFilter] = None
    _perf_logger: Optional[PerformanceLogger] = None
    _configured = False
    
    @classmethod
    def configure(cls) -> None:
        """
        Perform one-time configuration of the application's logging system using values from CONFIG.
        
        Reads logging settings (level, format, date format) and sets up the root logger, a shared ContextFilter, and enabled handlers (file/console/Streamlit). If CONFIG disables logging, logging is globally disabled. The method is idempotent and will no-op if configuration has already been performed. On error it prints a warning to stderr and marks configuration as complete to avoid repeated attempts.
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
            numeric_level = getattr(logging, log_level.upper(), None)
            if numeric_level is None:
                print(f"[WARNING] Invalid log level '{log_level}', defaulting to INFO", file=sys.stderr)
                numeric_level = logging.INFO
            root_logger.setLevel(numeric_level)
            
            # Clear existing handlers only if re-configuring
            if root_logger.handlers:
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
        Configure rotating file logging for the given root logger using settings from CONFIG.
        
        Creates the log directory if missing and attaches a RotatingFileHandler formatted with `formatter`. Rotation parameters and file path are read from CONFIG; on any setup error a warning is printed to stderr and the function returns without raising.
        
        Parameters:
            root_logger (logging.Logger): The root logger to which the file handler will be attached.
            formatter (logging.Formatter): Formatter to apply to the file handler.
        
        Notes:
            Uses CONFIG keys: 'logging.log_dir' (default 'logs'), 'logging.log_file' (default 'app.log'),
            'logging.max_log_size' (default 10485760), and 'logging.backup_count' (default 5).
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
        Configure console (stdout) logging for the given root logger using the provided formatter.
        
        Reads the console log level from CONFIG['logging.console_level'] (defaults to 'INFO'), creates a StreamHandler that writes to stdout, applies the formatter and the class-wide context filter, and attaches the handler to the root logger.
        
        Parameters:
            root_logger (logging.Logger): Root logger to which the console handler will be attached.
            formatter (logging.Formatter): Formatter to apply to console log messages.
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
        Reduce log verbosity for Streamlit and Altair by setting their logger levels from CONFIG.
        
        Reads 'logging.streamlit_level' from CONFIG (defaulting to 'WARNING') and applies that level to the 'streamlit' and 'altair' loggers. If configuration fails, a debug message is printed to stderr and execution continues.
        """
        try:
            streamlit_level = CONFIG.get('logging.streamlit_level', 'WARNING')
            logging.getLogger('streamlit').setLevel(getattr(logging, streamlit_level))
            logging.getLogger('altair').setLevel(getattr(logging, streamlit_level))
        
        except Exception as e:
            # 🟢 FIX #9: Silently fail for Streamlit config
            print(f"[DEBUG] Streamlit logging config skipped: {e}", file=sys.stderr)

    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_logger(cls, name: str) -> 'Logger':
        """
        Retrieve a cached custom Logger by name, configuring the logging system on first use if necessary.
        
        Parameters:
            name (str): Logger name (typically __name__).
        
        Returns:
            Logger: Custom Logger instance associated with `name`.
        """
        # Configure if not done
        if not cls._configured:
            cls.configure()
        
        # Create or return existing logger
        with cls._lock:
            if name not in cls._loggers:
                standard_logger = logging.getLogger(name)
                logger = Logger(standard_logger, cls._context_filter)
                cls._loggers[name] = logger
            return cls._loggers[name]
        
        return cls._loggers[name]
    
    @classmethod
    def get_performance_logger(cls) -> PerformanceLogger:
        """
        Return the singleton PerformanceLogger instance used for recording operation timings.
        
        This method returns a shared PerformanceLogger for the application, creating and caching it on first access.
        
        Returns:
            PerformanceLogger: Singleton PerformanceLogger for tracking and retrieving performance timings.
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
        Wrap the provided standard logger with contextual and performance-tracking support.
        
        Parameters:
            standard_logger (logging.Logger): The underlying Python logger to delegate log calls to.
            context_filter (ContextFilter | None): Optional ContextFilter whose context will be applied to log records.
        """
        self._logger = standard_logger
        self._context_filter = context_filter
        self._perf_logger = LoggerFactory.get_performance_logger()
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """
        Log an informational message via the wrapped logger.
        
        Parameters:
            msg (str): Message format string or message object.
            *args: Positional arguments for message formatting.
            **kwargs: Keyword arguments forwarded to the underlying logger (e.g., `exc_info`, `stack_info`).
        """
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """
        Log a message with severity WARNING.
        """
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """
        Log a message with ERROR severity.
        
        Parameters:
            msg (str): The message format string.
            *args: Positional arguments used for message formatting.
            **kwargs: Keyword arguments forwarded to the underlying logger (for example, `exc_info`).
        """
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """
        Log a message with CRITICAL severity.
        
        Parameters:
            msg (str): Message format string.
            *args: Positional arguments for message formatting.
            **kwargs: Additional keyword arguments forwarded to the logger (for example `exc_info` or `stack_info`).
        """
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """
        Log a message and include the current exception traceback.
        
        Records exception information from the active exception context and formats the message using any supplied positional or keyword arguments.
        """
        self._logger.exception(msg, *args, **kwargs)
    
    def log_operation(self, operation: str, status: str = "started", **details) -> None:
        """
        Log an operation event with optional details.
        
        Builds a single-line message containing the operation name in brackets, an uppercase status, and any key=value pairs provided in `details`. Uses the ERROR level when `status` is "failed" (case-insensitive) and INFO level for other statuses.
        
        Parameters:
            operation (str): Name of the operation.
            status (str): Operation status such as "started", "completed", or "failed".
            **details: Additional key/value pairs to include in the log message.
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
        else:
            self.info(msg)
    
    def log_data_summary(self, df_name: str, shape: tuple, dtypes: Dict[str, str]) -> None:
        """
        Log a concise summary of a DataFrame's size and column type composition.
        
        Logs the DataFrame name, its shape (rows, columns), and counts of numeric and object-typed columns. This log is emitted only when CONFIG['logging.log_data_operations'] is truthy.
        
        Parameters:
            df_name (str): Identifier or name of the DataFrame.
            shape (tuple): Tuple of (rows, columns).
            dtypes (Dict[str, str]): Mapping from column name to dtype string (e.g., 'int64', 'float32', 'object').
        """
        if CONFIG.get('logging.log_data_operations'):
            self.info(
                f"📊 {df_name}: shape={shape}, "
                f"numeric={sum(1 for t in dtypes.values() if 'int' in t.lower() or 'float' in t.lower())}, "
                f"object={sum(1 for t in dtypes.values() if 'object' in t)}"
            )
    
    def log_analysis(self, analysis_type: str, outcome: str, n_vars: int, n_samples: int) -> None:
        """
        Log a concise summary of an analysis run.
        
        Logs the analysis type, outcome variable, number of predictor variables, and sample count. This message is emitted only when the `logging.log_analysis_operations` configuration flag is enabled.
        
        Parameters:
            analysis_type (str): Human-readable analysis name or type (e.g., "regression", "clustering").
            outcome (str): Outcome variable or target description.
            n_vars (int): Number of predictor variables used in the analysis.
            n_samples (int): Number of samples or observations processed.
        """
        if CONFIG.get('logging.log_analysis_operations'):
            self.info(
                f"📈 {analysis_type}: outcome='{outcome}', "
                f"predictors={n_vars}, n={n_samples}"
            )
    
    @contextmanager
    def track_time(self, operation: str, log_level: str = "DEBUG"):
        """
        Provide a context manager that records elapsed time for the named operation and logs the duration at the specified level.
        
        Parameters:
            operation (str): Name of the operation being measured.
            log_level (str): Logging level name to use when emitting the timing message (e.g., "DEBUG", "INFO"). Defaults to "DEBUG".
        """
        with self._perf_logger.track_time(operation, log_level):
            yield
    
    def get_timings(self) -> Dict[str, list]:
        """
        Retrieve recorded performance timings for all tracked operations.
        
        Returns:
            dict: Mapping from operation name (str) to a list of elapsed times in seconds (List[float]).
        """
        return self._perf_logger.get_timings()
    
    def set_context(self, **kwargs) -> None:
        """
        Attach key-value context that will be included on subsequent log records.
        
        Adds or updates context keys used by the logging system; existing keys are preserved unless overwritten.
        
        Parameters:
            **kwargs: Arbitrary mapping of context names to values that will be added to future log records.
        """
        if self._context_filter:
            self._context_filter.set_context(**kwargs)
    
    def clear_context(self) -> None:
        """
        Remove all context key-value pairs previously set for this logger; no-op if no context is configured.
        """
        if self._context_filter:
            self._context_filter.clear_context()


# Convenience function
def get_logger(name: str) -> Logger:
    """
    Obtain a configured logger for the given name.
    
    Parameters:
        name (str): The logger name, typically `__name__`.
    
    Returns:
        Logger: A Logger instance configured according to the module's logging settings.
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

    # Test 7: Exception logging
    print("\n[Test 7] Exception logging:")
    try:
        raise ValueError("Simulated error for testing")
    except Exception:
        logger.exception(f"Caught an exception during processing: {type(e).__name__}")

    # Test 8: Performance summary
    print("\n[Test 8] Performance summary:")
    with logger.track_time("quick_operation"):
        time.sleep(0.02)
    with logger.track_time("data_processing"):
        time.sleep(0.15)
    # Print the performance summary
    perf_logger = LoggerFactory.get_performance_logger()
    perf_logger.print_summary()

    # Test 9: Get specific timings
    print("\n[Test 9] Retrieve specific timings:")
    timings = logger.get_timings()
    logger.info(f"Total operations tracked: {len(timings)}")
    if "data_processing" in timings:
        logger.info(f"data_processing was called {len(timings['data_processing'])} time(s)")

    # Test 10: Nested context tracking
    print("\n[Test 10] Nested operations with context:")
    logger.set_context(module="data_loader")
    with logger.track_time("data_validation", log_level="INFO"):
        logger.info("Validating data structure")
        time.sleep(0.05)
        logger.warning("Found 3 missing values")
    logger.clear_context()

    # Test 11: Critical logging scenario
    print("\n[Test 11] Critical scenario:")
    logger.critical("System resources exhausted - immediate action required")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")
