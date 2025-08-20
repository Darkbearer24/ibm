"""Centralized Logging Configuration for Multilingual Speech Translation System

This module provides a unified logging configuration for all components of the
speech translation pipeline, including structured logging, error tracking,
and performance monitoring.

Author: IBM Internship Project
Date: Sprint 7 - System Integration
"""

import logging
import logging.handlers
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
from enum import Enum
from dataclasses import dataclass, asdict


class LogLevel(Enum):
    """Enumeration of log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Enumeration of log categories."""
    SYSTEM = "system"
    PIPELINE = "pipeline"
    MODEL = "model"
    AUDIO = "audio"
    UI = "ui"
    PERFORMANCE = "performance"
    ERROR = "error"


@dataclass
class LogEntry:
    """Structured log entry data class."""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return asdict(self)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'category'):
            log_entry['category'] = record.category
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        if hasattr(record, 'metadata'):
            log_entry['metadata'] = record.metadata
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class LoggingManager:
    """Centralized logging manager for the speech translation system."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_structured: bool = True):
        """
        Initialize the logging manager.
        
        Parameters:
        -----------
        log_dir : str
            Directory for log files
        log_level : str
            Default logging level
        max_file_size : int
            Maximum size of log files before rotation
        backup_count : int
            Number of backup files to keep
        enable_console : bool
            Whether to enable console logging
        enable_structured : bool
            Whether to use structured JSON logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_structured = enable_structured
        
        # Initialize loggers
        self.loggers = {}
        self._setup_root_logger()
        self._setup_component_loggers()
    
    def _setup_root_logger(self):
        """Setup the root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler with rotation
        log_file = self.log_dir / "system.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        if self.enable_structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        root_logger.addHandler(file_handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            root_logger.addHandler(console_handler)
    
    def _setup_component_loggers(self):
        """Setup specialized loggers for different components."""
        components = {
            'pipeline': 'pipeline.log',
            'model': 'model.log',
            'audio': 'audio.log',
            'ui': 'ui.log',
            'performance': 'performance.log',
            'error': 'error.log'
        }
        
        for component, filename in components.items():
            logger = logging.getLogger(component)
            logger.setLevel(self.log_level)
            
            # File handler for component
            log_file = self.log_dir / filename
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            
            if self.enable_structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(levelname)s - %(message)s'
                    )
                )
            
            logger.addHandler(file_handler)
            logger.propagate = False  # Prevent duplicate logs in root logger
            
            self.loggers[component] = logger
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component."""
        if component in self.loggers:
            return self.loggers[component]
        else:
            return logging.getLogger(component)
    
    def log_structured(self, 
                      level: LogLevel,
                      category: LogCategory,
                      component: str,
                      message: str,
                      session_id: Optional[str] = None,
                      processing_time: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      error_details: Optional[Dict[str, Any]] = None):
        """Log structured entry."""
        logger = self.get_logger(category.value)
        
        # Create log record with extra fields
        extra = {
            'category': category.value,
            'component': component
        }
        
        if session_id:
            extra['session_id'] = session_id
        if processing_time:
            extra['processing_time'] = processing_time
        if metadata:
            extra['metadata'] = metadata
        if error_details:
            extra['error_details'] = error_details
        
        # Log with appropriate level
        log_method = getattr(logger, level.value.lower())
        log_method(message, extra=extra)
    
    def log_performance(self, 
                       component: str,
                       operation: str,
                       duration: float,
                       session_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        self.log_structured(
            level=LogLevel.INFO,
            category=LogCategory.PERFORMANCE,
            component=component,
            message=f"{operation} completed in {duration:.3f}s",
            session_id=session_id,
            processing_time=duration,
            metadata=metadata
        )
    
    def log_error(self, 
                  component: str,
                  error: Exception,
                  context: str = "",
                  session_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """Log error with full context."""
        error_details = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        self.log_structured(
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            component=component,
            message=f"Error in {component}: {str(error)}",
            session_id=session_id,
            metadata=metadata,
            error_details=error_details
        )
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of logs from the last N hours."""
        # This would typically query log files or a log database
        # For now, return a placeholder structure
        return {
            'period_hours': hours,
            'total_entries': 0,
            'by_level': {'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0},
            'by_component': {},
            'error_summary': [],
            'performance_summary': {}
        }


# Global logging manager instance
_logging_manager = None


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def setup_logging(log_dir: str = "logs", 
                 log_level: str = "INFO",
                 enable_structured: bool = True) -> LoggingManager:
    """Setup global logging configuration."""
    global _logging_manager
    _logging_manager = LoggingManager(
        log_dir=log_dir,
        log_level=log_level,
        enable_structured=enable_structured
    )
    return _logging_manager


# Convenience functions
def log_info(component: str, message: str, **kwargs):
    """Log info message."""
    get_logging_manager().log_structured(
        LogLevel.INFO, LogCategory.SYSTEM, component, message, **kwargs
    )


def log_warning(component: str, message: str, **kwargs):
    """Log warning message."""
    get_logging_manager().log_structured(
        LogLevel.WARNING, LogCategory.SYSTEM, component, message, **kwargs
    )


def log_error(component: str, error: Exception, context: str = "", **kwargs):
    """Log error message."""
    get_logging_manager().log_error(component, error, context, **kwargs)


def log_performance(component: str, operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    get_logging_manager().log_performance(component, operation, duration, **kwargs)