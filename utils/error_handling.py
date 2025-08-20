"""Comprehensive Error Handling Module for Multilingual Speech Translation System

This module provides standardized exception classes, error handling utilities,
and recovery mechanisms for all components of the speech translation pipeline.

Author: IBM Internship Project
Date: Sprint 7 - System Integration
"""

import sys
import traceback
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import logging
from pathlib import Path

# Import logging configuration
from .logging_config import get_logging_manager, LogLevel, LogCategory


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Enumeration of error categories."""
    SYSTEM = "system"
    AUDIO_PROCESSING = "audio_processing"
    MODEL_INFERENCE = "model_inference"
    RECONSTRUCTION = "reconstruction"
    VALIDATION = "validation"
    IO_ERROR = "io_error"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    RESOURCE = "resource"
    USER_INPUT = "user_input"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    session_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    user_id: Optional[str] = None


class SpeechTranslationError(Exception):
    """Base exception class for speech translation system."""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None,
                 recovery_suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.recovery_suggestions = recovery_suggestions or []
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error using the centralized logging system."""
        try:
            logging_manager = get_logging_manager()
            component = self.context.component if self.context else "unknown"
            
            metadata = {
                'category': self.category.value,
                'severity': self.severity.value,
                'recovery_suggestions': self.recovery_suggestions
            }
            
            if self.context:
                metadata.update({
                    'operation': self.context.operation,
                    'input_data': self.context.input_data,
                    'system_state': self.context.system_state
                })
            
            session_id = self.context.session_id if self.context else None
            
            logging_manager.log_error(
                component=component,
                error=self.original_error or self,
                context=self.message,
                session_id=session_id,
                metadata=metadata
            )
        except Exception:
            # Fallback to basic logging if centralized logging fails
            logging.error(f"Error in {component}: {self.message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': {
                'component': self.context.component if self.context else None,
                'operation': self.context.operation if self.context else None,
                'session_id': self.context.session_id if self.context else None
            },
            'recovery_suggestions': self.recovery_suggestions,
            'original_error': str(self.original_error) if self.original_error else None
        }


class AudioProcessingError(SpeechTranslationError):
    """Exception for audio processing errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUDIO_PROCESSING)
        kwargs.setdefault('recovery_suggestions', [
            "Check audio file format and integrity",
            "Verify sample rate compatibility",
            "Ensure sufficient audio duration",
            "Check for audio corruption"
        ])
        super().__init__(message, **kwargs)


class ModelInferenceError(SpeechTranslationError):
    """Exception for model inference errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL_INFERENCE)
        kwargs.setdefault('recovery_suggestions', [
            "Verify model weights are loaded correctly",
            "Check input tensor dimensions",
            "Ensure model is in evaluation mode",
            "Verify CUDA/CPU compatibility"
        ])
        super().__init__(message, **kwargs)


class ReconstructionError(SpeechTranslationError):
    """Exception for audio reconstruction errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RECONSTRUCTION)
        kwargs.setdefault('recovery_suggestions', [
            "Check feature matrix dimensions",
            "Verify reconstruction parameters",
            "Ensure sufficient memory for reconstruction",
            "Check output directory permissions"
        ])
        super().__init__(message, **kwargs)


class ValidationError(SpeechTranslationError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('recovery_suggestions', [
            "Check input parameters",
            "Verify data format",
            "Review validation criteria",
            "Check configuration settings"
        ])
        super().__init__(message, **kwargs)


class ConfigurationError(SpeechTranslationError):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_suggestions', [
            "Check configuration file syntax",
            "Verify all required parameters are set",
            "Review default configuration",
            "Check file permissions"
        ])
        super().__init__(message, **kwargs)


class ResourceError(SpeechTranslationError):
    """Exception for resource-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RESOURCE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_suggestions', [
            "Check available memory",
            "Verify disk space",
            "Monitor CPU usage",
            "Check file system permissions"
        ])
        super().__init__(message, **kwargs)


class IOError(SpeechTranslationError):
    """Exception for input/output errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.IO_ERROR)
        kwargs.setdefault('recovery_suggestions', [
            "Check file path exists",
            "Verify file permissions",
            "Ensure directory structure",
            "Check disk space"
        ])
        super().__init__(message, **kwargs)


class ErrorHandler:
    """Centralized error handler for the speech translation system."""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.logging_manager = get_logging_manager()
    
    def register_recovery_strategy(self, 
                                 error_type: type, 
                                 strategy: Callable[[Exception], Any]):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, 
                    error: Exception, 
                    context: Optional[ErrorContext] = None,
                    attempt_recovery: bool = True) -> Dict[str, Any]:
        """Handle an error with optional recovery attempt."""
        # Convert to SpeechTranslationError if needed
        if not isinstance(error, SpeechTranslationError):
            error = SpeechTranslationError(
                message=str(error),
                context=context,
                original_error=error
            )
        
        # Record error
        self.error_history.append(error)
        
        # Attempt recovery if enabled
        recovery_result = None
        if attempt_recovery and type(error) in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[type(error)](error)
            except Exception as recovery_error:
                self.logging_manager.log_error(
                    component="error_handler",
                    error=recovery_error,
                    context="Recovery strategy failed"
                )
        
        return {
            'error': error.to_dict(),
            'recovery_attempted': attempt_recovery,
            'recovery_successful': recovery_result is not None,
            'recovery_result': recovery_result
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        if not self.error_history:
            return {'total_errors': 0}
        
        by_category = {}
        by_severity = {}
        
        for error in self.error_history:
            # Count by category
            category = error.category.value
            by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'by_category': by_category,
            'by_severity': by_severity,
            'recent_errors': [error.to_dict() for error in self.error_history[-5:]]
        }


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def error_handler_decorator(component: str, 
                          operation: str,
                          reraise: bool = True,
                          return_on_error: Any = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                component=component,
                operation=operation,
                session_id=kwargs.get('session_id')
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                result = handler.handle_error(e, context)
                
                if reraise:
                    raise
                else:
                    return return_on_error
        
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                *args, 
                component: str = "unknown",
                operation: str = "unknown",
                default_return: Any = None,
                **kwargs) -> Any:
    """Safely execute a function with error handling."""
    context = ErrorContext(
        component=component,
        operation=operation
    )
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(e, context)
        return default_return


def validate_input(value: Any, 
                  validator: Callable[[Any], bool],
                  error_message: str,
                  component: str = "validation") -> Any:
    """Validate input with custom validator function."""
    try:
        if not validator(value):
            raise ValidationError(
                error_message,
                context=ErrorContext(
                    component=component,
                    operation="input_validation"
                )
            )
        return value
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        else:
            raise ValidationError(
                f"Validation failed: {str(e)}",
                context=ErrorContext(
                    component=component,
                    operation="input_validation"
                ),
                original_error=e
            )


def check_file_exists(file_path: str, component: str = "file_system") -> Path:
    """Check if file exists and return Path object."""
    path = Path(file_path)
    if not path.exists():
        raise IOError(
            f"File not found: {file_path}",
            context=ErrorContext(
                component=component,
                operation="file_check"
            )
        )
    return path


def check_directory_writable(dir_path: str, component: str = "file_system") -> Path:
    """Check if directory is writable."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Test write access
    test_file = path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise IOError(
            f"Directory not writable: {dir_path}",
            context=ErrorContext(
                component=component,
                operation="directory_check"
            ),
            original_error=e
        )
    
    return path


# Recovery strategies
def audio_processing_recovery(error: AudioProcessingError) -> Optional[Dict[str, Any]]:
    """Recovery strategy for audio processing errors."""
    # Implement audio-specific recovery logic
    return {
        'strategy': 'audio_fallback',
        'message': 'Attempting fallback audio processing parameters'
    }


def model_inference_recovery(error: ModelInferenceError) -> Optional[Dict[str, Any]]:
    """Recovery strategy for model inference errors."""
    # Implement model-specific recovery logic
    return {
        'strategy': 'model_fallback',
        'message': 'Attempting model inference with reduced precision'
    }


# Register default recovery strategies
def setup_default_recovery_strategies():
    """Setup default recovery strategies."""
    handler = get_error_handler()
    handler.register_recovery_strategy(AudioProcessingError, audio_processing_recovery)
    handler.register_recovery_strategy(ModelInferenceError, model_inference_recovery)


# Initialize default recovery strategies
setup_default_recovery_strategies()