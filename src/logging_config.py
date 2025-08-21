import logging
import json
import os
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

def setup_logging(level: str = None) -> None:
    """Setup structured logging for the application"""
    
    # Get log level from environment or default to INFO
    log_level = level or os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(JSONFormatter())
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to avoid duplicate logs
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.WARNING)
    
    logging.info("Logging configured", extra={"extra_fields": {"log_level": log_level}})

def log_request(request_id: str, message: str, **kwargs) -> None:
    """Log a request with request ID"""
    logging.info(message, extra={"extra_fields": {"request_id": request_id, **kwargs}})

def log_error(request_id: str, message: str, error: Exception = None, **kwargs) -> None:
    """Log an error with request ID and exception details"""
    extra_fields = {"request_id": request_id, **kwargs}
    if error:
        extra_fields["error_type"] = type(error).__name__
        extra_fields["error_message"] = str(error)
    
    logging.error(message, extra={"extra_fields": extra_fields}, exc_info=error is not None)
