"""
Logging configuration for production deployment.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.config import LOG_LEVEL, LOG_FILE, LOG_FORMAT, PROJECT_ROOT

def setup_logger(name: str = 'complaint_analyzer', log_file: str = None) -> logging.Logger:
    """Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path (overrides config)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_path = Path(log_file or LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()
