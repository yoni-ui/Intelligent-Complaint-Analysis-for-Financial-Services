"""
Utility functions for production deployment.
"""

import re
import hashlib
import time
from typing import Optional, Dict, Any
from functools import wraps
from src.config import MAX_QUERY_LENGTH, MAX_RETRIES, RETRY_DELAY
from src.logger import logger


def sanitize_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Input truncated from {len(text)} to {max_length} characters")
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate user query.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not isinstance(query, str):
        return False, "Query cannot be empty"
    
    if len(query.strip()) == 0:
        return False, "Query cannot be empty"
    
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
    
    # Check for suspicious patterns (basic security check)
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
        r'exec\s*\(',
        r'eval\s*\(',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected in query: {pattern}")
            return False, "Query contains invalid characters"
    
    return True, None


def hash_query(query: str) -> str:
    """Generate hash for query caching.
    
    Args:
        query: User query
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(query.encode('utf-8')).hexdigest()


def retry_on_failure(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Decorator for retrying functions on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator


def format_error_message(error: Exception, include_details: bool = False) -> str:
    """Format error message for user display.
    
    Args:
        error: Exception instance
        include_details: Whether to include detailed error information
        
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    
    if isinstance(error, ConnectionError):
        return "Unable to connect to the AI service. Please check if the service is running."
    elif isinstance(error, TimeoutError):
        return "Request timed out. Please try again with a shorter query."
    elif isinstance(error, FileNotFoundError):
        return "Required data files are missing. Please ensure the system is properly configured."
    elif isinstance(error, ValueError):
        return f"Invalid input: {str(error)}"
    else:
        if include_details:
            return f"An error occurred: {error_type} - {str(error)}"
        else:
            return "An unexpected error occurred. Please try again later."


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {ip: [timestamps]}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed.
        
        Args:
            identifier: Unique identifier (e.g., IP address)
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        
        # Clean old entries
        if identifier in self.requests:
            self.requests[identifier] = [
                ts for ts in self.requests[identifier]
                if now - ts < self.window_seconds
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier.
        
        Args:
            identifier: Unique identifier
            
        Returns:
            Number of remaining requests
        """
        now = time.time()
        if identifier not in self.requests:
            return self.max_requests
        
        self.requests[identifier] = [
            ts for ts in self.requests[identifier]
            if now - ts < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(self.requests[identifier]))


# Global rate limiter instance
rate_limiter = RateLimiter()
