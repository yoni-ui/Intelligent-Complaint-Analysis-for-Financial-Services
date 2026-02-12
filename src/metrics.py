"""
Simple metrics collection for monitoring.
"""

import time
from typing import Dict, Any
from collections import defaultdict
from src.config import ENABLE_METRICS
from src.logger import logger


class MetricsCollector:
    """Simple in-memory metrics collector."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.enabled = ENABLE_METRICS
        self.query_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.query_times = []
        self.error_types = defaultdict(int)
        self.start_time = time.time()
    
    def record_query(self, duration: float, cached: bool = False):
        """Record a query.
        
        Args:
            duration: Query duration in seconds
            cached: Whether result was from cache
        """
        if not self.enabled:
            return
        
        self.query_count += 1
        self.query_times.append(duration)
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Keep only last 1000 query times
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]
    
    def record_error(self, error_type: str):
        """Record an error.
        
        Args:
            error_type: Type of error
        """
        if not self.enabled:
            return
        
        self.error_count += 1
        self.error_types[error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics.
        
        Returns:
            Dictionary with metrics
        """
        if not self.enabled:
            return {'enabled': False}
        
        query_times = self.query_times[-100:] if self.query_times else []
        avg_time = sum(query_times) / len(query_times) if query_times else 0
        
        uptime = time.time() - self.start_time
        
        return {
            'enabled': True,
            'uptime_seconds': int(uptime),
            'query_count': self.query_count,
            'error_count': self.error_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            'avg_query_time_seconds': avg_time,
            'error_rate': (
                self.error_count / self.query_count
                if self.query_count > 0 else 0
            ),
            'error_types': dict(self.error_types)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.query_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.query_times = []
        self.error_types.clear()
        self.start_time = time.time()
        logger.info("Metrics reset")


# Global metrics collector
metrics_collector = MetricsCollector()
