"""
Caching layer for query results.
"""

import time
import pickle
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from src.config import ENABLE_CACHE, CACHE_TTL_SECONDS, CACHE_MAX_SIZE, PROJECT_ROOT
from src.logger import logger
from src.utils import hash_query


class QueryCache:
    """In-memory cache for query results."""
    
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS, max_size: int = CACHE_MAX_SIZE):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum number of entries
        """
        self.cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.enabled = ENABLE_CACHE
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query.
        
        Args:
            query: User query
            
        Returns:
            Cached result or None
        """
        if not self.enabled:
            return None
        
        cache_key = hash_query(query)
        
        if cache_key not in self.cache:
            return None
        
        timestamp, result = self.cache[cache_key]
        
        # Check if expired
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[cache_key]
            logger.debug(f"Cache entry expired for query: {query[:50]}...")
            return None
        
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return result
    
    def set(self, query: str, result: Dict[str, Any]) -> None:
        """Cache result for query.
        
        Args:
            query: User query
            result: Result to cache
        """
        if not self.enabled:
            return
        
        cache_key = hash_query(query)
        
        # Evict oldest entry if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry")
        
        self.cache[cache_key] = (time.time(), result)
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'enabled': self.enabled,
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


# Global cache instance
query_cache = QueryCache()
