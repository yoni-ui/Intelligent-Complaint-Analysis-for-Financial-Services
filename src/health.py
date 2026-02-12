"""
Health check endpoints and monitoring.
"""

import time
from typing import Dict, Any
from pathlib import Path
from src.config import FAISS_INDEX_PATH, METADATA_PATH, OLLAMA_BASE_URL, LLM_PROVIDER
from src.logger import logger
from src.llm.factory import get_llm_client
from src.cache import query_cache


class HealthChecker:
    """Health check service for monitoring system status."""
    
    def __init__(self):
        """Initialize health checker."""
        self.start_time = time.time()
        try:
            self.llm_client = get_llm_client()
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client for health checks: {e}")
            self.llm_client = None
    
    def check_vector_store(self) -> Dict[str, Any]:
        """Check vector store health.
        
        Returns:
            Dictionary with health status
        """
        try:
            index_exists = FAISS_INDEX_PATH.exists()
            metadata_exists = METADATA_PATH.exists()
            
            if not index_exists or not metadata_exists:
                return {
                    'status': 'unhealthy',
                    'index_exists': index_exists,
                    'metadata_exists': metadata_exists,
                    'error': 'Vector store files missing'
                }
            
            return {
                'status': 'healthy',
                'index_exists': True,
                'metadata_exists': True,
                'index_size_mb': FAISS_INDEX_PATH.stat().st_size / (1024 * 1024),
                'metadata_size_mb': METADATA_PATH.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_llm(self) -> Dict[str, Any]:
        """Check LLM service health.
        
        Returns:
            Dictionary with health status
        """
        try:
            if self.llm_client is None:
                return {
                    'status': 'unhealthy',
                    'available': False,
                    'error': 'LLM client not initialized'
                }
            
            is_available = self.llm_client.is_available()
            
            if not is_available:
                provider_name = LLM_PROVIDER.upper()
                return {
                    'status': 'unhealthy',
                    'available': False,
                    'provider': LLM_PROVIDER,
                    'error': f'{provider_name} service not responding'
                }
            
            models = self.llm_client.list_models()
            
            result = {
                'status': 'healthy',
                'available': True,
                'provider': LLM_PROVIDER,
                'model': getattr(self.llm_client, 'model', 'unknown'),
                'available_models': models
            }
            
            # Add provider-specific info
            if LLM_PROVIDER.lower() == 'ollama':
                result['base_url'] = OLLAMA_BASE_URL
            
            return result
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {
                'status': 'unhealthy',
                'provider': LLM_PROVIDER,
                'error': str(e)
            }
    
    def check_cache(self) -> Dict[str, Any]:
        """Check cache health.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            stats = query_cache.stats()
            return {
                'status': 'healthy',
                **stats
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def overall_health(self) -> Dict[str, Any]:
        """Get overall system health.
        
        Returns:
            Dictionary with overall health status
        """
        vector_store = self.check_vector_store()
        llm = self.check_llm()
        cache = self.check_cache()
        
        all_healthy = (
            vector_store.get('status') == 'healthy' and
            llm.get('status') == 'healthy'
        )
        
        uptime_seconds = time.time() - self.start_time
        
        return {
            'status': 'healthy' if all_healthy else 'degraded',
            'uptime_seconds': int(uptime_seconds),
            'components': {
                'vector_store': vector_store,
                'llm': llm,
                'cache': cache
            }
        }


# Global health checker instance
health_checker = HealthChecker()
