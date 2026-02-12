"""
LLM client factory that supports multiple providers.
"""

from typing import Optional, Union
from src.config import LLM_PROVIDER, GOOGLE_API_KEY
from src.logger import logger

# Import clients
from .local_ollama import OllamaClient, get_llm_client as get_ollama_client
from .google_gemini import GoogleGeminiClient, get_gemini_client


def get_llm_client(model: Optional[str] = None) -> Union[OllamaClient, GoogleGeminiClient]:
    """Factory function to get LLM client based on configuration.
    
    Args:
        model: Optional model name override
        
    Returns:
        Configured LLM client (OllamaClient or GoogleGeminiClient)
    """
    provider = LLM_PROVIDER.lower()
    
    if provider == 'google' or (provider == 'gemini'):
        if not GOOGLE_API_KEY:
            logger.warning("Google API key not found. Falling back to Ollama.")
            return get_ollama_client(model=model)
        
        logger.info(f"Using Google Gemini provider with model: {model or 'default'}")
        return get_gemini_client(model=model)
    
    elif provider == 'ollama':
        logger.info(f"Using Ollama provider with model: {model or 'default'}")
        return get_ollama_client(model=model)
    
    else:
        logger.warning(f"Unknown LLM provider: {provider}. Falling back to Ollama.")
        return get_ollama_client(model=model)


def get_llm_client_by_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Union[OllamaClient, GoogleGeminiClient]:
    """Get LLM client for a specific provider.
    
    Args:
        provider: Provider name ('ollama' or 'google')
        model: Optional model name override
        api_key: Optional API key (for Google)
        
    Returns:
        Configured LLM client
    """
    provider = provider.lower()
    
    if provider in ['google', 'gemini']:
        return get_gemini_client(api_key=api_key, model=model)
    elif provider == 'ollama':
        return get_ollama_client(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'google'.")
