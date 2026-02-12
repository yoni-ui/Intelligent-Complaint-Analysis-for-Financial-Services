"""
Local LLM backend using Ollama.

This module provides a simple interface to call local LLMs via Ollama's HTTP API.
"""

import requests
from typing import Optional
from src.config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL, LLM_TIMEOUT, LLM_MAX_TOKENS, LLM_TEMPERATURE
from src.logger import logger
from src.utils import retry_on_failure

# Default Ollama configuration (fallback if config not loaded)
DEFAULT_MODEL = DEFAULT_LLM_MODEL


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        """Initialize Ollama client.
        
        Args:
            model: Model name to use (e.g., 'mistral:7b-instruct', 'gemma3:4b')
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
    
    @retry_on_failure(max_retries=2)
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> str:
        """Generate text completion from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1), uses config default if None
            max_tokens: Maximum tokens to generate, uses config default if None
            stream: Whether to stream response (not implemented yet)
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        max_toks = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_toks
            }
        }
        
        logger.debug(f"Generating with model {self.model}, temperature={temp}, max_tokens={max_toks}")
        
        try:
            response = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "")
            logger.debug(f"Generated {len(answer)} characters")
            return answer
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama at {self.base_url}: {e}")
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama request timed out after {LLM_TIMEOUT}s")
            raise TimeoutError("Ollama request timed out. Try a shorter prompt or increase timeout.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request error: {e}")
            raise RuntimeError(f"Ollama request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {e}")
            raise RuntimeError(f"Ollama error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            is_avail = response.status_code == 200
            if not is_avail:
                logger.warning(f"Ollama health check returned status {response.status_code}")
            return is_avail
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except:
            return []


def get_llm_client(model: Optional[str] = None) -> OllamaClient:
    """Factory function to get LLM client.
    
    Args:
        model: Optional model name override
        
    Returns:
        Configured OllamaClient
    """
    return OllamaClient(model=model or DEFAULT_MODEL)


# Quick test
if __name__ == "__main__":
    client = OllamaClient()
    
    print(f"Ollama available: {client.is_available()}")
    print(f"Available models: {client.list_models()}")
    
    if client.is_available():
        print("\nTest generation:")
        response = client.generate("Say hello in one sentence.")
        print(f"Response: {response}")
