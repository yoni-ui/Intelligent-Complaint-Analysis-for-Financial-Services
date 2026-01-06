"""
Local LLM backend using Ollama.

This module provides a simple interface to call local LLMs via Ollama's HTTP API.
"""

import requests
from typing import Optional

# Default Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b-instruct"


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
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> str:
        """Generate text completion from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response (not implemented yet)
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out. Try a shorter prompt or increase timeout.")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
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
