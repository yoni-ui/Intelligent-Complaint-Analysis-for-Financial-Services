"""
Google Gemini API client for LLM inference.

This module provides an interface to call Google's Gemini API.
"""

import requests
from typing import Optional
from src.config import (
    GOOGLE_API_KEY, GOOGLE_MODEL, LLM_TIMEOUT, 
    LLM_MAX_TOKENS, LLM_TEMPERATURE
)
from src.logger import logger
from src.utils import retry_on_failure

# Google Gemini API endpoint
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleGeminiClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize Google Gemini client.
        
        Args:
            api_key: Google API key (uses config default if None)
            model: Model name to use (uses config default if None)
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.model = model or GOOGLE_MODEL
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env file.")
        
        logger.info(f"Initialized Google Gemini client with model: {self.model}")
    
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
        url = f"{GEMINI_API_BASE_URL}/models/{self.model}:generateContent"
        
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        max_toks = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "key": self.api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_toks,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        logger.debug(f"Generating with Gemini model {self.model}, temperature={temp}, max_tokens={max_toks}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=LLM_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract text from Gemini response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        answer = parts[0]["text"]
                        logger.debug(f"Generated {len(answer)} characters")
                        return answer
            
            logger.warning("Unexpected response format from Gemini API")
            return ""
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from Gemini API: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - Status: {e.response.status_code}"
            logger.error(error_msg)
            raise RuntimeError(f"Gemini API error: {error_msg}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Gemini API request timed out after {LLM_TIMEOUT}s")
            raise TimeoutError("Gemini API request timed out. Try a shorter prompt or increase timeout.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request error: {e}")
            raise RuntimeError(f"Gemini API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Gemini generation: {e}")
            raise RuntimeError(f"Gemini error: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        try:
            # Simple check by trying to list models
            url = f"{GEMINI_API_BASE_URL}/models"
            params = {"key": self.api_key}
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Gemini API health check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models."""
        try:
            url = f"{GEMINI_API_BASE_URL}/models"
            params = {"key": self.api_key}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except:
            return []


def get_gemini_client(api_key: Optional[str] = None, model: Optional[str] = None) -> GoogleGeminiClient:
    """Factory function to get Gemini client.
    
    Args:
        api_key: Optional API key override
        model: Optional model name override
        
    Returns:
        Configured GoogleGeminiClient
    """
    return GoogleGeminiClient(api_key=api_key, model=model)


# Quick test
if __name__ == "__main__":
    try:
        client = GoogleGeminiClient()
        
        print(f"Gemini available: {client.is_available()}")
        print(f"Available models: {client.list_models()}")
        
        if client.is_available():
            print("\nTest generation:")
            response = client.generate("Say hello in one sentence.")
            print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
