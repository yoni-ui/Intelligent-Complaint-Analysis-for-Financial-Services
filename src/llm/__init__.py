"""LLM backends for the RAG pipeline."""

from .local_ollama import OllamaClient, get_llm_client

__all__ = ['OllamaClient', 'get_llm_client']
