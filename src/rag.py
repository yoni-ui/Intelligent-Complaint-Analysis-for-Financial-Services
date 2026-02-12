"""
Task 3: RAG Pipeline - Retrieval Augmented Generation

This module implements the core RAG logic:
1. Retriever: Query FAISS index for relevant complaint chunks
2. Prompt Engineering: Build context-aware prompts
3. Generator: Use LLM to generate grounded answers

Usage:
    from src.rag import RAGPipeline
    
    rag = RAGPipeline()
    answer, sources = rag.answer("Why are people unhappy with credit cards?")
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

from src.llm.factory import get_llm_client
from src.config import (
    FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, DEFAULT_TOP_K,
    MAX_TOP_K, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
)
from src.logger import logger
from src.cache import query_cache
from src.utils import retry_on_failure, format_error_message

# Prompt template
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on real complaint data.

INSTRUCTIONS:
- Use ONLY the provided complaint excerpts to formulate your answer
- If the context doesn't contain enough information to answer, say "I don't have enough information in the complaints to answer this question."
- Be concise and specific
- When relevant, mention which product categories the issues relate to
- Summarize common themes if multiple complaints mention similar issues

COMPLAINT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


class Retriever:
    """Handles semantic search over the FAISS index."""
    
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """Initialize retriever with FAISS index and embedding model."""
        logger.info(f"Initializing Retriever with model: {embedding_model}")
        try:
            self.index = self._load_index(index_path)
            self.metadata = self._load_metadata(metadata_path)
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"Retriever initialized successfully. Index size: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to initialize Retriever: {e}")
            raise
        
    @retry_on_failure(max_retries=3)
    def _load_index(self, path: Path) -> faiss.Index:
        """Load FAISS index from disk."""
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found at {path}. Run index_vector_store.py first.")
        logger.debug(f"Loading FAISS index from {path}")
        return faiss.read_index(str(path))
    
    @retry_on_failure(max_retries=3)
    def _load_metadata(self, path: Path) -> List[Dict]:
        """Load metadata from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found at {path}. Run index_vector_store.py first.")
        logger.debug(f"Loading metadata from {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @retry_on_failure(max_retries=2)
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        product_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of results to return (clamped to MAX_TOP_K)
            product_filter: Optional product category to filter by
            
        Returns:
            List of chunk dictionaries with text, metadata, and distance
        """
        # Clamp top_k to prevent excessive retrieval
        top_k = min(top_k, MAX_TOP_K)
        
        logger.debug(f"Retrieving top {top_k} chunks for query: {query[:50]}...")
        
        try:
            # Embed query
            query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
            
            # Search (retrieve more if filtering)
            search_k = top_k * 3 if product_filter else top_k
            distances, indices = self.index.search(query_embedding, search_k)
            
            # Build results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                    
                meta = self.metadata[idx]
                
                # Apply product filter if specified
                if product_filter and meta.get('product') != product_filter:
                    continue
                
                results.append({
                    'text': meta['text'],
                    'complaint_id': meta['complaint_id'],
                    'product': meta['product'],
                    'issue': meta.get('issue', ''),
                    'company': meta.get('company', ''),
                    'distance': float(dist)
                })
                
                if len(results) >= top_k:
                    break
            
            logger.info(f"Retrieved {len(results)} chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        llm_model: str = None,
        top_k: int = DEFAULT_TOP_K
    ):
        """Initialize RAG pipeline.
        
        Args:
            llm_model: Ollama model name (uses config default if None)
            top_k: Default number of chunks to retrieve
        """
        logger.info("Initializing RAG Pipeline...")
        self.retriever = Retriever()
        self.llm = get_llm_client(model=llm_model)
        self.top_k = min(top_k, MAX_TOP_K)
        logger.info(f"RAG Pipeline initialized with model: {self.llm.model}, top_k: {self.top_k}")
        
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Complaint {i}] Product: {chunk['product']} | Issue: {chunk['issue']}\n"
                f"{chunk['text']}"
            )
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the full prompt for the LLM."""
        return PROMPT_TEMPLATE.format(context=context, question=question)
    
    def answer(
        self,
        question: str,
        product_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        temperature: float = None,
        use_cache: bool = True
    ) -> Tuple[str, List[Dict]]:
        """Answer a question using RAG.
        
        Args:
            question: User's question
            product_filter: Optional product category filter
            top_k: Number of chunks to retrieve (overrides default)
            temperature: LLM temperature (uses config default if None)
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (answer_text, source_chunks)
        """
        # Check cache first
        if use_cache:
            cache_key = f"{question}|{product_filter}|{top_k or self.top_k}"
            cached_result = query_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result['answer'], cached_result['sources']
        
        try:
            k = min(top_k or self.top_k, MAX_TOP_K)
            temp = temperature if temperature is not None else LLM_TEMPERATURE
            
            logger.info(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant chunks
            chunks = self.retriever.retrieve(
                query=question,
                top_k=k,
                product_filter=product_filter
            )
            
            if not chunks:
                logger.warning("No relevant chunks found for query")
                return "No relevant complaints found for your query.", []
            
            # Build prompt
            context = self._build_context(chunks)
            prompt = self._build_prompt(question, context)
            
            # Generate answer with retry
            try:
                answer = self.llm.generate(
                    prompt,
                    temperature=temp,
                    max_tokens=LLM_MAX_TOKENS
                )
                answer = answer.strip()
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                # Fallback: return summary of retrieved chunks
                answer = self._fallback_answer(chunks)
            
            result = (answer, chunks)
            
            # Cache result
            if use_cache:
                query_cache.set(cache_key, {
                    'answer': answer,
                    'sources': chunks
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            error_msg = format_error_message(e, include_details=False)
            return error_msg, []
    
    def _fallback_answer(self, chunks: List[Dict]) -> str:
        """Generate fallback answer from chunks when LLM fails.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Fallback answer text
        """
        products = set(chunk['product'] for chunk in chunks)
        issues = [chunk['issue'] for chunk in chunks if chunk.get('issue')]
        
        answer_parts = [
            f"Found {len(chunks)} relevant complaint(s) related to your query."
        ]
        
        if products:
            answer_parts.append(f"Products involved: {', '.join(products)}")
        
        if issues:
            unique_issues = list(set(issues))[:5]
            answer_parts.append(f"Common issues: {', '.join(unique_issues)}")
        
        return " ".join(answer_parts)
    
    def retrieve_only(
        self,
        question: str,
        product_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve chunks without generating an answer (for debugging)."""
        k = top_k or self.top_k
        return self.retriever.retrieve(question, top_k=k, product_filter=product_filter)


# Quick test
if __name__ == "__main__":
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    print("\nTesting retrieval only:")
    chunks = rag.retrieve_only("billing dispute credit card", top_k=3)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{i}. Product: {chunk['product']} | Issue: {chunk['issue']}")
        print(f"   Text: {chunk['text'][:150]}...")
    
    print("\n" + "="*60)
    print("Testing full RAG pipeline:")
    print("="*60)
    
    question = "What are the main complaints about credit cards?"
    print(f"\nQuestion: {question}")
    
    answer, sources = rag.answer(question)
    print(f"\nAnswer:\n{answer}")
    
    print(f"\nSources ({len(sources)} chunks used):")
    for i, src in enumerate(sources, 1):
        print(f"  {i}. [{src['product']}] {src['issue']}")
