"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing

This script:
1. Loads the filtered complaints dataset
2. Chunks long narratives using RecursiveCharacterTextSplitter
3. Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
4. Stores embeddings + metadata in FAISS

Usage:
    python src/index_vector_store.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import hashlib

# Paths
FILTERED_DATA_PATH = Path('data/filtered_complaints.csv')
VECTOR_STORE_PATH = Path('vector_store')
FAISS_INDEX_PATH = VECTOR_STORE_PATH / 'faiss_index.bin'
METADATA_PATH = VECTOR_STORE_PATH / 'metadata.pkl'

# Chunking parameters
# Rationale:
# - chunk_size=500: Most complaint narratives are 100-500 words (~500-2500 chars).
#   500 chars keeps chunks focused while preserving context.
# - chunk_overlap=100: Ensures continuity between chunks, prevents cutting
#   mid-sentence issues from being lost.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Embedding model
# paraphrase-MiniLM-L3-v2: Very fast, lightweight (61MB), 384-dim embeddings
# Optimized for paraphrase detection and semantic similarity
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
EMBEDDING_DIM = 384

# Batch size for embedding (to manage memory)
BATCH_SIZE = 1000


def load_filtered_data(path: Path) -> pd.DataFrame:
    """Load the filtered complaints dataset."""
    print(f"Loading filtered data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} complaints")
    return df


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create text splitter with configured parameters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_complaints(df: pd.DataFrame, splitter: RecursiveCharacterTextSplitter) -> List[Dict[str, Any]]:
    """Chunk all complaint narratives."""
    print(f"Chunking {len(df):,} complaints...")
    chunks = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        narrative = row['narrative']
        if pd.isna(narrative) or not narrative.strip():
            continue
            
        text_chunks = splitter.split_text(narrative)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = hashlib.md5(f"{row['complaint_id']}_{i}".encode()).hexdigest()
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'metadata': {
                    'complaint_id': str(row['complaint_id']),
                    'product': row['product'],
                    'product_original': row['product_original'] if pd.notna(row['product_original']) else '',
                    'issue': row['issue'] if pd.notna(row['issue']) else '',
                    'company': row['company'] if pd.notna(row['company']) else '',
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
            })
    
    print(f"Created {len(chunks):,} chunks from {len(df):,} complaints")
    print(f"Average chunks per complaint: {len(chunks)/len(df):.2f}")
    
    return chunks


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the sentence transformer embedding model."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def build_faiss_index(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE
) -> tuple:
    """Generate embeddings and build FAISS index.
    
    Returns:
        Tuple of (faiss_index, metadata_list)
    """
    print(f"Building FAISS index for {len(chunks):,} chunks...")
    
    # Create FAISS index (L2 distance, can use IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Store metadata separately (FAISS only stores vectors)
    metadata_list = []
    
    # Process in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i + batch_size]
        texts = [c['text'] for c in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)
        
        # Store metadata
        for c in batch:
            metadata_list.append({
                'id': c['id'],
                'text': c['text'],
                **c['metadata']
            })
    
    # Concatenate all embeddings and add to index
    all_embeddings = np.vstack(all_embeddings).astype('float32')
    index.add(all_embeddings)
    
    print(f"FAISS index built with {index.ntotal:,} vectors")
    
    return index, metadata_list


def save_index(index: faiss.Index, metadata: List[Dict], index_path: Path, metadata_path: Path) -> None:
    """Save FAISS index and metadata to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, str(index_path))
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Saved successfully!")


def load_index(index_path: Path, metadata_path: Path) -> tuple:
    """Load FAISS index and metadata from disk."""
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata


def verify_index(index: faiss.Index, metadata: List[Dict], model: SentenceTransformer) -> None:
    """Verify the index by running a test query."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Test Query")
    print("=" * 60)
    
    print(f"Index contains {index.ntotal:,} vectors")
    
    test_query = "billing dispute credit card"
    print(f"\nTest query: '{test_query}'")
    
    # Embed query
    query_embedding = model.encode([test_query], convert_to_numpy=True).astype('float32')
    
    # Search
    k = 3
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nTop {k} results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        meta = metadata[idx]
        print(f"\n{i}. [Distance: {dist:.4f}] Product: {meta['product']}")
        print(f"   Issue: {meta['issue']}")
        print(f"   Text: {meta['text'][:200]}...")


def main():
    """Main indexing pipeline."""
    print("=" * 60)
    print("Task 2: Vector Store Indexing Pipeline (FAISS)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Chunk overlap: {CHUNK_OVERLAP}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Vector store path: {VECTOR_STORE_PATH}")
    print()
    
    # Load data
    df = load_filtered_data(FILTERED_DATA_PATH)
    
    # Create text splitter
    splitter = create_text_splitter()
    
    # Chunk complaints
    chunks = chunk_complaints(df, splitter)
    
    # Load embedding model
    model = load_embedding_model(EMBEDDING_MODEL)
    
    # Build FAISS index
    index, metadata = build_faiss_index(chunks, model)
    
    # Save index
    save_index(index, metadata, FAISS_INDEX_PATH, METADATA_PATH)
    
    # Verify
    verify_index(index, metadata, model)
    
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
