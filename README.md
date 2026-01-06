# Intelligent Complaint Analysis for Financial Services

A compact, local RAG (Retrieval-Augmented Generation) system to analyze CFPB customer complaints. It pairs a FAISS vector store of complaint text chunks with a local LLM (via Ollama) to provide grounded, explainable answers about complaint trends and issues.

## Highlights

- Fast semantic search over complaint chunks using FAISS
- Retrieval + generation pipeline for grounded answers
- Filter by product category (credit cards, loans, accounts, transfers)
- Local LLM inference via Ollama for privacy and offline use

## Repository Layout

```
app.py                     # Gradio UI application
data/                      # raw/ (input) and processed/ (preprocessed)
notebooks/                 # EDA and experiments
src/                       # Core modules: preprocessing, indexing, RAG, evaluation
vector_store/              # Persisted FAISS index (gitignored)
requirements.txt
```

## Quick Start

Prerequisites

- Python 3.10+
- Ollama running locally (for the generator LLM)

Install and run

```bash
# Create and activate a virtualenv
python -m venv .venv
.venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Pull the local LLM model for Ollama
#(run if using Ollama and you want Mistral)
ollama pull mistral:7b-instruct

# Preprocess raw complaints (place complaints.csv into data/raw/ first)
python src/preprocess.py

# Build the FAISS index
python src/index_vector_store.py

# Start the Gradio app
python app.py
```

Open http://127.0.0.1:7860 to use the chat UI.

## Data & Indexing Notes

- Expected workflow: download CFPB `complaints.csv` → `data/raw/` → `python src/preprocess.py` → `python src/index_vector_store.py`.
- Preprocessing filters target product categories and chunks complaint texts (default chunk size and overlap are defined in `src/index_vector_store.py`).
- Embeddings use `sentence-transformers/paraphrase-MiniLM-L3-v2` by default.

## RAG Pipeline

- Retriever: FAISS L2 search (top-k retrieval)
- Generator: LLM called via `src/llm/local_ollama.py` (default model `mistral:7b-instruct`)
- The system returns answers with supporting complaint excerpts for traceability.

## Common Commands

- Preprocess data: `python src/preprocess.py`
- Build index: `python src/index_vector_store.py`
- Run evaluation: `python -m src.evaluate`
- Start UI: `python app.py`

## Configuration

Key parameters live in `src/index_vector_store.py` and `src/rag.py` (chunk size, overlap, embedding model, and `DEFAULT_TOP_K`). Adjust them to trade off index size, retrieval granularity, and performance.

## License & Acknowledgments

MIT License

Thanks to CFPB (data), Sentence-Transformers (embeddings), Ollama (local LLMs), and FAISS (vector search).

---

If you'd like a shorter README, a version with expanded developer instructions, or a GitHub-ready summary for the project homepage, tell me which flavor and I will update it.
