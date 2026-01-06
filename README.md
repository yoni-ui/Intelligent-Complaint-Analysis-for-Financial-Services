# 🏦 Intelligent Complaint Analysis for Financial Services

A RAG-powered chatbot for analyzing CFPB customer complaints, built for CrediTrust Financial.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables intelligent analysis of customer complaints from the Consumer Financial Protection Bureau (CFPB) database. The system allows users to ask natural language questions about complaint patterns, issues, and trends across different financial products.

## Features

- **Semantic Search**: Query 1.6M+ complaint chunks using FAISS vector similarity
- **RAG Pipeline**: Combines retrieval with Mistral-7B LLM for grounded answers
- **Product Filtering**: Filter by credit cards, personal loans, savings accounts, or money transfers
- **Interactive UI**: Gradio-based chat interface with source document display
- **Local LLM**: Uses Ollama for privacy-preserving, offline inference

## Project Structure

```
├── app.py                      # Gradio UI application
├── src/
│   ├── preprocess.py           # Data preprocessing pipeline
│   ├── index_vector_store.py   # FAISS indexing script
│   ├── rag.py                  # RAG pipeline (retriever + generator)
│   ├── evaluate.py             # Qualitative evaluation script
│   └── llm/
│       └── local_ollama.py     # Ollama LLM client
├── notebooks/
│   └── 01_eda_preprocessing.ipynb  # EDA notebook
├── data/
│   ├── raw/                    # Raw CFPB data (gitignored)
│   └── filtered_complaints.csv # Preprocessed data
├── vector_store/               # FAISS index + metadata (gitignored)
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- ~6GB disk space for raw data
- ~2.5GB for FAISS index

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Intelligent-Complaint-Analysis-for-Financial-Services.git
cd Intelligent-Complaint-Analysis-for-Financial-Services

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Mistral model for Ollama
ollama pull mistral:7b-instruct
```

### Data Setup

1. Download CFPB complaints data from [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
2. Place `complaints.csv` in `data/raw/`
3. Run preprocessing:

```bash
python src/preprocess.py
```

### Build Vector Index

```bash
python src/index_vector_store.py
```

This will:
- Chunk 471k complaints into 1.6M text segments
- Generate embeddings using `paraphrase-MiniLM-L3-v2`
- Build and persist FAISS index (~23 minutes)

### Run the Application

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

## Technical Details

### Data Pipeline

| Stage | Input | Output |
|-------|-------|--------|
| Raw Data | 6GB complaints.csv | 4.7M rows |
| Filtering | Target 4 products | 471k complaints |
| Chunking | 500 chars, 100 overlap | 1.6M chunks |
| Embedding | paraphrase-MiniLM-L3-v2 | 384-dim vectors |

### Product Categories

- `credit_card` - Credit cards and prepaid cards
- `personal_loan` - Consumer loans, payday loans
- `savings_account` - Bank accounts, checking/savings
- `money_transfer` - Money transfers, virtual currency

### RAG Architecture

1. **Retriever**: FAISS L2 similarity search (top-k=5)
2. **Prompt**: Structured template with complaint excerpts
3. **Generator**: Mistral-7B-Instruct via Ollama

## Evaluation

Run qualitative evaluation with 10 test questions:

```bash
python -m src.evaluate
```

## Configuration

Key parameters in `src/index_vector_store.py`:
- `CHUNK_SIZE = 500` - Characters per chunk
- `CHUNK_OVERLAP = 100` - Overlap between chunks
- `EMBEDDING_MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'`

Key parameters in `src/rag.py`:
- `DEFAULT_TOP_K = 5` - Number of chunks to retrieve
- `llm_model = "mistral:7b-instruct"` - Ollama model

## License

MIT License

## Acknowledgments

- CFPB for the public complaints database
- Sentence-Transformers for embedding models
- Ollama for local LLM inference
- FAISS for efficient similarity search
