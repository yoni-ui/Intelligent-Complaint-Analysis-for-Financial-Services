# Intelligent Complaint Analysis for Financial Services

A production-ready, local RAG (Retrieval-Augmented Generation) system to analyze CFPB customer complaints. It pairs a FAISS vector store of complaint text chunks with a local LLM (via Ollama) to provide grounded, explainable answers about complaint trends and issues.

## Highlights

- 🚀 **Production-Ready**: Comprehensive error handling, logging, caching, and monitoring
- 🔒 **Secure**: Input validation, rate limiting, and security best practices
- ⚡ **Fast**: Query caching and optimized retrieval
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 📊 **Monitoring**: Built-in health checks and metrics collection
- 🔍 **Fast semantic search** over complaint chunks using FAISS
- 🤖 **Retrieval + generation pipeline** for grounded answers
- 🎯 **Filter by product category** (credit cards, loans, accounts, transfers)
- 🔐 **Local LLM inference** via Ollama for privacy and offline use

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

### Prerequisites

- Python 3.10+
- Ollama running locally (for the generator LLM)
- Docker and Docker Compose (optional, for containerized deployment)

### Installation

#### Option 1: Direct Python Installation

```bash
# Create and activate a virtualenv
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy example and edit)
cp .env.example .env
# Edit .env with your settings

# Pull the local LLM model for Ollama
ollama pull mistral:7b-instruct

# Preprocess raw complaints (place complaints.csv into data/raw/ first)
python src/preprocess.py

# Build the FAISS index
python src/index_vector_store.py

# Start the Gradio app
python app.py
```

#### Option 2: Docker Compose (Recommended for Production)

```bash
# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services (includes Ollama)
docker-compose up -d

# View logs
docker-compose logs -f complaint-analyzer
```

#### Option 3: Using Startup Scripts

**Linux/Mac:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Windows:**
```cmd
scripts\start.bat
```

Open http://127.0.0.1:7860 to use the chat UI.

### Production Deployment

For detailed production deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Data & Indexing Notes

- Expected workflow: download CFPB `complaints.csv` → `data/raw/` → `python src/preprocess.py` → `python src/index_vector_store.py`.
- Preprocessing filters target product categories and chunks complaint texts (default chunk size and overlap are defined in `src/index_vector_store.py`).
- Embeddings use `sentence-transformers/paraphrase-MiniLM-L3-v2` by default.

## RAG Pipeline

- Retriever: FAISS L2 search (top-k retrieval)
- Generator: LLM called via `src/llm/local_ollama.py` (default model `mistral:7b-instruct`)
- The system returns answers with supporting complaint excerpts for traceability.

## Common Commands

### Development
- Preprocess data: `python src/preprocess.py`
- Build index: `python src/index_vector_store.py`
- Run evaluation: `python -m src.evaluate`
- Start UI: `python app.py`

### Production
- Start with Docker: `docker-compose up -d`
- View logs: `docker-compose logs -f`
- Check health: Access health status in UI (debug mode) or via health checker
- View metrics: Use metrics collector API

## Production Features

### Security
- ✅ Input validation and sanitization
- ✅ Rate limiting (configurable per IP)
- ✅ Query length limits
- ✅ Error handling with user-friendly messages
- ✅ Secure configuration management

### Performance
- ✅ Query result caching (configurable TTL)
- ✅ Retry logic with exponential backoff
- ✅ Optimized vector search
- ✅ Batch processing support

### Monitoring & Observability
- ✅ Comprehensive logging (file + console)
- ✅ Health check endpoints
- ✅ Metrics collection (query count, errors, cache hits)
- ✅ System status monitoring

### Reliability
- ✅ Graceful error handling
- ✅ Fallback mechanisms
- ✅ Connection retries
- ✅ Health checks for all components

## Architecture

```
┌─────────────────┐
│   Gradio UI     │
│   (app.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│  (src/rag.py)    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────┐
│ FAISS  │  │  Ollama  │
│ Index  │  │   LLM    │
└────────┘  └──────────┘
```

## Project Structure

```
.
├── app.py                 # Main application (Gradio UI)
├── src/
│   ├── config.py         # Configuration management
│   ├── logger.py         # Logging setup
│   ├── utils.py          # Utility functions
│   ├── cache.py          # Caching layer
│   ├── health.py         # Health checks
│   ├── metrics.py        # Metrics collection
│   ├── rag.py            # RAG pipeline
│   ├── preprocess.py     # Data preprocessing
│   ├── index_vector_store.py  # Vector store indexing
│   └── llm/
│       └── local_ollama.py    # Ollama client
├── data/                 # Data files
├── vector_store/         # FAISS index (generated)
├── logs/                 # Application logs
├── scripts/              # Startup scripts
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
└── DEPLOYMENT.md        # Production deployment guide
```

## Configuration

Configuration is managed through environment variables (see `.env.example`). Key settings include:

- **Application**: Host, port, debug mode, title
- **LLM**: Model selection, timeout, temperature
- **Security**: Rate limiting, query length limits
- **Performance**: Caching, batch sizes, retrieval parameters
- **Monitoring**: Logging levels, metrics collection

All configuration is centralized in `src/config.py` and can be overridden via environment variables.

### Key Configuration Files

- `.env` - Environment variables (create from `.env.example`)
- `src/config.py` - Configuration management
- `src/rag.py` - RAG pipeline parameters
- `src/index_vector_store.py` - Indexing parameters

## License & Acknowledgments

MIT License

Thanks to CFPB (data), Sentence-Transformers (embeddings), Ollama (local LLMs), and FAISS (vector search).

---

If you'd like a shorter README, a version with expanded developer instructions, or a GitHub-ready summary for the project homepage, tell me which flavor and I will update it.
