#!/bin/bash
# Production startup script

set -e

echo "=========================================="
echo "CrediTrust Complaint Analyzer"
echo "Production Startup Script"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your configuration before continuing."
fi

# Check for vector store
if [ ! -f "vector_store/faiss_index.bin" ]; then
    echo "Warning: Vector store not found!"
    echo "Please run: python src/index_vector_store.py"
    exit 1
fi

# Check Ollama connection
echo "Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Warning: Ollama not responding at http://localhost:11434"
    echo "Please start Ollama: ollama serve"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start application
echo "Starting application..."
python app.py
