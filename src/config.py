"""
Configuration management for production deployment.

Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = PROJECT_ROOT / 'vector_store'
FAISS_INDEX_PATH = VECTOR_STORE_PATH / 'faiss_index.bin'
METADATA_PATH = VECTOR_STORE_PATH / 'metadata.pkl'
DATA_PATH = PROJECT_ROOT / 'data'
FILTERED_DATA_PATH = DATA_PATH / 'filtered_complaints.csv'

# Embedding configuration
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L3-v2')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '384'))

# Chunking configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1000'))

# LLM configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')  # 'ollama' or 'google'
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'mistral:7b-instruct')

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GOOGLE_MODEL = os.getenv('GOOGLE_MODEL', 'gemini-pro')

# Common LLM settings
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '120'))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '1024'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))

# RAG configuration
DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', '5'))
MAX_TOP_K = int(os.getenv('MAX_TOP_K', '20'))

# Application configuration
APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
# Use the platform-provided PORT if available (e.g., Render), else fall back to APP_PORT or 7860
APP_PORT = int(os.getenv('PORT', os.getenv('APP_PORT', '7860')))
APP_SHARE = os.getenv('APP_SHARE', 'False').lower() == 'true'
APP_DEBUG = os.getenv('APP_DEBUG', 'False').lower() == 'true'
APP_TITLE = os.getenv('APP_TITLE', 'CrediTrust Complaint Analyzer')

# Security configuration
ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'True').lower() == 'true'
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))
MAX_QUERY_LENGTH = int(os.getenv('MAX_QUERY_LENGTH', '1000'))
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

# Caching configuration
ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'True').lower() == 'true'
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', str(PROJECT_ROOT / 'logs' / 'app.log'))
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Monitoring configuration
ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'True').lower() == 'true'
METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))

# Health check configuration
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))

# Retry configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))

# Ensure directories exist
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)
(DATA_PATH / 'raw').mkdir(parents=True, exist_ok=True)
(DATA_PATH / 'processed').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
