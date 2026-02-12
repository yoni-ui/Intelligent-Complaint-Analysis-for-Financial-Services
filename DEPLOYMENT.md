# Production Deployment Guide

This guide covers deploying the CrediTrust Complaint Analyzer chatbot to production.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- Ollama installed and running (for LLM inference)
- Sufficient disk space for vector store and data files

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your production settings.

### 2. Data Preparation

Ensure you have:
- Preprocessed data at `data/filtered_complaints.csv`
- Vector store index at `vector_store/faiss_index.bin` and `vector_store/metadata.pkl`

If not, run:
```bash
python src/preprocess.py
python src/index_vector_store.py
```

### 3. Start Ollama Service

```bash
# Start Ollama server
ollama serve

# Pull required model (in another terminal)
ollama pull mistral:7b-instruct
```

### 4. Run Application

#### Option A: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

#### Option B: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f complaint-analyzer
```

#### Option C: Docker Only

```bash
# Build image
docker build -t complaint-analyzer .

# Run container
docker run -d \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/vector_store:/app/vector_store:ro \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  complaint-analyzer
```

## Configuration

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Server host |
| `APP_PORT` | `7860` | Server port |
| `APP_DEBUG` | `False` | Enable debug mode |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service URL |
| `DEFAULT_LLM_MODEL` | `mistral:7b-instruct` | LLM model name |
| `ENABLE_RATE_LIMITING` | `True` | Enable rate limiting |
| `RATE_LIMIT_PER_MINUTE` | `30` | Rate limit per IP |
| `ENABLE_CACHE` | `True` | Enable query caching |
| `LOG_LEVEL` | `INFO` | Logging level |

### Production Recommendations

1. **Security**
   - Set `APP_DEBUG=False` in production
   - Configure `ALLOWED_ORIGINS` for CORS
   - Use reverse proxy (nginx/traefik) with SSL/TLS
   - Enable rate limiting

2. **Performance**
   - Enable caching (`ENABLE_CACHE=True`)
   - Adjust `CACHE_TTL_SECONDS` based on data freshness needs
   - Monitor `CACHE_MAX_SIZE` to prevent memory issues

3. **Monitoring**
   - Enable metrics (`ENABLE_METRICS=True`)
   - Set up log aggregation
   - Monitor health endpoints

## Health Checks

The application includes health check functionality:

```python
from src.health import health_checker

# Check overall health
health = health_checker.overall_health()
print(health['status'])  # 'healthy' or 'degraded'

# Check individual components
vector_store_health = health_checker.check_vector_store()
llm_health = health_checker.check_llm()
cache_health = health_checker.check_cache()
```

## Monitoring

### Metrics

Access metrics via the metrics collector:

```python
from src.metrics import metrics_collector

stats = metrics_collector.get_stats()
print(stats)
```

### Logs

Logs are written to:
- Console (stdout)
- File: `logs/app.log` (rotating, 10MB max, 5 backups)

### Health Endpoint

In debug mode, health status is available in the UI. For production, consider adding a dedicated health endpoint.

## Reverse Proxy Setup (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Scaling Considerations

1. **Horizontal Scaling**: The application is stateless (except cache). Multiple instances can run behind a load balancer.

2. **Cache**: Each instance has its own cache. Consider Redis for shared cache in multi-instance deployments.

3. **Vector Store**: The FAISS index is read-only and can be shared across instances via volume mounts.

4. **LLM Service**: Ollama can be scaled separately or use a shared Ollama instance.

## Troubleshooting

### Common Issues

1. **Vector store not found**
   - Ensure `vector_store/faiss_index.bin` exists
   - Run `python src/index_vector_store.py`

2. **Ollama connection failed**
   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Check `OLLAMA_BASE_URL` in `.env`

3. **Out of memory**
   - Reduce `CACHE_MAX_SIZE`
   - Use smaller embedding model
   - Reduce `MAX_TOP_K`

4. **Slow queries**
   - Enable caching
   - Check Ollama performance
   - Consider GPU acceleration for Ollama

## Security Checklist

- [ ] Set `APP_DEBUG=False`
- [ ] Configure `ALLOWED_ORIGINS`
- [ ] Enable rate limiting
- [ ] Use HTTPS (via reverse proxy)
- [ ] Restrict file permissions
- [ ] Use non-root user in Docker
- [ ] Keep dependencies updated
- [ ] Monitor logs for suspicious activity

## Backup and Recovery

### Critical Files to Backup

- `vector_store/faiss_index.bin`
- `vector_store/metadata.pkl`
- `data/filtered_complaints.csv`
- `.env` (securely)

### Recovery Procedure

1. Restore data files
2. Restore vector store files
3. Restore environment configuration
4. Restart services

## Performance Tuning

### For High Traffic

1. Increase `CACHE_MAX_SIZE`
2. Use faster embedding model (trade-off with quality)
3. Use GPU-accelerated Ollama
4. Implement Redis cache
5. Use CDN for static assets

### For Low Latency

1. Pre-warm cache with common queries
2. Use smaller `MAX_TOP_K`
3. Optimize chunk size during indexing
4. Use faster LLM model

## Support

For issues or questions:
1. Check logs: `logs/app.log`
2. Review health status
3. Check metrics for patterns
4. Consult error messages in UI
