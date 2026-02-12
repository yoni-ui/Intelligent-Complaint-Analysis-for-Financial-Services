# Production Readiness Checklist

Use this checklist to ensure your deployment is production-ready.

## Pre-Deployment

### Environment Setup
- [ ] Created `.env` file from `.env.example`
- [ ] Configured all required environment variables
- [ ] Set `APP_DEBUG=False` for production
- [ ] Configured `ALLOWED_ORIGINS` for CORS
- [ ] Set appropriate `LOG_LEVEL` (INFO or WARNING)

### Data Preparation
- [ ] Preprocessed data (`python src/preprocess.py`)
- [ ] Built vector store index (`python src/index_vector_store.py`)
- [ ] Verified vector store files exist:
  - [ ] `vector_store/faiss_index.bin`
  - [ ] `vector_store/metadata.pkl`
- [ ] Verified data files exist:
  - [ ] `data/filtered_complaints.csv`

### Infrastructure
- [ ] Ollama service is running and accessible
- [ ] Required LLM model is pulled (`ollama pull mistral:7b-instruct`)
- [ ] Sufficient disk space for logs and cache
- [ ] Network ports are available (7860 for app, 11434 for Ollama)

### Security
- [ ] Reviewed and configured rate limiting
- [ ] Set secure `MAX_QUERY_LENGTH`
- [ ] Configured CORS origins
- [ ] File permissions are secure
- [ ] Using non-root user in Docker (if applicable)
- [ ] SSL/TLS configured (via reverse proxy)

## Deployment

### Docker Deployment
- [ ] Built Docker image successfully
- [ ] Tested docker-compose configuration
- [ ] Verified volume mounts are correct
- [ ] Health checks are working
- [ ] Logs are accessible

### Direct Python Deployment
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Application starts without errors
- [ ] Health checks pass

## Post-Deployment

### Verification
- [ ] Application is accessible at configured URL
- [ ] Health status shows "healthy"
- [ ] Test queries return expected results
- [ ] Error handling works correctly
- [ ] Rate limiting is active
- [ ] Caching is working

### Monitoring
- [ ] Logs are being written to `logs/app.log`
- [ ] Log rotation is working
- [ ] Metrics are being collected
- [ ] Health checks are accessible
- [ ] System resources are within limits

### Performance
- [ ] Query response times are acceptable
- [ ] Cache hit rate is reasonable
- [ ] No memory leaks observed
- [ ] System handles expected load

## Maintenance

### Regular Tasks
- [ ] Monitor logs for errors
- [ ] Review metrics regularly
- [ ] Check disk space usage
- [ ] Update dependencies periodically
- [ ] Backup vector store and data files
- [ ] Review and rotate logs

### Updates
- [ ] Test updates in staging first
- [ ] Backup before updates
- [ ] Update dependencies carefully
- [ ] Verify health after updates

## Troubleshooting

### Common Issues

**Vector store not found**
- Run: `python src/index_vector_store.py`
- Verify files exist in `vector_store/`

**Ollama connection failed**
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Verify `OLLAMA_BASE_URL` in `.env`

**High memory usage**
- Reduce `CACHE_MAX_SIZE`
- Reduce `MAX_TOP_K`
- Use smaller embedding model

**Slow queries**
- Enable caching (`ENABLE_CACHE=True`)
- Check Ollama performance
- Consider GPU acceleration

## Support Resources

- Logs: `logs/app.log`
- Health checks: Use health checker API
- Metrics: Use metrics collector API
- Documentation: See `DEPLOYMENT.md`
