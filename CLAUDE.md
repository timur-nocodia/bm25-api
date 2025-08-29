# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BM25 Sparse Vector API - A lightweight HTTP service for generating BM25 sparse vectors using FastEmbed, designed for integration with Qdrant vector database and n8n workflow automation.

## Key Commands

### Local Development
```bash
# Run locally without Docker
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Build and run with Docker Compose
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop service
docker-compose stop

# Run with custom port (8080 is often reserved)
HOST_PORT=8081 docker-compose up -d
```

### Testing the API
```bash
# Test without authentication
curl -X POST http://localhost:8080/sparse/bm25 \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test document"]}'

# Test with API key authentication
curl -X POST http://localhost:8080/sparse/bm25 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"texts": ["test document"]}'

# Health check
curl http://localhost:8080/health
```

### Deployment with API Key
```bash
# Start with API key protection
API_KEY=your_secret_key docker-compose up -d

# Or create .env file from template
cp .env.example .env
# Edit .env to set API_KEY and other settings
docker-compose up -d
```

## Architecture

### Core Components

1. **main.py** - FastAPI application with:
   - BM25 model initialization using `Qdrant/bm25` from FastEmbed
   - Bearer token authentication (optional via API_KEY env var)
   - `/sparse/bm25` endpoint for vector generation
   - `/health` endpoint with model validation
   - avg_len parameter support for Qdrant BM25 scoring

2. **Docker Setup**:
   - Multi-stage build with non-root user
   - Model caching in `/cache` volume
   - Health checks with auto-restart
   - Resource limits (600MB RAM, 1.5 CPU cores)
   - Never-die restart policy

3. **Environment Variables** - All 30+ settings exposed including:
   - `API_KEY` - Optional Bearer token authentication
   - `HOST_PORT` - External port (default 8080)
   - `BATCH_SIZE_DEFAULT` - Embedding batch size (default 256)
   - `THREADS_DEFAULT` - Thread count (default 1)
   - Resource limits, restart policies, health check intervals

## Integration Points

### Qdrant Integration
- Sparse vectors use `SparseVectorParams` with `Modifier.IDF`
- avg_len parameter passed during BM25 calculation (not stored)
- Hybrid search combines with dense vectors (e.g., OpenAI text-embedding-3-small)
- RRF fusion for ranking results

### n8n Workflow Integration
- HTTP Request node with Bearer token in headers: `{{$env.API_KEY}}`
- JSON body with texts array from document processing
- Sparse vector output format: `{"indices": [], "values": []}`

## Important Context

1. **BM25 Model**: Uses `Qdrant/bm25` (~30MB, cached after first download)

2. **avg_len Parameter**: 
   - Used DURING BM25 calculation, not stored in Qdrant
   - Represents average document length in corpus
   - API calculates it automatically or accepts provided value

3. **Authentication**:
   - Optional - when API_KEY not set, no auth required
   - When set, requires `Authorization: Bearer <token>` header
   - Implemented via FastAPI's HTTPBearer security

4. **Network Configuration**:
   - Service binds to 0.0.0.0 for container networking
   - Use actual IP (e.g., 192.168.x.x) when accessing from external services
   - Common issue: n8n on server can't reach localhost

5. **Common Deployment Scenarios**:
   - Portainer: Use docker-compose.yml directly
   - Production: Set API_KEY, adjust resource limits
   - Development: Run without API_KEY for easier testing