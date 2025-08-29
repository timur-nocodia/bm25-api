# BM25 Sparse Vector API

Flexible HTTP service for generating BM25 sparse vectors and optional dense embeddings using FastEmbed. Supports three deployment modes: **Sparse-Only**, **Dense-Only**, and **Hybrid Search**.

## Features

- 🚀 **Multiple Modes**: Sparse-only (119MB), Dense-only, or Hybrid search
- 💾 **Memory Optimized**: Dense models only loaded when enabled
- 🌍 **Multilingual**: Excellent Russian, Chinese, English support
- 🔧 **Production Ready**: FastAPI with health checks and auto-restart
- ⚙️ **Fully Configurable**: All settings via environment variables
- 🔐 **Optional Security**: Bearer token authentication
- 🐳 **Docker Optimized**: Resource limits adjust to deployment mode

## Deployment Modes

### 🎯 **Mode 1: Sparse-Only (Default)**
**Memory**: ~119MB | **Startup**: 5-10s | **Use Case**: BM25/keyword search only

```bash
# Default mode - most memory efficient
docker-compose up -d

# Memory usage verified: 119MB
curl http://localhost:8080/health
```

### 🧠 **Mode 2: Dense-Only** 
**Memory**: 800MB-3GB | **Startup**: 30-60s | **Use Case**: Semantic search only

```bash
# Enable dense embeddings, disable sparse
ENABLE_DENSE=true DENSE_MODEL=BAAI/bge-small-en-v1.5 docker-compose up -d

# Available endpoints: /dense/embed
curl http://localhost:8080/dense/embed -H "Content-Type: application/json" -d '{"texts": ["Hello world"]}'
```

### 🔄 **Mode 3: Hybrid Search**
**Memory**: 800MB-3GB | **Startup**: 30-60s | **Use Case**: Best of both worlds

```bash
# Enable both sparse and dense
ENABLE_DENSE=true DENSE_MODEL=BAAI/bge-small-en-v1.5 docker-compose up -d

# Get both vector types in one call
curl http://localhost:8080/hybrid/embed -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "include_sparse": true, "include_dense": true}'
```

## API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{"ok": true}
```

### `POST /sparse/bm25`
Generate BM25 sparse vectors (always available)

### `POST /dense/embed`
Generate dense embeddings (only when `ENABLE_DENSE=true`)

### `POST /hybrid/embed`  
Generate both sparse and dense embeddings (only when `ENABLE_DENSE=true`)

**Request:**
```json
{
  "texts": ["text 1", "text 2"],
  "batch_size": 256,
  "threads": 1,
  "include_sparse": true,
  "include_dense": true
}
```

**Headers (when API key is configured):**
```
Authorization: Bearer your_api_key_here
```

**Response:**
```json
{
  "sparse_vectors": [
    {"indices": [123, 456], "values": [0.5, 0.3]}
  ],
  "dense_vectors": [
    [0.1, 0.2, 0.3, ...]
  ],
  "dense_model": "BAAI/bge-small-en-v1.5",
  "dense_dimensions": 384
}
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Sparse-only mode (default)
docker-compose up -d

# Enable dense embeddings
ENABLE_DENSE=true DENSE_MODEL=BAAI/bge-small-en-v1.5 docker-compose up -d
```

### Using Docker directly

```bash
# Build image
docker build -t bm25-api .

# Run container
docker run -d \
  --name bm25-api \
  -p 8080:8080 \
  -v bm25_cache:/cache \
  --restart unless-stopped \
  bm25-api
```

## Environment Variables Configuration

All settings are configurable via environment variables. Create a `.env` file or set them directly:

### 🔧 **Core Settings**

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DENSE` | `false` | Enable dense embeddings (`true`/`false`) |
| `DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | Dense model to use (when enabled) |
| `API_KEY` | - | Optional Bearer token for authentication |
| `HOST_PORT` | `8080` | External port mapping |
| `MEMORY_LIMIT` | `600M` | Container memory limit |

### 🎛️ **Application Settings**

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE_DEFAULT` | `256` | Default embedding batch size |
| `THREADS_DEFAULT` | `1` | Default thread count |
| `WORKERS` | `1` | Uvicorn worker processes |
| `LOG_LEVEL` | `info` | Log level (debug/info/warning/error) |

### 🧠 **Dense Model Options**

| Model | Dimensions | Size | Languages | Best For |
|-------|------------|------|-----------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | English | Fast, general purpose |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | English | Higher accuracy |
| `intfloat/multilingual-e5-large` | 1024 | 2.2GB | 100+ langs | Multilingual production |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90MB | English | Very fast |

### 📊 **Resource Planning**

| Mode | `ENABLE_DENSE` | `MEMORY_LIMIT` | Recommended Use |
|------|---------------|----------------|-----------------|
| **Sparse-Only** | `false` | `600M` | BM25/keyword search, minimal resources |
| **Small Dense** | `true` + bge-small | `1200M` | Hybrid search, balanced performance |
| **Large Dense** | `true` + e5-large | `3G` | Production multilingual, maximum accuracy |

### Docker Configuration
- `CONTAINER_NAME`: Container name (default: `bm25-api`)
- `HOST_PORT`: Host port mapping (default: `8080`)
- `CONTAINER_PORT`: Container internal port (default: `8080`)
- `CACHE_PATH`: Cache mount path (default: `/cache`)

### Application Settings
- `APP_HOST`: Application bind host (default: `0.0.0.0`)
- `WORKERS`: Number of worker processes (default: `1`)
- `LOG_LEVEL`: Log level - debug, info, warning, error (default: `info`)
- `BATCH_SIZE_DEFAULT`: Default batch size for embeddings (default: `256`)
- `THREADS_DEFAULT`: Default thread count (default: `1`)

### Performance Tuning
- `OMP_NUM_THREADS`: OpenMP threads limit (default: `1`)
- `OPENBLAS_NUM_THREADS`: OpenBLAS threads limit (default: `1`)
- `MKL_NUM_THREADS`: MKL threads limit (default: `1`)

### Cache Settings
- `FASTEMBED_CACHE`: Model cache directory (default: `/cache`)
- `HF_HOME`: Hugging Face cache directory (default: `/cache`)

### Resource Limits
- `CPU_LIMIT`: CPU cores limit (default: `1.5`)
- `MEMORY_LIMIT`: Memory limit (default: `600M`)
- `CPU_RESERVATION`: CPU cores reservation (default: `0.5`)
- `MEMORY_RESERVATION`: Memory reservation (default: `256M`)

### Restart & Health Check
- `RESTART_POLICY`: Docker restart policy (default: `always`)
- `RESTART_CONDITION`: Restart condition - none, on-failure, any (default: `any`)
- `RESTART_DELAY`: Delay between restart attempts (default: `5s`)
- `RESTART_MAX_ATTEMPTS`: Max restart attempts, 0=unlimited (default: `0`)
- `RESTART_WINDOW`: Time window for restart attempts (default: `120s`)
- `HEALTH_CHECK_INTERVAL`: Health check interval (default: `30s`)
- `HEALTH_CHECK_TIMEOUT`: Health check timeout (default: `5s`)
- `HEALTH_CHECK_RETRIES`: Health check retries (default: `5`)
- `HEALTH_CHECK_START_PERIOD`: Grace period before health checks (default: `10s`)

## 🚀 **Deployment Examples**

### **Example 1: Sparse-Only (Minimal Resources)**
```bash
# Perfect for: BM25/keyword search, production efficiency, limited resources
# Memory: ~119MB, Startup: 5-10 seconds

docker-compose up -d

# Verify deployment
curl http://localhost:8080/health
# Expected: {"status": "healthy", "models": {"sparse": {"status": "ready"}, "dense": {"status": "disabled"}}}
```

### **Example 2: Hybrid with Small Model (Balanced)**
```bash
# Perfect for: Development, balanced performance, English content
# Memory: ~1.2GB, Startup: 30-60 seconds

ENABLE_DENSE=true DENSE_MODEL=BAAI/bge-small-en-v1.5 MEMORY_LIMIT=1200M docker-compose up -d

# Test hybrid search
curl -X POST http://localhost:8080/hybrid/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "include_sparse": true, "include_dense": true}'
```

### **Example 3: Production Multilingual (Maximum Quality)**
```bash
# Perfect for: Production, multilingual support, maximum accuracy
# Memory: ~3GB, Startup: 60-120 seconds

ENABLE_DENSE=true \
DENSE_MODEL=intfloat/multilingual-e5-large \
MEMORY_LIMIT=3G \
CPU_LIMIT=4.0 \
docker-compose up -d

# Test with multiple languages
curl -X POST http://localhost:8080/dense/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Привет мир", "你好世界"]}'
```

### **Example 4: Secure Production Deployment**
```bash
# With API key authentication and optimized resources
API_KEY=your_secret_production_key_here \
ENABLE_DENSE=true \
DENSE_MODEL=BAAI/bge-small-en-v1.5 \
HOST_PORT=8081 \
MEMORY_LIMIT=1500M \
docker-compose up -d

# Authenticated request
curl -X POST http://localhost:8081/hybrid/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_production_key_here" \
  -d '{"texts": ["Secure hybrid search"]}'
```

### **Using .env File**

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
docker-compose up -d
```

**Example .env for Hybrid Mode:**
```env
# Enable hybrid search
ENABLE_DENSE=true
DENSE_MODEL=BAAI/bge-small-en-v1.5

# Security
API_KEY=my_secure_api_key_123

# Resources  
HOST_PORT=8081
MEMORY_LIMIT=1200M
CPU_LIMIT=2.0

# Performance
BATCH_SIZE_DEFAULT=128
WORKERS=1
```

## Integration with Qdrant

### Create Collection

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="my_collection",
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)
```

### Index Documents

```python
import requests

# Get sparse vectors (with optional API key)
headers = {"Authorization": "Bearer your_api_key"} if api_key else {}
response = requests.post(
    "http://localhost:8080/sparse/bm25",
    json={"texts": documents, "batch_size": 256},
    headers=headers
)
sparse_vectors = response.json()["vectors"]

# Upsert to Qdrant
points = []
for i, (doc, sparse) in enumerate(zip(documents, sparse_vectors)):
    points.append({
        "id": i,
        "vector": {"sparse": sparse},
        "payload": {"text": doc}
    })

client.upsert("my_collection", points)
```

## Integration with n8n

### HTTP Request Node Configuration

1. **Method:** POST
2. **URL:** `http://bm25-api:8080/sparse/bm25`
3. **Headers:** `Authorization: Bearer {{$env.API_KEY}}` (if API key is configured)
4. **Body Type:** JSON
5. **Body:**
```json
{
  "texts": {{$json["documents"]}},
  "batch_size": 256,
  "threads": 1
}
```

## Performance & Memory Usage

### Sparse-Only Mode (Default)
- **Memory:** ~400-600MB
- **Model:** BM25 (~30MB)
- **Startup:** Fast (~5-10 seconds)
- **No wasted resources** - Dense models NOT loaded when `ENABLE_DENSE=false`

### With Dense Embeddings
- **Small model (bge-small):** ~800-1200MB
- **Large model (e5-large):** ~2-3GB
- **Startup:** Slower (30-60 seconds first time)
- **Recommendation:** Set appropriate `MEMORY_LIMIT` in docker-compose

### Optimization Tips
```bash
# Sparse-only (minimal memory)
docker-compose up -d

# With small dense model
ENABLE_DENSE=true DENSE_MODEL=BAAI/bge-small-en-v1.5 MEMORY_LIMIT=1200M docker-compose up -d

# With large multilingual model
ENABLE_DENSE=true DENSE_MODEL=intfloat/multilingual-e5-large MEMORY_LIMIT=3G docker-compose up -d
```

## Requirements

- Docker & Docker Compose
- 600MB RAM minimum
- 1GB disk space (including model cache)

## 📝 **Quick Reference**

### **Choose Your Mode**

| Need | Mode | Command |
|------|------|---------|
| **Keyword search only** | Sparse | `docker-compose up -d` |
| **Semantic search only** | Dense | `ENABLE_DENSE=true docker-compose up -d` |
| **Best of both worlds** | Hybrid | `ENABLE_DENSE=true docker-compose up -d` |
| **Multilingual production** | Hybrid + Large | `ENABLE_DENSE=true DENSE_MODEL=intfloat/multilingual-e5-large MEMORY_LIMIT=3G docker-compose up -d` |

### **Key Endpoints**

| Endpoint | Available When | Purpose |
|----------|----------------|---------|
| `/sparse/bm25` | Always | BM25 sparse vectors for keyword search |
| `/dense/embed` | `ENABLE_DENSE=true` | Dense vectors for semantic search |
| `/hybrid/embed` | `ENABLE_DENSE=true` | Both vector types in one call |
| `/health` | Always | Check model status and health |

### **Memory Usage Guide**

- **119MB**: Sparse-only (verified)
- **~1.2GB**: Hybrid with small model
- **~3GB**: Hybrid with large multilingual model

### **Troubleshooting**

- **Out of Memory?** → Increase `MEMORY_LIMIT` or use smaller `DENSE_MODEL`
- **Slow startup?** → Normal for dense models (30-60s first time)
- **404 on dense endpoints?** → Check `ENABLE_DENSE=true` is set
- **401 errors?** → Add `Authorization: Bearer <API_KEY>` header

## License

MIT