# BM25 Sparse Vector API

Lightweight HTTP service for generating BM25 sparse vectors using FastEmbed. Designed to work alongside Qdrant or any vector database supporting sparse vectors.

## Features

- 🚀 Fast BM25 sparse vector generation
- 🔧 RESTful API with FastAPI
- 🐳 Production-ready Docker setup
- 💾 Model caching for performance
- 🔒 Resource-limited deployment
- 🏥 Built-in health checks with auto-restart
- ⚙️ Fully configurable via environment variables
- 🔄 Never-die policy with unlimited restart attempts
- 👤 Non-root user execution for security

## Quick Start

```bash
# Clone repository
git clone <your-repo>
cd qdrant-bm25

# Start service
docker-compose up -d

# Check health
curl http://localhost:8080/health
```

## API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{"ok": true}
```

### `POST /sparse/bm25`
Generate BM25 sparse vectors

**Request:**
```json
{
  "texts": ["text 1", "text 2"],
  "batch_size": 256,
  "threads": 1
}
```

**Headers (when API key is configured):**
```
Authorization: Bearer your_api_key_here
```

**Response:**
```json
{
  "vectors": [
    {
      "indices": [123, 456],
      "values": [0.5, 0.3]
    }
  ]
}
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
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

## Configuration

All settings are configurable via environment variables. Create a `.env` file or modify `docker-compose.yml`:

### Security Settings
- `API_KEY`: Optional Bearer token for API authentication (default: none - no auth required)

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

### Example Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your settings
docker-compose up -d
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

## Performance

- First request downloads and caches the BM25 model (~30MB)
- Subsequent requests use cached model
- Batch processing for efficiency
- Thread limiting prevents resource exhaustion

## Requirements

- Docker & Docker Compose
- 600MB RAM minimum
- 1GB disk space (including model cache)

## License

MIT