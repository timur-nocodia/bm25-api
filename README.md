# BM25 Sparse Vector API

Lightweight HTTP service for generating BM25 sparse vectors using FastEmbed. Designed to work alongside Qdrant or any vector database supporting sparse vectors.

## Features

- 🚀 Fast BM25 sparse vector generation
- 🔧 RESTful API with FastAPI
- 🐳 Production-ready Docker setup
- 💾 Model caching for performance
- 🔒 Resource-limited deployment
- 🏥 Built-in health checks

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

### Environment Variables

- `FASTEMBED_CACHE`: Model cache directory (default: `/cache`)
- `HF_HOME`: Hugging Face cache directory (default: `/cache`)
- `OMP_NUM_THREADS`: OpenMP threads limit (default: `1`)
- `OPENBLAS_NUM_THREADS`: OpenBLAS threads limit (default: `1`)
- `MKL_NUM_THREADS`: MKL threads limit (default: `1`)

### Resource Limits

Default limits in `docker-compose.yml`:
- CPU: 1.5 cores max, 0.5 cores reserved
- Memory: 600MB max, 256MB reserved

Adjust based on your requirements.

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

# Get sparse vectors
response = requests.post(
    "http://localhost:8080/sparse/bm25",
    json={"texts": documents, "batch_size": 256}
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
3. **Body Type:** JSON
4. **Body:**
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