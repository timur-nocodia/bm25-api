# Hybrid Vector API - Dense + Sparse Embeddings

Enhanced version that supports both **sparse (BM25)** and **dense embeddings** using FastEmbed for complete hybrid search capabilities.

## Features

- 🚀 **Dual Mode**: Sparse BM25 + Dense embeddings in one service
- 🔄 **Hybrid Endpoint**: Generate both vectors in single request
- 📊 **Multiple Dense Models**: Support for various embedding models
- 🎯 **Drop-in Replacement**: Compatible with existing BM25 API
- 🐳 **Production Ready**: Docker setup with resource limits
- 🔒 **API Security**: Optional Bearer token authentication

## Quick Start

```bash
# Using hybrid mode (both sparse and dense)
docker-compose -f docker-compose.hybrid.yml up -d

# Check available models
curl http://localhost:8080/dense/models

# Generate hybrid embeddings
curl -X POST http://localhost:8080/hybrid/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["document 1", "document 2"],
    "include_sparse": true,
    "include_dense": true
  }'
```

## API Endpoints

### `GET /health`
Health check with model status for both sparse and dense models

### `POST /sparse/bm25`
Original BM25 sparse vector endpoint (backward compatible)

### `GET /dense/models`
List available dense embedding models

### `POST /dense/embed`
Generate dense embeddings only
```json
{
  "texts": ["text 1", "text 2"],
  "batch_size": 256,
  "threads": 1
}
```

### `POST /hybrid/embed`
Generate both sparse and dense embeddings
```json
{
  "texts": ["text 1", "text 2"],
  "include_sparse": true,
  "include_dense": true,
  "batch_size": 256,
  "threads": 1,
  "avg_len": null
}
```

Response:
```json
{
  "sparse_vectors": [
    {"indices": [123, 456], "values": [0.5, 0.3]}
  ],
  "dense_vectors": [
    [0.1, 0.2, 0.3, ...]
  ],
  "avg_len": 10.5,
  "dense_model": "BAAI/bge-small-en-v1.5",
  "dense_dimensions": 384
}
```

## Dense Model Selection

Configure via `DENSE_MODEL` environment variable:

```bash
# Small & Fast (384 dims, 0.13 GB)
DENSE_MODEL=BAAI/bge-small-en-v1.5

# Very Fast (384 dims, 0.09 GB)  
DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Higher Accuracy (768 dims, 0.44 GB)
DENSE_MODEL=BAAI/bge-base-en-v1.5

# Multilingual (1024 dims, 2.24 GB)
DENSE_MODEL=intfloat/multilingual-e5-large
```

## Qdrant Integration for Hybrid Search

### Create Collection with Both Vector Types
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="hybrid_collection",
    vectors_config={
        "dense": models.VectorParams(
            size=384,  # Match your dense model dimensions
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

### Index Documents with Hybrid Embeddings
```python
import requests

# Get hybrid embeddings
response = requests.post(
    "http://localhost:8080/hybrid/embed",
    json={
        "texts": documents,
        "include_sparse": true,
        "include_dense": true
    }
)
result = response.json()

# Prepare points for Qdrant
points = []
for i, doc in enumerate(documents):
    points.append({
        "id": i,
        "vector": {
            "dense": result["dense_vectors"][i],
            "sparse": result["sparse_vectors"][i]
        },
        "payload": {"text": doc}
    })

client.upsert("hybrid_collection", points)
```

### Hybrid Search with RRF Fusion
```python
# Search with both vectors
query_response = requests.post(
    "http://localhost:8080/hybrid/embed",
    json={"texts": [query_text]}
)
query_vectors = query_response.json()

# Qdrant hybrid search
search_result = client.query_points(
    collection_name="hybrid_collection",
    prefetch=[
        models.Prefetch(
            query=query_vectors["sparse_vectors"][0],
            using="sparse",
            limit=20
        ),
        models.Prefetch(
            query=query_vectors["dense_vectors"][0],
            using="dense",
            limit=20
        )
    ],
    query=models.FusionQuery(
        fusion=models.Fusion.RRF
    ),
    limit=10
)
```

## n8n Integration

### HTTP Request Node for Hybrid Embeddings
1. **Method:** POST
2. **URL:** `http://hybrid-api:8080/hybrid/embed`
3. **Headers:** `Authorization: Bearer {{$env.API_KEY}}` (if configured)
4. **Body:**
```json
{
  "texts": {{$json["documents"]}},
  "include_sparse": true,
  "include_dense": true
}
```

## Performance Considerations

- **First Request**: Downloads both BM25 (~30MB) and dense model (varies by model)
- **Memory Usage**: 
  - Sparse only: ~600MB
  - Hybrid (small model): ~1.2GB
  - Hybrid (large model): ~2.5GB
- **Speed**: Dense embeddings add ~20-50ms per document depending on model

## Configuration

### Environment Variables
```bash
# Mode selection
MODE=hybrid  # or "sparse" for BM25-only

# Dense model selection
DENSE_MODEL=BAAI/bge-small-en-v1.5

# Resource limits (adjust based on model)
MEMORY_LIMIT=1200M  # Increase for larger models
CPU_LIMIT=2.0

# API security
API_KEY=your_secret_key
```

### Deployment Examples

```bash
# Sparse-only mode (backward compatible)
MODE=sparse docker-compose -f docker-compose.hybrid.yml up -d

# Hybrid with small model (recommended for most cases)
DENSE_MODEL=BAAI/bge-small-en-v1.5 docker-compose -f docker-compose.hybrid.yml up -d

# Hybrid with multilingual support
DENSE_MODEL=intfloat/multilingual-e5-large MEMORY_LIMIT=2500M docker-compose -f docker-compose.hybrid.yml up -d
```

## Migration from Sparse-Only

The hybrid API is fully backward compatible:
1. `/sparse/bm25` endpoint remains unchanged
2. Existing integrations continue to work
3. Add dense embeddings when ready by using `/hybrid/embed`
4. No changes needed to existing Qdrant collections (add dense vector config when ready)

## License

MIT