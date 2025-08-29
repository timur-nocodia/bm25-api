# BGE-M3 Multi-Functional Embedding API

**The most advanced embedding service** - BGE-M3 is the first model supporting dense, sparse, and ColBERT retrieval in one unified framework with 100+ language support including excellent Russian capabilities.

## 🚀 Key Features

- **Multi-Functional**: Dense + Sparse + ColBERT embeddings in single model
- **Multilingual**: 100+ languages with excellent Russian support  
- **Long Context**: Up to 8,192 tokens input length
- **State-of-the-Art**: New benchmarks on MIRACL, MKQA, MLDR
- **Production Ready**: Docker deployment with resource optimization
- **Backward Compatible**: Drop-in replacement for existing BM25 API

## Quick Start

```bash
# Deploy BGE-M3 service
docker-compose -f docker-compose.bge-m3.yml up -d

# Check model status
curl http://localhost:8080/health

# Generate all three embedding types
curl -X POST http://localhost:8080/bge-m3/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Привет мир", "Hello world"],
    "return_dense": true,
    "return_sparse": true,
    "return_colbert_vecs": false
  }'
```

## API Endpoints

### `POST /bge-m3/embed` - Multi-Functional Embeddings
Generate dense, sparse, and/or ColBERT embeddings in one call:

```json
{
  "texts": ["document text", "another document"],
  "batch_size": 32,
  "return_dense": true,
  "return_sparse": true, 
  "return_colbert_vecs": false
}
```

**Response:**
```json
{
  "dense_vectors": [[0.1, 0.2, ...]], 
  "sparse_vectors": [{"indices": [123, 456], "values": [0.8, 0.6]}],
  "model": "BAAI/bge-m3",
  "dimensions": {"dense": 1024},
  "avg_len": 15.2
}
```

### `POST /sparse/bm25` - Backward Compatible
Original BM25 endpoint - automatically uses BGE-M3 sparse if available

### `POST /dense/embed` - Dense Only  
Generate dense embeddings (1024 dimensions)

### `GET /models/info` - Model Information
Detailed info about BGE-M3 capabilities and configuration

## BGE-M3 vs Other Models

| Feature | BGE-M3 | FastEmbed Dense | Traditional BM25 |
|---------|--------|----------------|------------------|
| Dense Search | ✅ (1024d) | ✅ (384-1024d) | ❌ |
| Sparse Search | ✅ (Native) | ❌ | ✅ |
| ColBERT Search | ✅ | ❌ | ❌ |
| Multilingual | ✅ (100+ langs) | ✅ (Limited) | ❌ |
| Russian Support | ✅ (Excellent) | ✅ (Good) | ✅ (Basic) |
| Max Length | 8,192 tokens | 512 tokens | Unlimited |
| Model Size | ~2GB | ~0.1-2GB | ~30MB |

## Qdrant Integration - Ultimate Hybrid Search

### Collection Setup for Triple-Vector Search
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

# Create collection with all three vector types
client.create_collection(
    collection_name="bge_m3_collection",
    vectors_config={
        "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }
    # ColBERT vectors stored as payload for fine-grained matching
)
```

### Indexing with BGE-M3
```python
import requests

# Generate all embedding types
response = requests.post(
    "http://localhost:8080/bge-m3/embed",
    json={
        "texts": documents,
        "return_dense": true,
        "return_sparse": true,
        "return_colbert_vecs": false,
        "batch_size": 32
    }
)
embeddings = response.json()

# Upload to Qdrant
points = []
for i, doc in enumerate(documents):
    points.append({
        "id": i,
        "vector": {
            "dense": embeddings["dense_vectors"][i],
            "sparse": embeddings["sparse_vectors"][i]
        },
        "payload": {
            "text": doc,
            "colbert": embeddings.get("colbert_vectors", [None])[i]  # Optional
        }
    })

client.upsert("bge_m3_collection", points)
```

### Advanced Hybrid Search with RRF
```python
# Query with BGE-M3
query_response = requests.post(
    "http://localhost:8080/bge-m3/embed", 
    json={"texts": [query], "return_dense": true, "return_sparse": true}
)
query_vectors = query_response.json()

# Hybrid search with RRF fusion
results = client.query_points(
    collection_name="bge_m3_collection",
    prefetch=[
        models.Prefetch(
            query=query_vectors["sparse_vectors"][0],
            using="sparse",
            limit=50
        ),
        models.Prefetch(
            query=query_vectors["dense_vectors"][0], 
            using="dense",
            limit=50
        )
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10
)
```

## Language Support Examples

BGE-M3 excels with multilingual content:

```bash
# Russian text
curl -X POST http://localhost:8080/bge-m3/embed \
  -d '{"texts": ["Поиск по документам на русском языке"], "return_dense": true}'

# Chinese text  
curl -X POST http://localhost:8080/bge-m3/embed \
  -d '{"texts": ["中文文档搜索"], "return_dense": true}'

# Mixed languages
curl -X POST http://localhost:8080/bge-m3/embed \
  -d '{"texts": ["English", "Русский", "中文"], "return_dense": true}'
```

## Performance Optimization

### Resource Requirements
- **CPU**: 2-4 cores recommended
- **RAM**: 4GB minimum (6GB+ for optimal performance)
- **Storage**: 3GB for model cache
- **Batch Size**: 16-32 (vs 256 for simple models)

### Configuration Options
```bash
# Memory optimization with FP16
USE_FP16=true docker-compose -f docker-compose.bge-m3.yml up -d

# CPU optimization
OMP_NUM_THREADS=4 docker-compose -f docker-compose.bge-m3.yml up -d

# Sparse-only mode for backward compatibility  
USE_BGE_M3=false docker-compose -f docker-compose.bge-m3.yml up -d
```

## n8n Integration

### HTTP Request Node Setup
1. **Method**: POST
2. **URL**: `http://bge-m3-api:8080/bge-m3/embed`
3. **Headers**: `Authorization: Bearer {{$env.API_KEY}}`
4. **Body**:
```json
{
  "texts": {{$json["documents"]}},
  "return_dense": true,
  "return_sparse": true,
  "batch_size": 16
}
```

### Response Processing
The response contains multiple embedding types:
- `dense_vectors`: 1024-dim arrays for semantic search
- `sparse_vectors`: Sparse format for keyword matching
- Use both in Qdrant for ultimate hybrid search

## Migration Guide

### From BM25-only Service
1. **Zero Downtime**: BGE-M3 includes sparse embeddings
2. **Enhanced Results**: Better multilingual sparse embeddings
3. **Add Dense Search**: Enable `return_dense: true` when ready
4. **Update Collection**: Add dense vector config to Qdrant

### From OpenAI Embeddings  
1. **Cost Savings**: No API costs after initial setup
2. **Better Multilingual**: Especially for Russian/Chinese
3. **Longer Context**: 8,192 vs 1,536 tokens  
4. **Sparse Bonus**: Get keyword matching for free

## Troubleshooting

### Common Issues
- **Out of Memory**: Reduce batch_size to 8-16, enable USE_FP16=true
- **Slow Startup**: Normal - model download takes 2-5 minutes first time
- **Import Error**: FlagEmbedding not installed - container handles this automatically

### Fallback Mode
If BGE-M3 fails to load, service automatically falls back to FastEmbed models for compatibility.

## License

MIT - BGE-M3 model is also MIT licensed