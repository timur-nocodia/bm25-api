import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import SparseTextEmbedding

# Configuration from environment variables
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE_DEFAULT", "256"))
THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", "1"))

app = FastAPI(
    title="BM25 Sparse Vector API",
    description="FastEmbed-powered BM25 sparse vector generation service",
    version="1.0.0"
)

# Initialize model with error handling
try:
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
except Exception as e:
    print(f"Failed to initialize BM25 model: {e}")
    raise

class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = BATCH_SIZE_DEFAULT
    threads: Optional[int] = THREADS_DEFAULT
    avg_len: Optional[float] = None  # For Qdrant BM25 compatibility

class SparseVec(BaseModel):
    indices: List[int]
    values: List[float]

class EmbedResponse(BaseModel):
    vectors: List[SparseVec]
    avg_len: Optional[float] = None  # For Qdrant BM25 integration

@app.get("/health")
def health():
    """Health check endpoint with model status"""
    try:
        # Quick model validation
        test_embedding = model.embed(["test"], batch_size=1, threads=1)
        next(test_embedding)  # Try to get first embedding
        return {
            "status": "healthy",
            "model": "Qdrant/bm25",
            "cache_path": os.getenv("FASTEMBED_CACHE", "/cache")
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model": "Qdrant/bm25"
        }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "BM25 Sparse Vector API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_vectors": "/sparse/bm25"
        },
        "documentation": "/docs"
    }

def calculate_avg_length(texts: List[str]) -> float:
    """Calculate average word length for BM25 parameter"""
    if not texts:
        return 0.0
    total_words = sum(len(text.split()) for text in texts)
    return total_words / len(texts)

@app.post("/sparse/bm25", response_model=EmbedResponse)
def sparse_bm25(req: EmbedRequest):
    """Generate BM25 sparse vectors for input texts"""
    try:
        embeddings = model.embed(
            req.texts,
            batch_size=req.batch_size,
            threads=req.threads
        )
        out = []
        for e in embeddings:
            out.append({"indices": e.indices.tolist(), "values": e.values.tolist()})
        
        # Calculate or use provided avg_len for Qdrant integration
        avg_len = req.avg_len if req.avg_len is not None else calculate_avg_length(req.texts)
        
        return {"vectors": out, "avg_len": avg_len}
    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {str(e)}")