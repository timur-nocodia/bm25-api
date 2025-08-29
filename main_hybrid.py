import os
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastembed import SparseTextEmbedding, TextEmbedding

# Configuration from environment variables
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE_DEFAULT", "256"))
THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", "1"))
API_KEY = os.getenv("API_KEY", None)  # Optional API key for security
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")  # Default dense model

# Security setup
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key if configured"""
    if API_KEY is None or API_KEY == "":
        return True  # No API key required
    
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

app = FastAPI(
    title="Hybrid Vector API",
    description="FastEmbed-powered sparse (BM25) and dense vector generation service",
    version="2.0.0"
)

# Initialize models with error handling
try:
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print(f"Sparse BM25 model loaded successfully")
except Exception as e:
    print(f"Failed to initialize BM25 model: {e}")
    raise

try:
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    print(f"Dense model '{DENSE_MODEL}' loaded successfully")
except Exception as e:
    print(f"Failed to initialize dense model '{DENSE_MODEL}': {e}")
    print("Dense embeddings will not be available")
    dense_model = None

class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = BATCH_SIZE_DEFAULT
    threads: Optional[int] = THREADS_DEFAULT
    avg_len: Optional[float] = None  # For Qdrant BM25 compatibility

class HybridEmbedRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = BATCH_SIZE_DEFAULT
    threads: Optional[int] = THREADS_DEFAULT
    avg_len: Optional[float] = None
    include_sparse: bool = True
    include_dense: bool = True

class SparseVec(BaseModel):
    indices: List[int]
    values: List[float]

class SparseEmbedResponse(BaseModel):
    vectors: List[SparseVec]
    avg_len: Optional[float] = None

class DenseEmbedResponse(BaseModel):
    vectors: List[List[float]]
    model: str
    dimensions: int

class HybridEmbedResponse(BaseModel):
    sparse_vectors: Optional[List[SparseVec]] = None
    dense_vectors: Optional[List[List[float]]] = None
    avg_len: Optional[float] = None
    dense_model: Optional[str] = None
    dense_dimensions: Optional[int] = None

@app.get("/health")
def health():
    """Health check endpoint with model status"""
    status = {"status": "healthy", "models": {}}
    
    # Check sparse model
    try:
        test_sparse = sparse_model.embed(["test"], batch_size=1, threads=1)
        next(test_sparse)
        status["models"]["sparse"] = {
            "status": "ready",
            "model": "Qdrant/bm25"
        }
    except Exception as e:
        status["models"]["sparse"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check dense model
    if dense_model:
        try:
            test_dense = dense_model.embed(["test"], batch_size=1, threads=1)
            next(test_dense)
            status["models"]["dense"] = {
                "status": "ready",
                "model": DENSE_MODEL,
                "dimensions": dense_model.dim
            }
        except Exception as e:
            status["models"]["dense"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        status["models"]["dense"] = {"status": "not_loaded"}
    
    return status

@app.get("/")
def root():
    """Root endpoint with API information"""
    endpoints = {
        "health": "/health",
        "sparse_bm25": "/sparse/bm25",
        "list_dense_models": "/dense/models"
    }
    
    if dense_model:
        endpoints["dense_embeddings"] = "/dense/embed"
        endpoints["hybrid_embeddings"] = "/hybrid/embed"
    
    return {
        "service": "Hybrid Vector API",
        "version": "2.0.0",
        "endpoints": endpoints,
        "documentation": "/docs"
    }

def calculate_avg_length(texts: List[str]) -> float:
    """Calculate average word length for BM25 parameter"""
    if not texts:
        return 0.0
    total_words = sum(len(text.split()) for text in texts)
    return total_words / len(texts)

@app.post("/sparse/bm25", response_model=SparseEmbedResponse)
def sparse_bm25(req: EmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate BM25 sparse vectors for input texts"""
    try:
        embeddings = sparse_model.embed(
            req.texts,
            batch_size=req.batch_size,
            threads=req.threads
        )
        out = []
        for e in embeddings:
            out.append({"indices": e.indices.tolist(), "values": e.values.tolist()})
        
        avg_len = req.avg_len if req.avg_len is not None else calculate_avg_length(req.texts)
        
        return {"vectors": out, "avg_len": avg_len}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sparse embeddings: {str(e)}")

@app.get("/dense/models")
def list_dense_models(authorized: bool = Depends(verify_api_key)):
    """List available dense embedding models"""
    try:
        models = TextEmbedding.list_supported_models()
        # Return simplified model info
        return {
            "current_model": DENSE_MODEL if dense_model else None,
            "available_models": [
                {
                    "name": m["model"],
                    "dimensions": m["dim"],
                    "size_gb": m.get("size_in_GB", "N/A"),
                    "license": m.get("license", "N/A")
                }
                for m in models[:20]  # Return first 20 models
            ],
            "recommended": [
                "BAAI/bge-small-en-v1.5",
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-base-en-v1.5"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/dense/embed", response_model=DenseEmbedResponse)
def dense_embed(req: EmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate dense embeddings for input texts"""
    if not dense_model:
        raise HTTPException(status_code=503, detail="Dense model not available")
    
    try:
        embeddings = dense_model.embed(
            req.texts,
            batch_size=req.batch_size,
            threads=req.threads
        )
        vectors = [e.tolist() for e in embeddings]
        
        return {
            "vectors": vectors,
            "model": DENSE_MODEL,
            "dimensions": dense_model.dim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dense embeddings: {str(e)}")

@app.post("/hybrid/embed", response_model=HybridEmbedResponse)
def hybrid_embed(req: HybridEmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate both sparse and dense embeddings for hybrid search"""
    response = {}
    
    # Generate sparse embeddings if requested
    if req.include_sparse:
        try:
            sparse_embeddings = sparse_model.embed(
                req.texts,
                batch_size=req.batch_size,
                threads=req.threads
            )
            sparse_vecs = []
            for e in sparse_embeddings:
                sparse_vecs.append({"indices": e.indices.tolist(), "values": e.values.tolist()})
            
            response["sparse_vectors"] = sparse_vecs
            response["avg_len"] = req.avg_len if req.avg_len is not None else calculate_avg_length(req.texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate sparse embeddings: {str(e)}")
    
    # Generate dense embeddings if requested and available
    if req.include_dense:
        if not dense_model:
            if req.include_sparse:
                # Return partial response with only sparse
                return response
            else:
                raise HTTPException(status_code=503, detail="Dense model not available")
        
        try:
            dense_embeddings = dense_model.embed(
                req.texts,
                batch_size=req.batch_size,
                threads=req.threads
            )
            response["dense_vectors"] = [e.tolist() for e in dense_embeddings]
            response["dense_model"] = DENSE_MODEL
            response["dense_dimensions"] = dense_model.dim
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate dense embeddings: {str(e)}")
    
    return response