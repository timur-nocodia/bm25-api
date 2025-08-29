import os
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastembed import SparseTextEmbedding, TextEmbedding

# Configuration from environment variables
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE_DEFAULT", "256"))
THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", "1"))
API_KEY = os.getenv("API_KEY", None)
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-m3")
USE_BGE_M3 = os.getenv("USE_BGE_M3", "true").lower() == "true"
USE_FP16 = os.getenv("USE_FP16", "false").lower() == "true"  # For BGE-M3

# Security setup
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key if configured"""
    if API_KEY is None or API_KEY == "":
        return True
    
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

app = FastAPI(
    title="BGE-M3 Hybrid Vector API",
    description="Multi-functional embedding service with BGE-M3 (dense, sparse, colbert) or FastEmbed models",
    version="3.0.0"
)

# Model initialization
bge_m3_model = None
fastembed_dense_model = None
sparse_model = None

# Try to initialize BGE-M3 if requested
if USE_BGE_M3 and DENSE_MODEL == "BAAI/bge-m3":
    try:
        from FlagEmbedding import BGEM3FlagModel
        print(f"Initializing BGE-M3 model (use_fp16={USE_FP16})...")
        bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=USE_FP16)
        print("BGE-M3 model loaded successfully")
    except ImportError:
        print("FlagEmbedding not installed. Install with: pip install FlagEmbedding")
        print("Falling back to FastEmbed models")
        USE_BGE_M3 = False
    except Exception as e:
        print(f"Failed to load BGE-M3: {e}")
        print("Falling back to FastEmbed models")
        USE_BGE_M3 = False

# Initialize FastEmbed models if not using BGE-M3
if not USE_BGE_M3 or not bge_m3_model:
    # Initialize sparse model (always use FastEmbed for this)
    try:
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("Sparse BM25 model loaded successfully")
    except Exception as e:
        print(f"Failed to initialize BM25 model: {e}")
        raise
    
    # Initialize dense model
    try:
        # Use a good multilingual model if BGE-M3 isn't available
        fallback_model = "intfloat/multilingual-e5-large" if "m3" in DENSE_MODEL.lower() else DENSE_MODEL
        fastembed_dense_model = TextEmbedding(model_name=fallback_model)
        print(f"Dense model '{fallback_model}' loaded successfully")
    except Exception as e:
        print(f"Failed to initialize dense model: {e}")
        fastembed_dense_model = None
else:
    # Still need sparse model for backward compatibility
    try:
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("Sparse BM25 model loaded for hybrid mode")
    except:
        pass

class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = BATCH_SIZE_DEFAULT
    threads: Optional[int] = THREADS_DEFAULT
    avg_len: Optional[float] = None

class BGEM3Request(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = BATCH_SIZE_DEFAULT
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert_vecs: bool = False
    avg_len: Optional[float] = None

class SparseVec(BaseModel):
    indices: List[int]
    values: List[float]

class BGEM3Response(BaseModel):
    dense_vectors: Optional[List[List[float]]] = None
    sparse_vectors: Optional[List[SparseVec]] = None
    colbert_vectors: Optional[List[List[List[float]]]] = None
    avg_len: Optional[float] = None
    model: str
    dimensions: Optional[Dict[str, int]] = None

@app.get("/health")
def health():
    """Health check endpoint with model status"""
    status = {"status": "healthy", "models": {}}
    
    if bge_m3_model:
        status["models"]["bge-m3"] = {
            "status": "ready",
            "capabilities": ["dense", "sparse", "colbert"],
            "multilingual": True,
            "max_length": 8192
        }
    
    if sparse_model:
        status["models"]["sparse"] = {
            "status": "ready",
            "model": "Qdrant/bm25"
        }
    
    if fastembed_dense_model:
        status["models"]["dense"] = {
            "status": "ready",
            "model": DENSE_MODEL if not USE_BGE_M3 else "fallback",
            "dimensions": fastembed_dense_model.dim
        }
    
    return status

@app.get("/")
def root():
    """Root endpoint with API information"""
    endpoints = {
        "health": "/health",
        "model_info": "/models/info"
    }
    
    if bge_m3_model:
        endpoints["bge_m3_embed"] = "/bge-m3/embed"
    
    if sparse_model:
        endpoints["sparse_bm25"] = "/sparse/bm25"
    
    if fastembed_dense_model:
        endpoints["dense_embed"] = "/dense/embed"
    
    return {
        "service": "BGE-M3 Hybrid Vector API",
        "version": "3.0.0",
        "active_models": {
            "bge_m3": bge_m3_model is not None,
            "sparse": sparse_model is not None,
            "dense": fastembed_dense_model is not None
        },
        "endpoints": endpoints,
        "documentation": "/docs"
    }

@app.get("/models/info")
def model_info(authorized: bool = Depends(verify_api_key)):
    """Get detailed information about available models"""
    info = {}
    
    if bge_m3_model:
        info["bge-m3"] = {
            "name": "BAAI/bge-m3",
            "type": "Multi-functional",
            "capabilities": {
                "dense": {
                    "dimensions": 1024,
                    "description": "Dense embeddings for semantic search"
                },
                "sparse": {
                    "description": "Lexical/sparse embeddings for keyword matching"
                },
                "colbert": {
                    "description": "Multi-vector embeddings for fine-grained matching"
                }
            },
            "languages": "100+ languages including Russian, Chinese, English",
            "max_length": 8192,
            "use_fp16": USE_FP16
        }
    
    if sparse_model:
        info["sparse"] = {
            "name": "Qdrant/bm25",
            "type": "Sparse/BM25",
            "description": "Traditional BM25 sparse vectors"
        }
    
    if fastembed_dense_model:
        info["dense"] = {
            "name": DENSE_MODEL if not USE_BGE_M3 else "intfloat/multilingual-e5-large",
            "type": "Dense",
            "dimensions": fastembed_dense_model.dim,
            "description": "Dense embeddings from FastEmbed"
        }
    
    return info

def calculate_avg_length(texts: List[str]) -> float:
    """Calculate average word length for BM25 parameter"""
    if not texts:
        return 0.0
    total_words = sum(len(text.split()) for text in texts)
    return total_words / len(texts)

@app.post("/bge-m3/embed", response_model=BGEM3Response)
def bge_m3_embed(req: BGEM3Request, authorized: bool = Depends(verify_api_key)):
    """Generate embeddings using BGE-M3 model (dense, sparse, and/or colbert)"""
    if not bge_m3_model:
        raise HTTPException(status_code=503, detail="BGE-M3 model not available")
    
    try:
        # BGE-M3 encoding
        embeddings = bge_m3_model.encode(
            req.texts,
            batch_size=req.batch_size,
            return_dense=req.return_dense,
            return_sparse=req.return_sparse,
            return_colbert_vecs=req.return_colbert_vecs
        )
        
        response = {
            "model": "BAAI/bge-m3",
            "dimensions": {}
        }
        
        # Process dense embeddings
        if req.return_dense and 'dense_vecs' in embeddings:
            dense_vecs = embeddings['dense_vecs']
            if isinstance(dense_vecs, np.ndarray):
                response["dense_vectors"] = dense_vecs.tolist()
            else:
                response["dense_vectors"] = [v.tolist() if hasattr(v, 'tolist') else v for v in dense_vecs]
            response["dimensions"]["dense"] = len(response["dense_vectors"][0]) if response["dense_vectors"] else 0
        
        # Process sparse embeddings
        if req.return_sparse and 'lexical_weights' in embeddings:
            sparse_vecs = []
            for weights in embeddings['lexical_weights']:
                indices = list(weights.keys())
                values = list(weights.values())
                sparse_vecs.append({"indices": indices, "values": values})
            response["sparse_vectors"] = sparse_vecs
            
            # Calculate avg_len for BM25 compatibility
            response["avg_len"] = req.avg_len if req.avg_len is not None else calculate_avg_length(req.texts)
        
        # Process ColBERT embeddings
        if req.return_colbert_vecs and 'colbert_vecs' in embeddings:
            colbert_vecs = embeddings['colbert_vecs']
            if isinstance(colbert_vecs, np.ndarray):
                response["colbert_vectors"] = colbert_vecs.tolist()
            else:
                response["colbert_vectors"] = [v.tolist() if hasattr(v, 'tolist') else v for v in colbert_vecs]
            if response["colbert_vectors"] and len(response["colbert_vectors"]) > 0:
                response["dimensions"]["colbert"] = len(response["colbert_vectors"][0][0]) if len(response["colbert_vectors"][0]) > 0 else 0
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate BGE-M3 embeddings: {str(e)}")

@app.post("/sparse/bm25")
def sparse_bm25(req: EmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate BM25 sparse vectors (backward compatible endpoint)"""
    if not sparse_model:
        # Try to use BGE-M3 sparse if available
        if bge_m3_model:
            bge_req = BGEM3Request(
                texts=req.texts,
                batch_size=req.batch_size,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
                avg_len=req.avg_len
            )
            result = bge_m3_embed(bge_req, authorized)
            return {
                "vectors": result.sparse_vectors,
                "avg_len": result.avg_len
            }
        else:
            raise HTTPException(status_code=503, detail="Sparse model not available")
    
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

@app.post("/dense/embed")
def dense_embed(req: EmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate dense embeddings"""
    # Try BGE-M3 first if available
    if bge_m3_model:
        bge_req = BGEM3Request(
            texts=req.texts,
            batch_size=req.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        result = bge_m3_embed(bge_req, authorized)
        return {
            "vectors": result.dense_vectors,
            "model": result.model,
            "dimensions": result.dimensions.get("dense", 1024)
        }
    
    if not fastembed_dense_model:
        raise HTTPException(status_code=503, detail="Dense model not available")
    
    try:
        embeddings = fastembed_dense_model.embed(
            req.texts,
            batch_size=req.batch_size,
            threads=req.threads
        )
        vectors = [e.tolist() for e in embeddings]
        
        return {
            "vectors": vectors,
            "model": DENSE_MODEL,
            "dimensions": fastembed_dense_model.dim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dense embeddings: {str(e)}")