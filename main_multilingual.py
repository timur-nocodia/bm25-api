import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastembed import SparseTextEmbedding, TextEmbedding

# Configuration from environment variables
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE_DEFAULT", "64"))
THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", "1"))
API_KEY = os.getenv("API_KEY", None)
# Use best multilingual model available in FastEmbed
DENSE_MODEL = os.getenv("DENSE_MODEL", "intfloat/multilingual-e5-large")

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
    title="Multilingual Hybrid Vector API",
    description="FastEmbed-powered multilingual dense + sparse vector generation",
    version="2.1.0"
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
    print(f"Model dimensions: {dense_model.dim}")
except Exception as e:
    print(f"Failed to initialize dense model '{DENSE_MODEL}': {e}")
    dense_model = None

class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = BATCH_SIZE_DEFAULT
    threads: Optional[int] = THREADS_DEFAULT
    avg_len: Optional[float] = None

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

class MultilingualResponse(BaseModel):
    sparse_vectors: Optional[List[SparseVec]] = None
    dense_vectors: Optional[List[List[float]]] = None
    avg_len: Optional[float] = None
    dense_model: str
    dense_dimensions: int
    languages_supported: List[str]

@app.get("/health")
def health():
    """Health check endpoint with multilingual model status"""
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
                "dimensions": dense_model.dim,
                "multilingual": True,
                "languages": ["English", "Russian", "Chinese", "Spanish", "French", "German", "Arabic", "Japanese", "Korean", "Hindi", "Portuguese", "Italian", "Dutch", "Polish", "Turkish", "Czech", "Ukrainian", "Hebrew", "Thai", "Vietnamese", "Indonesian", "Swedish", "Norwegian", "Danish", "Finnish", "Greek", "Hungarian", "Romanian", "Bulgarian", "Croatian", "Slovak", "Slovenian", "Lithuanian", "Latvian", "Estonian"][:20]  # Top 20
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
    return {
        "service": "Multilingual Hybrid Vector API",
        "version": "2.1.0",
        "features": {
            "multilingual": True,
            "languages_supported": 100,
            "hybrid_search": True,
            "russian_support": "excellent"
        },
        "models": {
            "dense": DENSE_MODEL,
            "sparse": "Qdrant/bm25"
        },
        "endpoints": {
            "health": "/health",
            "sparse_bm25": "/sparse/bm25",
            "dense_embed": "/dense/embed", 
            "hybrid_embed": "/hybrid/embed",
            "test_multilingual": "/test/multilingual"
        },
        "documentation": "/docs"
    }

def calculate_avg_length(texts: List[str]) -> float:
    """Calculate average word length for BM25 parameter"""
    if not texts:
        return 0.0
    total_words = sum(len(text.split()) for text in texts)
    return total_words / len(texts)

@app.post("/sparse/bm25")
def sparse_bm25(req: EmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate BM25 sparse vectors (backward compatible)"""
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
    """Generate multilingual dense embeddings"""
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
            "dimensions": dense_model.dim,
            "multilingual": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dense embeddings: {str(e)}")

@app.post("/hybrid/embed", response_model=MultilingualResponse)
def hybrid_embed(req: HybridEmbedRequest, authorized: bool = Depends(verify_api_key)):
    """Generate both sparse and dense embeddings for multilingual hybrid search"""
    response_data = {
        "dense_model": DENSE_MODEL,
        "dense_dimensions": dense_model.dim if dense_model else 0,
        "languages_supported": ["en", "ru", "zh", "es", "fr", "de", "ar", "ja", "ko", "hi", "pt", "it", "nl", "pl", "tr", "cs", "uk", "he", "th", "vi"]
    }
    
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
            
            response_data["sparse_vectors"] = sparse_vecs
            response_data["avg_len"] = req.avg_len if req.avg_len is not None else calculate_avg_length(req.texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate sparse embeddings: {str(e)}")
    
    # Generate dense embeddings if requested and available
    if req.include_dense:
        if not dense_model:
            if req.include_sparse:
                return response_data  # Return partial response with only sparse
            else:
                raise HTTPException(status_code=503, detail="Dense model not available")
        
        try:
            dense_embeddings = dense_model.embed(
                req.texts,
                batch_size=req.batch_size,
                threads=req.threads
            )
            response_data["dense_vectors"] = [e.tolist() for e in dense_embeddings]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate dense embeddings: {str(e)}")
    
    return response_data

@app.get("/test/multilingual")
def test_multilingual(authorized: bool = Depends(verify_api_key)):
    """Test endpoint with multilingual examples"""
    test_texts = {
        "english": "Hello world, this is a test document",
        "russian": "Привет мир, это тестовый документ", 
        "chinese": "你好世界，这是一个测试文档",
        "spanish": "Hola mundo, este es un documento de prueba",
        "french": "Bonjour le monde, ceci est un document de test",
        "german": "Hallo Welt, dies ist ein Testdokument"
    }
    
    if not dense_model:
        return {
            "message": "Dense model not available for multilingual testing",
            "test_texts": test_texts,
            "sparse_available": True
        }
    
    try:
        # Test with all languages
        texts = list(test_texts.values())
        embeddings = dense_model.embed(texts, batch_size=len(texts), threads=1)
        vectors = [e.tolist()[:5] for e in embeddings]  # Show first 5 dimensions
        
        results = {}
        for i, (lang, text) in enumerate(test_texts.items()):
            results[lang] = {
                "text": text,
                "embedding_preview": vectors[i],
                "dimensions": len(embeddings[i])
            }
        
        return {
            "model": DENSE_MODEL,
            "status": "success",
            "results": results,
            "note": "All texts successfully embedded with multilingual model"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "test_texts": test_texts
        }