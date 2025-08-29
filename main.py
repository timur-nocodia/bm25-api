from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import SparseTextEmbedding

app = FastAPI(title="fastembed-bm25-api")
model = SparseTextEmbedding(model_name="Qdrant/bm25")

class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 256
    threads: Optional[int] = 1

class SparseVec(BaseModel):
    indices: List[int]
    values: List[float]

class EmbedResponse(BaseModel):
    vectors: List[SparseVec]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/sparse/bm25", response_model=EmbedResponse)
def sparse_bm25(req: EmbedRequest):
    embeddings = model.embed(
        req.texts,
        batch_size=req.batch_size,
        threads=req.threads
    )
    out = []
    for e in embeddings:
        out.append({"indices": e.indices.tolist(), "values": e.values.tolist()})
    return {"vectors": out}