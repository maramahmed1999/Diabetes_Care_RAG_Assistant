"""
FastAPI backend for the Diabetes RAG system.

Exposes:
    GET  /health   -> quick readiness check
    POST /query    -> ask a question, get a grounded answer + sources

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.rag.generator import generate_rag_response
from backend.schemas import HealthResponse, QueryRequest, QueryResponse, Source

app = FastAPI(
    title="Diabetes RAG API",
    description="Retrieval-augmented question answering over diabetes care guidelines.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Reports whether the Qdrant index has been built yet."""
    return HealthResponse(
        status="ok",
        collection=config.COLLECTION_NAME,
        index_ready=config.QDRANT_PATH.exists(),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answers a question using the RAG pipeline (hybrid search + Ollama LLM)."""
    if not config.QDRANT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "The index hasn't been built yet. Run scripts/run_ingestion.py "
                "and scripts/build_index.py first."
            ),
        )

    try:
        answer, contexts = generate_rag_response(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        Source(
            text=ctx["text"],
            score=ctx["score"],
            document_title=ctx["metadata"].get("document_title"),
            organization=ctx["metadata"].get("organization"),
            source=ctx["metadata"].get("source"),
            page=ctx["metadata"].get("page"),
            chunk_id=ctx["metadata"].get("chunk_id"),
        )
        for ctx in contexts
    ]

    return QueryResponse(answer=answer, sources=sources)
