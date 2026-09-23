"""
Retrieval — hybrid (dense + BM25) search over the Qdrant collection.

This module owns the models and the Qdrant client as a singleton so
they're loaded once per process (important for a FastAPI server /
Streamlit app, unlike the notebook which just ran top to bottom).
"""

import shutil
from pathlib import Path
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from src import config

_lock = Lock()
_state = {
    "dense_model": None,
    "sparse_model": None,
    "client": None,
}


def _get_safe_qdrant_path() -> Path:
    """
    Qdrant's local (embedded) mode locks the storage directory.
    To allow the ingestion scripts and the running server to coexist
    without lock conflicts, the server reads from a throwaway copy
    of the index, just like the notebook did.
    """
    original = config.QDRANT_PATH
    safe_copy = config.QDRANT_PATH.parent / f"{config.QDRANT_PATH.name}_serving_copy"

    if not original.exists():
        raise FileNotFoundError(
            f"Qdrant index not found at {original}. "
            "Run the ingestion + indexing scripts first "
            "(scripts/run_ingestion.py and scripts/build_index.py)."
        )

    if safe_copy.exists():
        shutil.rmtree(safe_copy)

    shutil.copytree(original, safe_copy)

    return safe_copy


def get_models_and_client():
    """
    Lazily loads (once) the dense model, the sparse model and the
    Qdrant client, then reuses them for every subsequent query.
    """
    with _lock:
        if _state["client"] is None:
            _state["dense_model"] = SentenceTransformer(config.DENSE_MODEL_NAME)
            _state["sparse_model"] = SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)

            safe_path = _get_safe_qdrant_path()
            _state["client"] = QdrantClient(path=str(safe_path))

    return _state["dense_model"], _state["sparse_model"], _state["client"]


def hybrid_search(query_text: str, top_k: int = config.TOP_K) -> list[dict]:
    """
    Runs a hybrid dense + BM25 search (fused with RRF) and returns
    the top_k matching chunks with their metadata and score.
    """
    dense_model, sparse_model, client = get_models_and_client()

    query_dense = dense_model.encode(query_text, normalize_embeddings=True).tolist()
    query_sparse = list(sparse_model.query_embed(query_text))[0]

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using=config.DENSE_VECTOR_NAME, limit=5),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist(),
                ),
                using=config.SPARSE_VECTOR_NAME,
                limit=5,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )

    contexts = []
    for item in results.points:
        contexts.append({
            "text": item.payload["text"],
            "metadata": item.payload.get("metadata", {}),
            "score": item.score,
        })

    return contexts
