"""
Phase 4 — Embedding & indexing.

Loads the chunks produced by the ingestion pipeline, embeds them with
a dense model (BGE) and a sparse BM25 model, and upserts everything
into a local Qdrant collection configured for hybrid search.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from src import config


def load_chunks(file_path: Path) -> list[dict]:
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def get_chunk_text(chunk: dict) -> str:
    text = chunk.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def build_index(
    input_file: Path | None = None,
    qdrant_path: Path | None = None,
    collection_name: str | None = None,
    verbose: bool = True,
) -> int:
    """
    Embeds every chunk in `input_file` and (re)builds the Qdrant
    collection used for retrieval. Returns the number of vectors
    written.
    """
    input_file = input_file or config.CHUNKS_FILE
    qdrant_path = qdrant_path or config.QDRANT_PATH
    collection_name = collection_name or config.COLLECTION_NAME

    if verbose:
        print("=" * 60)
        print("Loading chunks...")
        print("=" * 60)

    chunks = load_chunks(input_file)
    if verbose:
        print(f"Loaded {len(chunks)} chunks")

    if not chunks:
        raise ValueError("No chunks found. Run the ingestion pipeline first.")

    texts = []
    valid_chunks = []
    for chunk in chunks:
        text = get_chunk_text(chunk)
        if not text:
            continue
        texts.append(text)
        valid_chunks.append(chunk)

    chunks = valid_chunks
    if verbose:
        print(f"Valid chunks: {len(chunks)}")

    # ----------------------------------------------------------
    # Load models (dense + sparse BM25)
    # ----------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Loading embedding models (dense & BM25)...")
        print("=" * 60)

    dense_model = SentenceTransformer(config.DENSE_MODEL_NAME)
    dimension = dense_model.get_sentence_embedding_dimension()

    sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)

    # ----------------------------------------------------------
    # Initialize Qdrant client + collection
    # ----------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Initializing Qdrant and uploading...")
        print("=" * 60)

    qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            config.DENSE_VECTOR_NAME: models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    # ----------------------------------------------------------
    # Create embeddings
    # ----------------------------------------------------------
    dense_embeddings = dense_model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        show_progress_bar=verbose,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for i, (chunk, text, dense_emb, sparse_emb) in enumerate(
        zip(chunks, texts, dense_embeddings, sparse_embeddings)
    ):
        payload = {
            "text": text,
            "metadata": chunk.get("metadata", {}),
        }

        point = models.PointStruct(
            id=i,
            vector={
                config.DENSE_VECTOR_NAME: dense_emb.tolist(),
                config.SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
            },
            payload=payload,
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    client.close()

    if verbose:
        print("=" * 60)
        print("DONE - Qdrant hybrid store ready")
        print("=" * 60)
        print(f"Total vectors : {len(points)}")
        print(f"Storage path  : {qdrant_path}")

    return len(points)


if __name__ == "__main__":
    build_index()
