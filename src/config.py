"""
Central configuration for the RAG pipeline.

Every path / model name / constant that used to be a hard-coded
variable somewhere in the notebook lives here now, so the ingestion
scripts, the FastAPI backend and the Streamlit frontend all agree on
the same values.
"""

from pathlib import Path

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = OUTPUT_DIR / "diabetes_chunks.jsonl"

QDRANT_PATH = BASE_DIR / "data" / "qdrant_db"
COLLECTION_NAME = "diabetes_collection"

# --------------------------------------------------------------------
# Source PDFs (same mapping used in the notebook)
# --------------------------------------------------------------------
PDF_FILES = {
    "ada_section5": DATA_DIR / "ada_section5.pdf",
    "ada_section6": DATA_DIR / "ada_section6.pdf",
    "niddk_eating": DATA_DIR / "niddk_eating.pdf",
    "who_diabetes": DATA_DIR / "who_diabetes.pdf",
}

# --------------------------------------------------------------------
# Document metadata (same content as the notebook)
# --------------------------------------------------------------------
DOCUMENT_METADATA = {
    "ada_section5": {
        "organization": "American Diabetes Association",
        "document_title": "Standards of Care in Diabetes—2026: Section 5",
        "source": "ADA",
        "year": 2026,
        "topic": "Nutrition and health behaviors",
    },
    "ada_section6": {
        "organization": "American Diabetes Association",
        "document_title": "Standards of Care in Diabetes—2026: Section 6",
        "source": "ADA",
        "year": 2026,
        "topic": "Glycemic goals and hypoglycemia",
    },
    "niddk_eating": {
        "organization": "National Institute of Diabetes and Digestive and Kidney Diseases",
        "document_title": "Eating and Diabetes",
        "source": "NIDDK",
        "year": None,
        "topic": "Nutrition and meal planning",
    },
    "who_diabetes": {
        "organization": "World Health Organization",
        "document_title": "Diabetes Fact Sheet",
        "source": "WHO",
        "year": 2012,
        "topic": "Diabetes overview",
    },
}

# --------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# --------------------------------------------------------------------
# Embedding models
# --------------------------------------------------------------------
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
EMBED_BATCH_SIZE = 32

DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "bm25_vector"

# --------------------------------------------------------------------
# LLM (Ollama)
# --------------------------------------------------------------------
OLLAMA_MODEL = "phi3"
OLLAMA_HOST = "http://localhost:11434"

# --------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------
TOP_K = 3
