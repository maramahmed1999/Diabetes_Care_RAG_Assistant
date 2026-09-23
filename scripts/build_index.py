"""
Run this after run_ingestion.py to embed every chunk (dense + BM25)
and build the local Qdrant hybrid-search collection.

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.index import build_index

if __name__ == "__main__":
    build_index()
