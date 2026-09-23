"""
Run this once (or whenever your source PDFs change) to turn the PDFs
in data/raw/ into cleaned, chunked passages saved at
data/processed/diabetes_chunks.jsonl.

Usage:
    python scripts/run_ingestion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion()
