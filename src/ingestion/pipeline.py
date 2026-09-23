"""
Ingestion pipeline orchestrator.

Runs the three ingestion phases in order:
    1. extract   (PDF -> ordered raw lines)
    2. clean     (raw lines -> clean per-page text + metadata)
    3. chunk     (per-page text -> overlapping chunks)

and writes the resulting chunks to a JSONL file, exactly like the
notebook's final ingestion cells did.
"""

import json

from src import config
from .chunk import create_chunks
from .clean import (
    attach_metadata,
    extract_document,
    find_repeated_lines,
    remove_repeated_lines,
)


def run_ingestion(
    pdf_files: dict | None = None,
    document_metadata: dict | None = None,
    output_file=None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    verbose: bool = True,
) -> list[dict]:
    """
    Runs the full extract -> clean -> chunk pipeline over every PDF
    declared in `pdf_files` and writes the resulting chunks as JSONL.

    Returns the list of chunks that were written.
    """
    pdf_files = pdf_files or config.PDF_FILES
    document_metadata = document_metadata or config.DOCUMENT_METADATA
    output_file = output_file or config.CHUNKS_FILE

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_page_documents = []

    for document_id, pdf_path in pdf_files.items():

        if verbose:
            print("=" * 80)
            print(f"Processing: {document_id}")
            print(f"File: {pdf_path}")

        if not pdf_path.exists():
            if verbose:
                print("WARNING: File not found! Skipping.")
            continue

        # ------------------------------------------
        # PDF -> ordered clean pages
        # ------------------------------------------
        extracted = extract_document(pdf_path, document_id)

        if verbose:
            print(f"Extracted pages: {len(extracted['pages'])}")

        repeated_lines = find_repeated_lines(extracted["pages"])

        if verbose:
            print(f"Repeated lines detected: {len(repeated_lines)}")

        cleaned_pages = remove_repeated_lines(extracted["pages"], repeated_lines)

        extracted["pages"] = cleaned_pages

        # ------------------------------------------
        # Attach metadata
        # ------------------------------------------
        documents = attach_metadata(extracted, document_id, document_metadata)

        all_page_documents.extend(documents)

    # ------------------------------------------
    # Chunk everything
    # ------------------------------------------
    chunks = create_chunks(
        all_page_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if verbose:
        print("=" * 80)
        print(f"Total page documents: {len(all_page_documents)}")
        print(f"Total chunks: {len(chunks)}")

    # ------------------------------------------
    # Save
    # ------------------------------------------
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if verbose:
        print("=" * 80)
        print(f"Saved to: {output_file}")

    return chunks


if __name__ == "__main__":
    run_ingestion()
