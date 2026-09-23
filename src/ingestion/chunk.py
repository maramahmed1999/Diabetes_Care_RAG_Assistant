"""
Phase 3 — Chunking.

Splits cleaned page text into overlapping chunks that respect
paragraph (and, when needed, sentence) boundaries, and attaches a
stable chunk id to each one.
"""

import re

from .clean import split_into_paragraphs


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[str]:
    """
    Create chunks while preserving paragraph boundaries as much as
    possible. chunk_size is measured approximately in characters.
    """
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # Paragraph itself is larger than chunk size
        if len(paragraph) > chunk_size:

            if current:
                chunks.append(current)
                current = ""

            # Split oversized paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            temp = ""

            for sentence in sentences:

                if len(temp) + len(sentence) + 1 <= chunk_size:
                    if temp:
                        temp += " "
                    temp += sentence
                else:
                    if temp:
                        chunks.append(temp)
                    temp = sentence

            if temp:
                current = temp

            continue

        # Normal paragraph
        if len(current) + len(paragraph) + 2 <= chunk_size:
            if current:
                current += "\n\n"
            current += paragraph
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    # ------------------------------------------------------------
    # Add overlap
    # ------------------------------------------------------------
    final_chunks = []

    for i, chunk in enumerate(chunks):

        if i == 0:
            final_chunks.append(chunk)
            continue

        previous = chunks[i - 1]

        overlap = previous[max(0, len(previous) - chunk_overlap):]

        final_chunks.append(overlap + "\n\n" + chunk)

    return final_chunks


def create_chunks(documents: list[dict], chunk_size: int = 1200, chunk_overlap: int = 200) -> list[dict]:
    all_chunks = []

    for document in documents:

        text = document["text"]

        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_index, chunk in enumerate(chunks):

            metadata = {
                **document["metadata"],
                "chunk_index": chunk_index,
                "chunk_id": (
                    f"{document['metadata']['document_id']}"
                    f"_p{document['metadata']['page']}"
                    f"_c{chunk_index:03d}"
                ),
            }

            all_chunks.append({
                "text": chunk,
                "metadata": metadata,
            })

    return all_chunks
