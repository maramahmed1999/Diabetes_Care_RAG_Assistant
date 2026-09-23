"""
Phase 2 — Cleaning.

Turns the raw, position-ordered lines produced by `extract.py` into
clean, readable paragraphs per page, removes running headers/footers
and page numbers, and attaches document metadata.
"""

import re
from collections import Counter

import pymupdf

from .extract import extract_lines, fix_pdf_ligatures, order_lines


def remove_page_number_lines(lines: list[dict]) -> list[dict]:
    cleaned = []

    for line in lines:

        text = line["text"].strip()

        # Examples: "3", "89", "S89", "S101"
        if re.fullmatch(r"S?\d{1,4}", text):
            continue

        cleaned.append(line)

    return cleaned


def clean_line(text: str) -> str:
    text = fix_pdf_ligatures(text)

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Collapse repeated spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def fix_hyphenation(text: str) -> str:
    """
    Example:
        hypo-
        glycemia
    ->
        hypoglycemia

    Only applied when the hyphen is clearly at the end of a line.
    """
    return re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", r"\1\2", text)


def reconstruct_paragraphs(lines: list[dict]) -> str:
    """
    Convert ordered PDF lines into readable paragraphs.

    Headings and blank-line boundaries are preserved where possible.
    """
    if not lines:
        return ""

    text_lines = [
        clean_line(line["text"])
        for line in lines
        if line["text"].strip()
    ]

    text_lines = [line for line in text_lines if line]

    if not text_lines:
        return ""

    # Join lines initially
    text = "\n".join(text_lines)

    # Fix words split at line boundaries
    text = fix_hyphenation(text)

    # ------------------------------------------------------------
    # Join normal wrapped lines. We avoid aggressive paragraph
    # detection here; a later, paragraph-aware step (chunking)
    # handles the final structure.
    # ------------------------------------------------------------
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def extract_page(page) -> str:
    """
    Extract and order one PDF page, fully cleaned.
    """
    lines = extract_lines(page)

    # Remove standalone page numbers
    lines = remove_page_number_lines(lines)

    # Correct reading order
    lines = order_lines(lines)

    text = reconstruct_paragraphs(lines)

    return text


def extract_document(pdf_path, document_id: str) -> dict:
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_index, page in enumerate(doc):

        page_number = page_index + 1

        text = extract_page(page)

        if not text:
            continue

        pages.append({
            "page": page_number,
            "text": text,
        })

    doc.close()

    return {
        "document_id": document_id,
        "pages": pages,
    }


def find_repeated_lines(pages: list[dict]) -> set:
    """
    Detect lines repeated across many pages.

    These are usually journal headers, footers, running titles or
    publication information.
    """
    counter = Counter()

    total_pages = len(pages)

    if total_pages < 5:
        return set()

    for page in pages:

        lines = set(
            line.strip()
            for line in page["text"].split("\n")
            if line.strip()
        )

        for line in lines:
            counter[line] += 1

    repeated = set()

    # At least 30% of pages
    threshold = max(3, int(total_pages * 0.30))

    for line, count in counter.items():

        if count >= threshold:
            # Avoid deleting huge chunks of actual content
            if len(line) < 250:
                repeated.add(line)

    return repeated


def remove_repeated_lines(pages: list[dict], repeated_lines: set) -> list[dict]:
    cleaned_pages = []

    for page in pages:

        lines = page["text"].split("\n")

        cleaned = [line for line in lines if line.strip() not in repeated_lines]

        text = "\n".join(cleaned)

        # Normalize excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned_pages.append({
            **page,
            "text": text.strip(),
        })

    return cleaned_pages


def attach_metadata(extracted_document: dict, document_id: str, document_metadata: dict) -> list[dict]:
    base_metadata = document_metadata[document_id]

    documents = []

    for page_data in extracted_document["pages"]:

        metadata = {
            **base_metadata,
            "document_id": document_id,
            "page": page_data["page"],
        }

        documents.append({
            "text": page_data["text"],
            "metadata": metadata,
        })

    return documents


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into logical paragraphs using blank lines as the
    primary signal.
    """
    paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs
