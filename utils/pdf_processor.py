"""PDF extraction and chunking helpers for the Advanced RAG Research Assistant."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import fitz


def _as_pdf_bytes(pdf_file: Any) -> bytes:
    """Accept bytes, bytearray, or an UploadedFile-like object."""
    if isinstance(pdf_file, bytes):
        return pdf_file
    if isinstance(pdf_file, bytearray):
        return bytes(pdf_file)
    if hasattr(pdf_file, "getvalue"):
        return pdf_file.getvalue()
    raise TypeError("pdf_file must be bytes or an object exposing getvalue().")


def extract_text_from_pdf(pdf_file: Any) -> list[dict[str, Any]]:
    """Extract selectable text per page and preserve original 1-based page numbers."""
    pdf_bytes = _as_pdf_bytes(pdf_file)
    pages: list[dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page_number": page_number, "text": text})
    return pages


def create_chunks(
    pages: list[dict[str, Any]], chunk_size: int = 500, overlap: int = 100
) -> list[dict[str, Any]]:
    """Split each page into overlapping word chunks without crossing page boundaries."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if (
        not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError("overlap must be an integer >= 0 and smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        if "page_number" not in page:
            raise ValueError("Each page must contain a page_number field.")
        words = text.split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                break
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": int(page["page_number"]),
                    "text": " ".join(chunk_words),
                }
            )
            chunk_id += 1
            if start + chunk_size >= len(words):
                break
    return chunks


def process_multiple_pdfs(
    documents: Iterable[tuple[str, bytes]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Process several ``(filename, pdf_bytes)`` documents.

    Returns combined page records, combined chunk records, and a per-document summary.
    Page numbers remain local to each PDF; each page/chunk also records ``source`` and
    ``document_id`` so same-numbered pages from different PDFs stay distinguishable.
    """
    all_pages: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for filename, raw_bytes in documents:
        if not filename or not isinstance(filename, str):
            raise ValueError("Each document must have a non-empty filename.")
        pdf_bytes = _as_pdf_bytes(raw_bytes)
        if not pdf_bytes:
            raise ValueError(f"The uploaded file {filename!r} is empty.")

        # Include the name and bytes so different documents with identical filenames
        # or identical text contents can still be tracked separately.
        identity = hashlib.sha256(filename.encode("utf-8") + b"\0" + pdf_bytes).hexdigest()[:16]
        if identity in seen_ids:
            # The same file can be selected only once in the UI normally. This guard
            # keeps collection IDs unambiguous if a caller passes duplicates.
            continue
        seen_ids.add(identity)

        pages = extract_text_from_pdf(pdf_bytes)
        chunks = create_chunks(pages, chunk_size=chunk_size, overlap=overlap)

        for page in pages:
            all_pages.append(
                {
                    "source": filename,
                    "document_id": identity,
                    "page_number": page["page_number"],
                    "text": page["text"],
                }
            )
        for chunk in chunks:
            all_chunks.append(
                {
                    **chunk,
                    "source": filename,
                    "document_id": identity,
                }
            )

        all_text_words = sum(len(page["text"].split()) for page in pages)
        summaries.append(
            {
                "source": filename,
                "document_id": identity,
                "page_count": len(pages),
                "word_count": all_text_words,
                "chunk_count": len(chunks),
                "file_size_bytes": len(pdf_bytes),
            }
        )

    return all_pages, all_chunks, summaries
