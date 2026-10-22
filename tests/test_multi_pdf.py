"""Tests for multi-PDF extraction and metadata preservation."""

import fitz

from utils.pdf_processor import create_chunks, extract_text_from_pdf, process_multiple_pdfs


def make_pdf(page_texts: list[str]) -> bytes:
    """Create a small in-memory PDF test fixture."""
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def test_extract_text_preserves_page_numbers() -> None:
    pdf_bytes = make_pdf(["First page text.", "", "Third page text."])
    pages = extract_text_from_pdf(pdf_bytes)
    assert [page["page_number"] for page in pages] == [1, 3]


def test_process_multiple_pdfs_preserves_sources_and_page_numbers() -> None:
    pdf_one = make_pdf(["Alpha information is in document one."])
    pdf_two = make_pdf(["Beta information is in document two.", "Another beta page."])

    pages, chunks, summaries = process_multiple_pdfs(
        [("alpha.pdf", pdf_one), ("beta.pdf", pdf_two)], chunk_size=100, overlap=10
    )

    assert {page["source"] for page in pages} == {"alpha.pdf", "beta.pdf"}
    assert all(page["document_id"] for page in pages)
    assert {chunk["source"] for chunk in chunks} == {"alpha.pdf", "beta.pdf"}
    assert all(chunk["document_id"] for chunk in chunks)
    assert len(summaries) == 2
    assert next(item for item in summaries if item["source"] == "alpha.pdf")["page_count"] == 1
    assert next(item for item in summaries if item["source"] == "beta.pdf")["page_count"] == 2


def test_chunks_keep_page_boundaries() -> None:
    pages = [
        {"page_number": 1, "text": "one two three four"},
        {"page_number": 2, "text": "five six seven eight"},
    ]
    chunks = create_chunks(pages, chunk_size=3, overlap=1)
    assert {chunk["page_number"] for chunk in chunks} == {1, 2}
    assert all(chunk["text"] for chunk in chunks)
