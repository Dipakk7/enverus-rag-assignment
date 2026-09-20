from pathlib import Path
import pytest
import fitz

from src.ingest import DEFAULT_PDF_PATH, PageRecord, extract_pages_from_pdf


def test_pdf_exists_and_can_be_opened():
    assert DEFAULT_PDF_PATH.exists(), f"Target PDF not found at {DEFAULT_PDF_PATH}"
    doc = fitz.open(DEFAULT_PDF_PATH)
    try:
        assert len(doc) > 0
    finally:
        doc.close()


def test_extracted_page_count_matches_actual_pdf():
    doc = fitz.open(DEFAULT_PDF_PATH)
    expected_page_count = len(doc)
    doc.close()

    records = extract_pages_from_pdf(DEFAULT_PDF_PATH)
    assert len(records) == expected_page_count
    assert len(records) == 44


def test_page_numbers_are_preserved_sequentially():
    records = extract_pages_from_pdf(DEFAULT_PDF_PATH)
    expected_numbers = list(range(1, len(records) + 1))
    actual_numbers = [record.page_number for record in records]
    assert actual_numbers == expected_numbers


def test_extracted_text_non_empty_for_expected_pages():
    records = extract_pages_from_pdf(DEFAULT_PDF_PATH)
    # The first page and subsequent main content pages must have text
    assert len(records[0].text.strip()) > 0
    # Check that across the document, pages have extractable text
    non_empty_pages = [r for r in records if r.text.strip()]
    assert len(non_empty_pages) == len(records)


def test_representative_known_text_present():
    records = extract_pages_from_pdf(DEFAULT_PDF_PATH)
    # The first page should contain the title and author name
    first_page_text = records[0].text
    assert "Agent-as-a-Judge" in first_page_text
    assert "Mingchen Zhuge" in first_page_text


def test_nonexistent_pdf_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        extract_pages_from_pdf(Path("data/nonexistent_file.pdf"))


def test_directory_path_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        extract_pages_from_pdf(tmp_path)
