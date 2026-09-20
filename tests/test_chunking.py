from src.chunking import Chunk, chunk_page_records
from src.ingest import extract_pages_from_pdf


def test_chunking_produces_non_empty_collection():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)
    assert len(chunks) > 0
    assert len(chunks) >= len(records)


def test_every_chunk_has_valid_id_and_non_empty_text():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_id.strip() != ""
        assert c.chunk_id.startswith("chunk_p")
        assert c.text.strip() != ""


def test_every_chunk_has_valid_1_indexed_page_number():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    valid_pages = set(r.page_number for r in records)
    for c in chunks:
        assert c.page_number in valid_pages
        assert 1 <= c.page_number <= 44


def test_chunk_ids_and_ordering_are_deterministic():
    records = extract_pages_from_pdf()
    run1 = chunk_page_records(records)
    run2 = chunk_page_records(records)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.page_number == c2.page_number
        assert c1.text == c2.text
        assert c1.section == c2.section


def test_chunk_objects_contain_all_expected_metadata():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    for c in chunks:
        data = c.to_dict()
        assert set(data.keys()) == {"chunk_id", "page_number", "text", "section"}
        assert isinstance(data["chunk_id"], str)
        assert isinstance(data["page_number"], int)
        assert isinstance(data["text"], str)
        assert data["section"] is None or isinstance(data["section"], str)


def test_chunk_size_boundaries():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    lengths = [len(c.text) for c in chunks]
    # No chunk should exceed the 1,620-character consolidation ceiling
    assert max(lengths) <= 1620
    # Every chunk should satisfy the minimum chunk threshold
    assert min(lengths) >= 200


def test_page_metadata_preservation():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    represented_pages = {c.page_number for c in chunks}
    assert len(represented_pages) == 44
    assert represented_pages == set(range(1, 45))


def test_important_content_from_multiple_parts_preserved():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    all_text = " ".join(c.text for c in chunks)
    assert "Agent-as-a-Judge" in all_text
    assert "Mingchen Zhuge" in all_text
    assert "DevAI" in all_text
    assert "MetaGPT" in all_text
    assert "OpenHands" in all_text
    assert "Conclusion" in all_text


def test_tables_are_preserved_in_chunks():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    for table_num in range(1, 8):
        table_label = f"Table {table_num}"
        matching_chunks = [c for c in chunks if table_label in c.text]
        assert len(matching_chunks) > 0, f"{table_label} was not found in any chunk"


def test_table_1_framework_names_and_cost_values_cooccur():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    t1_chunk = next((c for c in chunks if c.chunk_id == "chunk_p06_001"), None)
    assert t1_chunk is not None
    # Verify framework names and cost values exist in the same chunk
    assert "MetaGPT" in t1_chunk.text
    assert "GPT-Pilot" in t1_chunk.text
    assert "OpenHands" in t1_chunk.text
    assert "$1.19" in t1_chunk.text
    assert "$3.92" in t1_chunk.text
    assert "$6.38" in t1_chunk.text


def test_section_attribution_does_not_leak_next_section():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    p3_chunks = [c for c in chunks if c.page_number == 3]
    assert len(p3_chunks) >= 2

    # First chunk on page 3 contains Section 1 contributions
    c1 = p3_chunks[0]
    assert "Agent-as-a-Judge saves 97.72%" in c1.text
    assert c1.section == "1 Introduction"

    # Second chunk on page 3 introduces Section 2
    c2 = p3_chunks[1]
    assert "DevAI: A Dataset for Automated AI Development" in c2.text
    assert c2.section == "2 DevAI: A Dataset for Automated AI Development"


def test_table_3_continuation_retains_identifiable_context():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    p9_chunks = [c for c in chunks if c.page_number == 9]
    # Continuation chunk with Table 3 rows must retain table context
    c_data = next((c for c in p9_chunks if "LLM-as-a-Judge" in c.text and "Requirements Met (I)" in c.text), None)
    assert c_data is not None
    assert "Table 3 (continued)" in c_data.text
    assert "AI Judges and Their Shift/Alignment" in c_data.text


def test_no_empty_or_whitespace_chunks():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    for c in chunks:
        assert len(c.text.strip()) > 0
        assert not c.text.isspace()


def test_section_metadata_identified():
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)

    chunks_with_section = [c for c in chunks if c.section is not None]
    assert len(chunks_with_section) > 0

    section_names = {c.section for c in chunks_with_section}
    assert any("Introduction" in s for s in section_names)
    assert any("DevAI" in s for s in section_names)
    assert any("Appendix" in s for s in section_names)
