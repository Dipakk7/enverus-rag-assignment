from pathlib import Path
from typing import List, Optional
import pytest

from src.chunking import Chunk
from src.embeddings import EmbeddedChunk
from src.vector_store import StoredRecord, VectorStore


def make_dummy_chunk(
    chunk_id: str,
    page_number: int = 1,
    text: str = "Test chunk content",
    section: Optional[str] = "1 Introduction",
    dim: int = 384,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        section=section,
    )
    embedding = [0.01 * (i % 10) for i in range(dim)]
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


@pytest.fixture
def temp_store(tmp_path: Path) -> VectorStore:
    """Fixture providing an isolated VectorStore in a temporary directory."""
    return VectorStore(
        persist_dir=tmp_path / "test_chroma",
        collection_name="test_collection",
    )


def test_collection_creation_and_empty_behavior(temp_store: VectorStore):
    assert temp_store.collection is not None
    assert temp_store.count() == 0
    assert temp_store.get_all_ids() == []


def test_adding_embedded_chunks_and_count(temp_store: VectorStore):
    chunks = [
        make_dummy_chunk("chunk_p01_001", page_number=1, text="First paragraph"),
        make_dummy_chunk("chunk_p01_002", page_number=1, text="Second paragraph"),
        make_dummy_chunk("chunk_p02_001", page_number=2, text="Third paragraph"),
    ]
    upserted = temp_store.upsert_embedded_chunks(chunks)

    assert upserted == 3
    assert temp_store.count() == 3
    stored_ids = set(temp_store.get_all_ids())
    assert stored_ids == {"chunk_p01_001", "chunk_p01_002", "chunk_p02_001"}


def test_id_uniqueness_and_get_by_id(temp_store: VectorStore):
    item = make_dummy_chunk(
        "chunk_p05_001",
        page_number=5,
        text="Task 51 description",
        section="2.3 Preliminary Benchmark",
    )
    temp_store.upsert_embedded_chunks([item])

    record = temp_store.get_by_id("chunk_p05_001", include_embedding=True)
    assert record is not None
    assert isinstance(record, StoredRecord)
    assert record.chunk_id == "chunk_p05_001"
    assert record.page_number == 5
    assert record.section == "2.3 Preliminary Benchmark"
    assert record.document == "Task 51 description"
    assert record.embedding is not None
    assert len(record.embedding) == 384


def test_metadata_preservation_with_none_section(temp_store: VectorStore):
    item_none_sec = make_dummy_chunk(
        "chunk_p01_001",
        page_number=1,
        text="Preamble text without section header",
        section=None,
    )
    temp_store.upsert_embedded_chunks([item_none_sec])

    record = temp_store.get_by_id("chunk_p01_001")
    assert record is not None
    assert record.section is None
    assert record.page_number == 1


def test_document_text_exact_preservation(temp_store: VectorStore):
    raw_text = "Exact text preservation check with symbols: $1.19, 97.72%, and newlines.\nSecond line."
    item = make_dummy_chunk("chunk_exact", page_number=6, text=raw_text)
    temp_store.upsert_embedded_chunks([item])

    record = temp_store.get_by_id("chunk_exact")
    assert record is not None
    assert record.document == raw_text


def test_embedding_dimensionality(temp_store: VectorStore):
    item = make_dummy_chunk("chunk_dim_check", dim=384)
    temp_store.upsert_embedded_chunks([item])

    record = temp_store.get_by_id("chunk_dim_check", include_embedding=True)
    assert record is not None
    assert record.embedding is not None
    assert len(record.embedding) == 384


def test_get_by_ids_multiple_records(temp_store: VectorStore):
    items = [
        make_dummy_chunk("c1", page_number=1, text="Text 1"),
        make_dummy_chunk("c2", page_number=2, text="Text 2"),
        make_dummy_chunk("c3", page_number=3, text="Text 3"),
    ]
    temp_store.upsert_embedded_chunks(items)

    records = temp_store.get_by_ids(["c1", "c3", "nonexistent"])
    assert len(records) == 2
    rec_ids = {r.chunk_id for r in records}
    assert rec_ids == {"c1", "c3"}


def test_repeated_upsert_does_not_create_duplicates(temp_store: VectorStore):
    items = [
        make_dummy_chunk("chunk_dup_1", page_number=1, text="Text A"),
        make_dummy_chunk("chunk_dup_2", page_number=2, text="Text B"),
    ]
    # First indexing run
    temp_store.upsert_embedded_chunks(items)
    assert temp_store.count() == 2

    # Second indexing run with identical IDs
    temp_store.upsert_embedded_chunks(items)
    assert temp_store.count() == 2
    assert len(temp_store.get_all_ids()) == 2


def test_explicit_reset_behavior(temp_store: VectorStore):
    items = [make_dummy_chunk(f"c{i}") for i in range(5)]
    temp_store.upsert_embedded_chunks(items)
    assert temp_store.count() == 5

    temp_store.reset_collection()
    assert temp_store.count() == 0
    assert temp_store.get_all_ids() == []


def test_invalid_data_handling(temp_store: VectorStore):
    # Empty chunk list returns 0
    assert temp_store.upsert_embedded_chunks([]) == 0

    # Non-existent ID returns None
    assert temp_store.get_by_id("missing_id") is None

    # Empty string ID returns None
    assert temp_store.get_by_id("") is None

    # Empty chunk_ids list returns empty list
    assert temp_store.get_by_ids([]) == []

    # Invalid chunk with empty text raises ValueError
    bad_chunk = EmbeddedChunk(
        chunk=Chunk(chunk_id="bad", page_number=1, text="  ", section=None),
        embedding=[0.1] * 384,
    )
    with pytest.raises(ValueError):
        temp_store.upsert_embedded_chunks([bad_chunk])

    # Invalid chunk with empty chunk_id raises ValueError
    bad_id_chunk = EmbeddedChunk(
        chunk=Chunk(chunk_id="", page_number=1, text="Some text", section=None),
        embedding=[0.1] * 384,
    )
    with pytest.raises(ValueError):
        temp_store.upsert_embedded_chunks([bad_id_chunk])
