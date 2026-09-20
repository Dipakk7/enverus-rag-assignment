import math
import pytest

from src.chunking import Chunk
from src.embeddings import EmbeddedChunk, EmbeddingManager


@pytest.fixture(scope="module")
def embedding_manager():
    """Shared EmbeddingManager fixture to avoid reloading model weights repeatedly."""
    return EmbeddingManager()


def test_embedding_model_loads(embedding_manager):
    assert embedding_manager.model is not None
    assert embedding_manager.embedding_dimension == 384


def test_non_empty_text_produces_numeric_embedding(embedding_manager):
    text = "Agent-as-a-Judge: Evaluate Agents with Agents."
    vec = embedding_manager.embed_text(text)

    assert isinstance(vec, list)
    assert len(vec) == embedding_manager.embedding_dimension
    assert all(isinstance(x, float) for x in vec)

    # L2-normalized vector has Euclidean norm of ~1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_embedding_dimensionality_is_consistent(embedding_manager):
    dim = embedding_manager.embedding_dimension
    vec1 = embedding_manager.embed_text("Short text")
    vec2 = embedding_manager.embed_text(
        "A substantially longer text discussing the DevAI benchmark, 55 tasks, and agentic systems."
    )
    assert len(vec1) == dim
    assert len(vec2) == dim


def test_query_embedding_works(embedding_manager):
    query = "What are the DevAI dataset statistics?"
    query_vec = embedding_manager.embed_query(query)

    assert isinstance(query_vec, list)
    assert len(query_vec) == embedding_manager.embedding_dimension
    assert all(isinstance(x, float) for x in query_vec)


def test_repeated_embedding_is_deterministic(embedding_manager):
    text = "Deterministic numerical representation verification text."
    vec1 = embedding_manager.embed_text(text)
    vec2 = embedding_manager.embed_text(text)

    assert len(vec1) == len(vec2)
    for v1, v2 in zip(vec1, vec2):
        assert abs(v1 - v2) < 1e-6


def test_empty_or_whitespace_input_raises_value_error(embedding_manager):
    with pytest.raises(ValueError):
        embedding_manager.embed_text("")

    with pytest.raises(ValueError):
        embedding_manager.embed_text("   \n\t  ")

    with pytest.raises(ValueError):
        embedding_manager.embed_query("")


def test_multiple_chunks_produce_expected_number_of_embeddings(embedding_manager):
    chunks = [
        Chunk(chunk_id="c1", page_number=1, text="First test chunk", section="1 Introduction"),
        Chunk(chunk_id="c2", page_number=1, text="Second test chunk", section="1 Introduction"),
        Chunk(chunk_id="c3", page_number=2, text="Third test chunk with tables", section="2 DevAI"),
    ]
    embedded_chunks = embedding_manager.embed_chunks(chunks)

    assert len(embedded_chunks) == len(chunks)
    for ec, c in zip(embedded_chunks, chunks):
        assert isinstance(ec, EmbeddedChunk)
        assert ec.chunk_id == c.chunk_id
        assert ec.page_number == c.page_number
        assert ec.text == c.text
        assert ec.section == c.section
        assert len(ec.embedding) == embedding_manager.embedding_dimension


def test_embed_chunks_with_empty_chunk_raises_value_error(embedding_manager):
    bad_chunks = [
        Chunk(chunk_id="valid", page_number=1, text="Valid content", section=None),
        Chunk(chunk_id="empty", page_number=1, text="   ", section=None),
    ]
    with pytest.raises(ValueError):
        embedding_manager.embed_chunks(bad_chunks)
