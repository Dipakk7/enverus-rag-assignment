"""
Unit tests for the semantic retrieval module (src/retriever.py).
"""

from pathlib import Path
from typing import List, Optional
import pytest

from src.chunking import Chunk
from src.embeddings import EmbeddedChunk, EmbeddingManager
from src.retriever import (
    RetrievalResult,
    SemanticRetriever,
    get_retriever,
    retrieve,
)
from src.vector_store import VectorStore


class MockEmbeddingManager:
    """Mock embedding manager for fast, deterministic, offline unit testing."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.call_count = 0
        self.last_query: Optional[str] = None

    def embed_query(self, query: str) -> List[float]:
        self.call_count += 1
        self.last_query = query
        # Deterministic unit-vector patterns
        if "eval" in query.lower():
            return [1.0, 0.0, 0.0, 0.0]
        elif "cost" in query.lower():
            return [0.0, 1.0, 0.0, 0.0]
        elif "model" in query.lower():
            return [0.0, 0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.5, 0.5]


def create_synthetic_chunk(
    chunk_id: str,
    page_number: int,
    text: str,
    section: Optional[str],
    vector: List[float],
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            page_number=page_number,
            text=text,
            section=section,
        ),
        embedding=vector,
    )


@pytest.fixture
def populated_retriever(tmp_path: Path) -> SemanticRetriever:
    """Creates a SemanticRetriever with a temporary VectorStore populated with 4 synthetic records."""
    store = VectorStore(
        persist_dir=tmp_path / "test_retriever_chroma",
        collection_name="test_retriever_col",
    )
    mock_embeddings = MockEmbeddingManager(dim=4)

    # 4 distinct orthogonal vectors
    chunks = [
        create_synthetic_chunk(
            chunk_id="chunk_p01_001",
            page_number=1,
            text="Agent evaluation methodology and benchmarks.",
            section="1 Introduction",
            vector=[1.0, 0.0, 0.0, 0.0],
        ),
        create_synthetic_chunk(
            chunk_id="chunk_p02_001",
            page_number=2,
            text="Financial cost and computational runtime analysis.",
            section="2 Background",
            vector=[0.0, 1.0, 0.0, 0.0],
        ),
        create_synthetic_chunk(
            chunk_id="chunk_p03_001",
            page_number=3,
            text="Model architectures including SVM and LSTM classifiers.",
            section="3 Experiments",
            vector=[0.0, 0.0, 1.0, 0.0],
        ),
        create_synthetic_chunk(
            chunk_id="chunk_p04_001",
            page_number=4,
            text="General discussions and appendix details.",
            section=None,
            vector=[0.0, 0.0, 0.0, 1.0],
        ),
    ]
    store.upsert_embedded_chunks(chunks)

    return SemanticRetriever(
        vector_store=store,
        embedding_manager=mock_embeddings,
        default_top_k=2,
    )


# 1. Query validation
def test_query_validation_empty_or_invalid(populated_retriever: SemanticRetriever):
    with pytest.raises(ValueError, match="Query cannot be empty or whitespace-only"):
        populated_retriever.retrieve("")

    with pytest.raises(ValueError, match="Query cannot be empty or whitespace-only"):
        populated_retriever.retrieve("   \n\t  ")

    with pytest.raises(TypeError, match="Query must be a string"):
        populated_retriever.retrieve(None)  # type: ignore

    with pytest.raises(TypeError, match="Query must be a string"):
        populated_retriever.retrieve(12345)  # type: ignore


# 2. top_k validation
def test_top_k_validation_types_and_bounds(populated_retriever: SemanticRetriever):
    # Non-positive or zero
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        populated_retriever.retrieve("eval", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        populated_retriever.retrieve("eval", top_k=-2)

    # Non-integer / boolean
    with pytest.raises(TypeError, match="top_k must be an integer"):
        populated_retriever.retrieve("eval", top_k=True)  # type: ignore

    with pytest.raises(TypeError, match="top_k must be an integer"):
        populated_retriever.retrieve("eval", top_k="2")  # type: ignore

    # Exceeding collection count (collection has 4 records)
    with pytest.raises(ValueError, match="exceeds total collection size"):
        populated_retriever.retrieve("eval", top_k=5)


# 3. Query embedding is used
def test_query_embedding_invoked(populated_retriever: SemanticRetriever):
    mock_mgr: MockEmbeddingManager = populated_retriever.embedding_manager  # type: ignore
    assert mock_mgr.call_count == 0

    results = populated_retriever.retrieve("  eval agents  ", top_k=2)

    assert mock_mgr.call_count == 1
    assert mock_mgr.last_query == "eval agents"  # Verified whitespace stripping
    assert len(results) == 2


# 4. Results returned in rank order
def test_results_returned_in_rank_order(populated_retriever: SemanticRetriever):
    results = populated_retriever.retrieve("eval", top_k=3)
    assert len(results) == 3

    ranks = [r.rank for r in results]
    assert ranks == [1, 2, 3]

    # Scores should be non-increasing (rank 1 highest score / lowest distance)
    scores = [r.score for r in results]
    distances = [r.distance for r in results]
    assert scores == sorted(scores, reverse=True)
    assert distances == sorted(distances)
    # The perfect match for "eval" is chunk_p01_001
    assert results[0].chunk_id == "chunk_p01_001"


# 5. Metadata is preserved
def test_metadata_preserved(populated_retriever: SemanticRetriever):
    results = populated_retriever.retrieve("cost", top_k=1)
    res = results[0]
    assert res.chunk_id == "chunk_p02_001"
    assert res.page_number == 2
    assert res.section == "2 Background"

    # Test None section preservation
    res_all = populated_retriever.retrieve("general", top_k=4)
    p04_res = next(r for r in res_all if r.chunk_id == "chunk_p04_001")
    assert p04_res.section is None
    assert p04_res.page_number == 4


# 6. Text is preserved
def test_text_preserved(populated_retriever: SemanticRetriever):
    results = populated_retriever.retrieve("cost", top_k=1)
    assert results[0].text == "Financial cost and computational runtime analysis."


# 7. Scores/distances are present and numeric
def test_scores_and_distances_numeric_definition(populated_retriever: SemanticRetriever):
    results = populated_retriever.retrieve("eval", top_k=2)
    for res in results:
        assert isinstance(res.score, float)
        assert isinstance(res.distance, float)
        # Verify mathematical relationship: score == 1.0 - distance
        assert pytest.approx(res.score + res.distance, rel=1e-5) == 1.0
        # Check dictionary conversion
        d = res.to_dict()
        assert d["chunk_id"] == res.chunk_id
        assert d["score"] == res.score
        assert d["distance"] == res.distance


# 8. top_k controls result count
def test_top_k_controls_count(populated_retriever: SemanticRetriever):
    res_1 = populated_retriever.retrieve("cost", top_k=1)
    assert len(res_1) == 1

    res_3 = populated_retriever.retrieve("cost", top_k=3)
    assert len(res_3) == 3

    # Default top_k configured on fixture is 2
    res_default = populated_retriever.retrieve("cost")
    assert len(res_default) == 2


# 9. Empty collection is handled clearly
def test_empty_collection_handling(tmp_path: Path):
    empty_store = VectorStore(
        persist_dir=tmp_path / "empty_chroma",
        collection_name="empty_col",
    )
    retriever = SemanticRetriever(
        vector_store=empty_store,
        embedding_manager=MockEmbeddingManager(),
    )
    with pytest.raises(ValueError, match="Cannot retrieve from an empty vector collection"):
        retriever.retrieve("any query", top_k=1)


# 10. Retrieval results are deterministic
def test_deterministic_retrieval_results(populated_retriever: SemanticRetriever):
    run_1 = populated_retriever.retrieve("eval query", top_k=3)
    run_2 = populated_retriever.retrieve("eval query", top_k=3)

    assert len(run_1) == len(run_2)
    for r1, r2 in zip(run_1, run_2):
        assert r1.chunk_id == r2.chunk_id
        assert r1.rank == r2.rank
        assert pytest.approx(r1.score, rel=1e-6) == r2.score
        assert pytest.approx(r1.distance, rel=1e-6) == r2.distance
        assert r1.text == r2.text
        assert r1.page_number == r2.page_number
        assert r1.section == r2.section


# Module-level convenience functions
def test_convenience_functions(populated_retriever: SemanticRetriever):
    res = retrieve("eval", top_k=2, retriever=populated_retriever)
    assert len(res) == 2
    assert res[0].rank == 1
