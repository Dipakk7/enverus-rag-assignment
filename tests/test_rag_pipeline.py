"""
Unit tests for the end-to-end RAG pipeline (src/rag_pipeline.py).

All tests in this suite mock the retriever and generator components,
guaranteeing isolated, fast, and deterministic offline execution.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.generator import (
    DEFAULT_SYSTEM_PROMPT,
    LLMGenerator,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from src.rag_pipeline import (
    RAGPipeline,
    RAGResponse,
    RAGSource,
    get_rag_pipeline,
)
from src.retriever import RetrievalResult, SemanticRetriever
from src.safety import INSUFFICIENT_EVIDENCE_RESPONSE


def make_mock_retrieval_result(
    rank: int,
    chunk_id: str,
    page_number: int,
    section: str,
    score: float,
    distance: float,
    text: str,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=chunk_id,
        page_number=page_number,
        section=section,
        score=score,
        distance=distance,
        text=text,
    )


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.85,
            distance=0.15,
            text="MetaGPT average cost was $1.19 and average runtime was 775.29s.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p06_002",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.75,
            distance=0.25,
            text="GPT-Pilot incurred higher cost of $3.92 with 1622.38s runtime.",
        ),
    ]
    return retriever


@pytest.fixture
def mock_generator() -> MagicMock:
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = (
        "MetaGPT cost $1.19 with a runtime of 775.29 seconds [chunk_p06_001]."
    )
    return generator


# 1. Empty question rejected & 2. Whitespace question rejected
def test_empty_and_whitespace_question_rejected(
    mock_retriever: MagicMock, mock_generator: MagicMock
):
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
        pipeline.answer("")

    with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
        pipeline.answer("   \n\t  ")

    with pytest.raises(TypeError, match="Question must be a string"):
        pipeline.answer(None)  # type: ignore

    with pytest.raises(TypeError, match="Question must be a string"):
        pipeline.answer(12345)  # type: ignore


# 3. Retriever is called & 4. Correct top_k passed to retriever
def test_retriever_called_with_correct_top_k(
    mock_retriever: MagicMock, mock_generator: MagicMock
):
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator, default_top_k=5)

    # Use explicit top_k=3
    pipeline.answer("What is the cost of MetaGPT?", top_k=3)
    mock_retriever.retrieve.assert_called_once_with(
        query="What is the cost of MetaGPT?",
        top_k=3,
    )

    # Use default top_k=5
    mock_retriever.reset_mock()
    pipeline.answer("What is the cost of MetaGPT?")
    mock_retriever.retrieve.assert_called_once_with(
        query="What is the cost of MetaGPT?",
        top_k=5,
    )


# 5. Retrieved chunks converted correctly to evidence & 6. build_rag_prompt receives correct question/evidence
# 7. LLMGenerator receives constructed prompt
def test_prompt_construction_and_generator_call(
    mock_retriever: MagicMock, mock_generator: MagicMock
):
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    with patch("src.rag_pipeline.build_rag_prompt") as mock_prompt_builder:
        mock_prompt_builder.return_value = "Constructed RAG Prompt with Evidence"

        response = pipeline.answer("What is the cost of MetaGPT?", top_k=2)

        # Check prompt builder invocation
        mock_prompt_builder.assert_called_once()
        call_args, call_kwargs = mock_prompt_builder.call_args
        assert call_kwargs["question"] == "What is the cost of MetaGPT?"
        assert len(call_kwargs["evidence"]) == 2

        # Check generator invocation
        mock_generator.generate.assert_called_once_with(
            prompt="Constructed RAG Prompt with Evidence",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )


# 8. Answer text returned correctly
def test_answer_text_returned_correctly(
    mock_retriever: MagicMock, mock_generator: MagicMock
):
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer("What is the cost of MetaGPT?")

    assert response.answer == "MetaGPT cost $1.19 with a runtime of 775.29 seconds [chunk_p06_001]."
    assert response.question == "What is the cost of MetaGPT?"
    assert response.model == "qwen2.5:1.5b"


# 9. Source metadata preserved & 10. Retrieval score/distance preserved
# 11. Multiple retrieved chunks preserved in rank order & 15. No evidence dropped
def test_sources_metadata_preservation_and_ordering(
    mock_retriever: MagicMock, mock_generator: MagicMock
):
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer("What is the cost of MetaGPT?")

    assert len(response.sources) == 2
    assert response.top_k == 2

    # Check Rank 1 source
    s1 = response.sources[0]
    assert isinstance(s1, RAGSource)
    assert s1.rank == 1
    assert s1.chunk_id == "chunk_p06_001"
    assert s1.page_number == 6
    assert s1.section == "2.3 Preliminary Benchmark"
    assert s1.score == 0.85
    assert s1.distance == 0.15
    assert "MetaGPT average cost was $1.19" in s1.text

    # Check Rank 2 source
    s2 = response.sources[1]
    assert s2.rank == 2
    assert s2.chunk_id == "chunk_p06_002"
    assert s2.page_number == 6
    assert s2.section == "2.3 Preliminary Benchmark"
    assert s2.score == 0.75
    assert s2.distance == 0.25
    assert "GPT-Pilot incurred higher cost" in s2.text


# 12. Retriever errors propagate clearly
def test_retriever_errors_propagate(
    mock_generator: MagicMock
):
    failing_retriever = MagicMock(spec=SemanticRetriever)
    failing_retriever.retrieve.side_effect = ValueError("Collection is empty")

    pipeline = RAGPipeline(retriever=failing_retriever, generator=mock_generator)

    with pytest.raises(ValueError, match="Collection is empty"):
        pipeline.answer("Any query")


# 13. Generator errors propagate clearly
def test_generator_errors_propagate(
    mock_retriever: MagicMock
):
    failing_generator = MagicMock(spec=LLMGenerator)
    failing_generator.generate.side_effect = OllamaConnectionError(
        "Unable to connect to Ollama at http://localhost:11434. Ensure Ollama is running."
    )

    pipeline = RAGPipeline(retriever=mock_retriever, generator=failing_generator)

    with pytest.raises(OllamaConnectionError, match="Unable to connect to Ollama"):
        pipeline.answer("Any query")


# 14. Response schema is correct
def test_response_schema_and_serialization(
    mock_retriever: MagicMock, mock_generator: MagicMock
):
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer("What is the cost of MetaGPT?")

    assert isinstance(response, RAGResponse)
    assert response.latency is not None
    assert response.latency >= 0.0

    d = response.to_dict()
    assert d["question"] == "What is the cost of MetaGPT?"
    assert d["answer"] == response.answer
    assert d["model"] == "qwen2.5:1.5b"
    assert d["top_k"] == 2
    assert len(d["sources"]) == 2
    assert d["sources"][0]["chunk_id"] == "chunk_p06_001"
    assert d["sources"][1]["chunk_id"] == "chunk_p06_002"


# Factory helper
def test_factory_helper():
    pipeline = get_rag_pipeline(default_top_k=3)
    assert isinstance(pipeline, RAGPipeline)
    assert pipeline.default_top_k == 3


# ==============================================================================
# PHASE 8.1 GROUNDING & NUMERICAL SAFETY REGRESSION TESTS
# ==============================================================================

# 15. Missing entity identifier triggers canonical insufficient evidence response
def test_missing_entity_identifier_triggers_insufficient_evidence():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p04_004",
            page_number=4,
            section="2.2 The DevAI Dataset",
            score=0.47,
            distance=0.53,
            text="The requirements belonging to each task represent a milestone arranged as a DAG.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p05_002",
            page_number=5,
            section="2.3 Preliminary Benchmark",
            score=0.44,
            distance=0.56,
            text="Task requirements in DevAI are structured as a Directed Acyclic Graph (DAG).",
        ),
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    # Query with specific numbered entities missing from retrieved chunks
    response = pipeline.answer("What does Task 51 Requirement R1 specify?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "Task 51" in str(response.safety_reason) or "51" in str(response.safety_reason)
    # Generator should not even be called when entity sufficiency check fails
    generator.generate.assert_not_called()
    # Retrieved sources must still be preserved for complete traceability
    assert len(response.sources) == 2
    assert response.sources[0].chunk_id == "chunk_p04_004"


# 16. Ambiguous numerical evidence across conflated settings triggers safety fallback
def test_ambiguous_numerical_evidence_triggers_safety_fallback():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p10_003",
            page_number=10,
            section="4.2 Judging",
            score=0.41,
            distance=0.59,
            text="Agent-as-a-Judge reaches 92.07% and 90.44%, surpassing LLM-as-a-Judge's 70.76% and 60.38% in both gray-box and black-box settings.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    # Generator attempts to confidently output an ambiguous candidate number
    generator.generate.return_value = "The OpenHands black-box alignment rate is 90.44%."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What is the OpenHands black-box alignment rate?")

    # Confident selection of ambiguous number must be intercepted
    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "conflates" in str(response.safety_reason).lower()
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "chunk_p10_003"


# 17. Explicitly supported numerical evidence passes through safely
def test_explicitly_supported_numerical_evidence_passes():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p10_003",
            page_number=10,
            section="4.2 Judging",
            score=0.88,
            distance=0.12,
            text="In the black-box setting, the alignment rate is 90.16%.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "The black-box alignment rate is 90.16% [chunk_p10_003]."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What is the black-box alignment rate?")

    assert response.answer == "The black-box alignment rate is 90.16% [chunk_p10_003]."
    assert response.safety_triggered is False
    assert response.safety_reason is None


# 18. Hallucinated / unsupported numbers are intercepted
def test_hallucinated_numerical_claims_intercepted():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.85,
            distance=0.15,
            text="MetaGPT average cost was $1.19 and runtime was 775.29s.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    # Model hallucinates a number not in evidence
    generator.generate.return_value = "MetaGPT achieved a 99.85% pass rate."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What is the pass rate of MetaGPT?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "unsupported numerical claims" in str(response.safety_reason).lower()


# 19. Source metadata and traceability remain preserved under safety triggers
def test_source_metadata_preserved_under_safety_trigger():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p01_001",
            page_number=1,
            section="Abstract",
            score=0.72,
            distance=0.28,
            text="Paper overview text without requested specific details.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What does Task 99 specify?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert len(response.sources) == 1
    s = response.sources[0]
    assert s.rank == 1
    assert s.chunk_id == "chunk_p01_001"
    assert s.page_number == 1
    assert s.section == "Abstract"
    assert s.score == 0.72
    assert s.distance == 0.28


# ==============================================================================
# PHASE 8.1.1 TABLE-CONTEXT & SEMANTIC VALIDATION TESTS
# ==============================================================================

# 20. Correct table-context answer passes when question and evidence align
def test_correct_table_context_answer_passes():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p09_002",
            page_number=9,
            section="4.1 Proof-of-Concept",
            score=0.82,
            distance=0.18,
            text="Table 3 AI Judges and Their Shift/Alignment with Human-as-a-Judge. Alignment rate is 88.52%.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "In Table 3, the alignment rate is 88.52% [chunk_p09_002]."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    # Question explicitly asks about Table 3
    response = pipeline.answer("What are the results in Table 3?")

    assert response.answer == "In Table 3, the alignment rate is 88.52% [chunk_p09_002]."
    assert response.safety_triggered is False
    assert response.safety_reason is None


# 21. Wrong-table numerical evidence is rejected
def test_wrong_table_numerical_evidence_rejected():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.79,
            distance=0.21,
            text="Table 1 Preliminary Statistics of AI Developers. Cost was $1.19.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "Table 4 cost was $1.19."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    # Question explicitly asks for Table 4, but evidence only has Table 1
    response = pipeline.answer("What are the costs in Table 4?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "Table 4" in str(response.safety_reason)


# 22. Ambiguous table context triggers canonical insufficiency fallback
def test_ambiguous_table_context_triggers_fallback():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p07_002",
            page_number=7,
            section="3.1 Benchmark Baselines",
            score=0.58,
            distance=0.42,
            text="The results of this experiment are shown in Table 2. GPT-Pilot and OpenHands satisfied about 29% of requirements.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p09_002",
            page_number=9,
            section="4.1 Proof-of-Concept",
            score=0.54,
            distance=0.46,
            text="Table 3 (continued): AI Judges and Their Shift/Alignment with Human-as-a-Judge. Requirements Met (I) is 25.40%.",
        ),
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "Requirements Met (I) was 25.40%."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    # Question asks generally for results without specifying Table 2 vs Table 3
    response = pipeline.answer("What are the Requirements Met and Task Solve Rate results?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "multiple distinct tables" in str(response.safety_reason)


# 23. Preservation of retrieved sources under table ambiguity
def test_sources_preserved_under_table_ambiguity():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p07_002",
            page_number=7,
            section="3.1 Benchmark Baselines",
            score=0.58,
            distance=0.42,
            text="The results of this experiment are shown in Table 2.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p09_002",
            page_number=9,
            section="4.1 Proof-of-Concept",
            score=0.54,
            distance=0.46,
            text="Table 3 (continued): AI Judges and Their Shift/Alignment.",
        ),
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the Requirements Met results?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert len(response.sources) == 2
    assert response.sources[0].chunk_id == "chunk_p07_002"
    assert response.sources[0].rank == 1
    assert response.sources[0].page_number == 7
    assert response.sources[1].chunk_id == "chunk_p09_002"
    assert response.sources[1].rank == 2
    assert response.sources[1].page_number == 9


