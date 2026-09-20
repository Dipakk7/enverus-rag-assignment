"""
Tests for table-context safety and semantic disambiguation (Phase 9.1).
"""

from unittest.mock import MagicMock
from src.retriever import RetrievalResult, SemanticRetriever
from src.generator import LLMGenerator
from src.rag_pipeline import RAGPipeline
from src.safety import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    check_table_context_ambiguity,
    check_entity_sufficiency,
    check_numerical_ambiguity,
    validate_rag_grounding,
)


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
        chunk_id=chunk_id,
        page_number=page_number,
        section=section,
        text=text,
        score=score,
        distance=distance,
        rank=rank,
    )


# 1. Relevant table + incidental unrelated table does NOT falsely trigger
def test_relevant_table_with_incidental_unrelated_table_passes():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.88,
            distance=0.12,
            text="Table 1 Preliminary Statistics of AI Developers. MetaGPT average cost is $1.19 and average time is 775.29s.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p07_002",
            page_number=7,
            section="3.1 Benchmark Baselines",
            score=0.65,
            distance=0.35,
            text="The results of this experiment are shown in Table 2. GPT-Pilot satisfied 29% of requirements.",
        ),
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "MetaGPT average cost is $1.19 and average time is 775.29s [chunk_p06_001]."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the average cost and execution time figures of MetaGPT?")

    # Incidental mention of Table 2 must NOT block valid Table 1 cost/time answer
    assert response.answer == "MetaGPT average cost is $1.19 and average time is 775.29s [chunk_p06_001]."
    assert response.safety_triggered is False
    assert response.safety_reason is None


# 2. Component ablation with incidental Table 3 mention passes
def test_component_ablation_with_incidental_table3_passes():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p10_004",
            page_number=10,
            section="4.3 Ablations",
            score=0.85,
            distance=0.15,
            text="Table 4 Component Ablation Studies for Agent-as-a-Judge. With ask: 65.03%, with locate: 90.44%.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p10_002",
            page_number=10,
            section="4.2 Judging",
            score=0.55,
            distance=0.45,
            text="As shown in table 3, Agent-as-a-Judge consistently outperforms LLM-as-a-Judge across tasks.",
        ),
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "Table 4 ablation results show ask at 65.03% and locate at 90.44% [chunk_p10_004]."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the component ablation results for Agent-as-a-Judge?")

    assert response.answer == "Table 4 ablation results show ask at 65.03% and locate at 90.44% [chunk_p10_004]."
    assert response.safety_triggered is False
    assert response.safety_reason is None


# 3. Wrong table does NOT satisfy requested context
def test_wrong_table_does_not_satisfy_requested_table():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.79,
            distance=0.21,
            text="Table 1 Preliminary Statistics of AI Developers. MetaGPT cost was $1.19.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "Table 4 cost was $1.19."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the costs in Table 4?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "Table" in str(response.safety_reason)


# 4. Ambiguous / conflicting table context still triggers fallback
def test_ambiguous_conflicting_tables_still_triggers_fallback():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p07_002",
            page_number=7,
            section="3.1 Benchmark Baselines",
            score=0.60,
            distance=0.40,
            text="The results of this experiment are shown in Table 2. GPT-Pilot and OpenHands satisfied about 29% of requirements.",
        ),
        make_mock_retrieval_result(
            rank=2,
            chunk_id="chunk_p09_002",
            page_number=9,
            section="4.1 Proof-of-Concept",
            score=0.55,
            distance=0.45,
            text="Table 3 (continued): AI Judges and Their Shift/Alignment with Human-as-a-Judge. Requirements Met (I) is 25.40%.",
        ),
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "Requirements Met (I) is 25.40%."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the Requirements Met and Task Solve Rate results?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "multiple distinct tables" in str(response.safety_reason)


# 5. Correct setting vs wrong setting (conflated black-box/gray-box prose triggers fallback)
def test_conflated_setting_without_attribution_triggers():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p10_003",
            page_number=10,
            section="4.2 Judging",
            score=0.50,
            distance=0.50,
            text="Agent-as-a-Judge reaches 92.07% and 90.44% in both gray-box and black-box settings.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "The OpenHands black-box alignment rate is 90.44%."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What is the OpenHands black-box alignment rate?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "conflates" in str(response.safety_reason).lower()


# 6. Correct metric vs unrelated metric
def test_unrelated_metric_does_not_cause_conflict():
    evidence = [
        "Table 1 Preliminary Statistics of AI Developers. Average Cost is $1.19. Average Time is 775.29s.",
        "Table 2 Human-as-a-Judge for AI Developers. Requirements Met rate is 22.13%.",
    ]
    # Query asks specifically for cost
    is_ambiguous, reason = check_table_context_ambiguity(
        question="What is the average cost?",
        evidence_texts=evidence,
        candidate_answer="The average cost is $1.19.",
    )
    assert is_ambiguous is False
    assert reason is None


# 7. Q10 missing entity still triggers fallback at 0.0s
def test_q10_missing_entity_still_triggers_fallback():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p04_004",
            page_number=4,
            section="2.2 The DevAI Dataset",
            score=0.47,
            distance=0.53,
            text="Requirements represent a milestone in the development process arranged as a DAG.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What does Task 51 Requirement R1 specify?")

    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "Task 51" in str(response.safety_reason)
    assert response.latency == 0.0
    generator.generate.assert_not_called()


# 8. Source metadata preservation
def test_source_metadata_preserved_on_all_responses():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.88,
            distance=0.12,
            text="Table 1 Preliminary Statistics of AI Developers. Cost is $1.19.",
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = "Cost is $1.19."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What is the cost?")

    assert len(response.sources) == 1
    src = response.sources[0]
    assert src.rank == 1
    assert src.chunk_id == "chunk_p06_001"
    assert src.page_number == 6
    assert src.section == "2.3 Preliminary Benchmark"
    assert src.score == 0.88
    assert src.distance == 0.12
    assert "Table 1" in src.text


# 9. Incorrect Table 1 execution times rejected when authoritative evidence contains different values
def test_incorrect_table1_execution_times_rejected_when_authoritative_evidence_differs():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.88,
            distance=0.12,
            text=(
                "Table 1 Preliminary Statistics of AI Developers.\n"
                "(1) Average Cost: MetaGPT $1.19, GPT-Pilot $3.92, OpenHands $6.38\n"
                "(2) Average Time: MetaGPT 775.29s, GPT-Pilot 1622.38s, OpenHands 362.41s"
            ),
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    # Model generates the erroneous 3139.73s / 951.35s numbers not in the authoritative Table 1
    generator.generate.return_value = (
        "MetaGPT: $1.19 / 775.29s, GPT-Pilot: $3.92 / 3139.73s, OpenHands: $6.38 / 951.35s [chunk_p06_001]."
    )

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the average cost and execution time figures of MetaGPT, GPT-Pilot, and OpenHands?")

    # Incorrect execution-time values not present in evidence must be rejected by safety layer
    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert response.safety_triggered is True
    assert "unsupported numerical claims" in str(response.safety_reason).lower()


# 10. Authoritative Table 1 execution times accepted
def test_authoritative_table1_execution_times_accepted():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id="chunk_p06_001",
            page_number=6,
            section="2.3 Preliminary Benchmark",
            score=0.88,
            distance=0.12,
            text=(
                "Table 1 Preliminary Statistics of AI Developers.\n"
                "(1) Average Cost: MetaGPT $1.19, GPT-Pilot $3.92, OpenHands $6.38\n"
                "(2) Average Time: MetaGPT 775.29s, GPT-Pilot 1622.38s, OpenHands 362.41s"
            ),
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = (
        "MetaGPT average cost is $1.19 (775.29s), GPT-Pilot is $3.92 (1622.38s), OpenHands is $6.38 (362.41s) [chunk_p06_001]."
    )

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the average cost and execution time figures of MetaGPT, GPT-Pilot, and OpenHands?")

    assert response.safety_triggered is False
    assert "1622.38" in response.answer
    assert "362.41" in response.answer
    assert "775.29" in response.answer


# 11. End-to-end regression: authoritative chunk_p06_001 verifies exact Table 1 execution times and rejects discrepancies
def test_live_chunk_table1_numerical_verification_and_regression():
    from src.ingest import extract_pages_from_pdf
    from src.chunking import chunk_page_records

    # 1. Verify physical extraction of chunk_p06_001 directly from PDF
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)
    chunk_p06 = next((c for c in chunks if c.chunk_id == "chunk_p06_001"), None)
    assert chunk_p06 is not None

    # Confirm ground truth values actually present in the authoritative chunk
    assert "775.29s" in chunk_p06.text
    assert "1622.38s" in chunk_p06.text
    assert "362.41s" in chunk_p06.text
    assert "3139.73" not in chunk_p06.text
    assert "951.35" not in chunk_p06.text

    # 2. Assert pipeline rejects answers containing incorrect/hallucinated execution times
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id=chunk_p06.chunk_id,
            page_number=chunk_p06.page_number,
            section=chunk_p06.section,
            score=0.92,
            distance=0.08,
            text=chunk_p06.text,
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = (
        "MetaGPT: $1.19 / 775.29s, GPT-Pilot: $3.92 / 3139.73s, OpenHands: $6.38 / 951.35s [chunk_p06_001]."
    )

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the average cost and execution time figures of MetaGPT, GPT-Pilot, and OpenHands?")
    assert response.safety_triggered is True
    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert "unsupported numerical claims" in str(response.safety_reason).lower()

    # 3. Assert pipeline accepts answers containing the exact authoritative execution times
    generator.generate.return_value = (
        "MetaGPT: $1.19 / 775.29s, GPT-Pilot: $3.92 / 1622.38s, OpenHands: $6.38 / 362.41s [chunk_p06_001]."
    )
    response_ok = pipeline.answer("What are the average cost and execution time figures of MetaGPT, GPT-Pilot, and OpenHands?")
    assert response_ok.safety_triggered is False
    assert "1622.38" in response_ok.answer
    assert "362.41" in response_ok.answer
    assert "775.29" in response_ok.answer


# 12. End-to-end regression: authoritative chunk_p04_002 verifies DevAI dataset statistics and rejects discrepancies
def test_live_chunk_q01_devai_dataset_statistics_regression():
    from src.ingest import extract_pages_from_pdf
    from src.chunking import chunk_page_records

    # 1. Verify physical extraction of chunk_p04_002 directly from PDF
    records = extract_pages_from_pdf()
    chunks = chunk_page_records(records)
    chunk_p04 = next((c for c in chunks if c.chunk_id == "chunk_p04_002"), None)
    assert chunk_p04 is not None

    # Confirm authoritative values in chunk_p04_002 (Page 4, Section 2.2)
    assert "55" in chunk_p04.text
    assert "365 requirements" in chunk_p04.text
    assert "125 preferences" in chunk_p04.text
    assert "335" not in chunk_p04.text

    # 2. Assert pipeline rejects answers containing incorrect/hallucinated requirement counts (e.g. 335)
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.retrieve.return_value = [
        make_mock_retrieval_result(
            rank=1,
            chunk_id=chunk_p04.chunk_id,
            page_number=chunk_p04.page_number,
            section=chunk_p04.section,
            score=0.90,
            distance=0.10,
            text=chunk_p04.text,
        )
    ]
    generator = MagicMock(spec=LLMGenerator)
    generator.model = "qwen2.5:1.5b"
    generator.generate.return_value = (
        "DevAI consists of 55 tasks, 335 requirements, and 125 preferences [chunk_p04_002]."
    )

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.answer("What are the core quantitative dataset statistics for DevAI?")
    assert response.safety_triggered is True
    assert response.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert "unsupported numerical claims" in str(response.safety_reason).lower()

    # 3. Assert pipeline accepts answers containing the exact authoritative dataset statistics
    generator.generate.return_value = (
        "DevAI consists of 55 tasks, 365 requirements, and 125 preferences [chunk_p04_002]."
    )
    response_ok = pipeline.answer("What are the core quantitative dataset statistics for DevAI?")
    assert response_ok.safety_triggered is False
    assert "55" in response_ok.answer
    assert "365" in response_ok.answer
    assert "125" in response_ok.answer


