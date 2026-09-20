"""
Evaluation and audit script for Phase 6.1 strict retrieval evaluation.
Runs the 12 benchmark queries against ChromaDB, evaluates each result against
explicit relevance criteria, and computes verified Top-1, Top-3, and Top-5 success rates.
"""

import json
import sys
from typing import Any, Dict, List
from src.retriever import get_retriever

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BENCHMARK_SPEC = [
    {
        "query_number": 1,
        "query": "DevAI dataset statistics",
        "relevance_criterion": (
            "Must state core quantitative dataset statistics for DevAI (e.g., 55 tasks, "
            "365 requirements, 125 preferences, distribution of query lengths, or lines of code)."
        ),
        "ground_truth_chunks": ["chunk_p04_002", "chunk_p04_001"],
    },
    {
        "query_number": 2,
        "query": "Agent-as-a-Judge evaluation time and cost savings",
        "relevance_criterion": (
            "Must explicitly state quantitative savings in time and/or cost achieved by "
            "Agent-as-a-Judge compared to human evaluation (e.g., 97.72% time savings, "
            "97.64% cost savings, or 86.5h / $1,297.5 vs 1.97h / $30.6)."
        ),
        "ground_truth_chunks": ["chunk_p11_002", "chunk_p03_001"],
    },
    {
        "query_number": 3,
        "query": "Three open-source agentic frameworks",
        "relevance_criterion": (
            "Must explicitly name all three open-source code agent frameworks evaluated "
            "in the paper (MetaGPT, GPT-Pilot, OpenHands)."
        ),
        "ground_truth_chunks": ["chunk_p02_006", "chunk_p05_003"],
    },
    {
        "query_number": 4,
        "query": "Average cost and time of MetaGPT, GPT-Pilot and OpenHands",
        "relevance_criterion": (
            "Must state the quantitative average cost and execution time figures for MetaGPT, "
            "GPT-Pilot, and OpenHands (e.g., Table 1 values: $1.19 / 775.29s, $3.92 / 1622.38s, $6.38 / 362.41s)."
        ),
        "ground_truth_chunks": ["chunk_p06_001", "chunk_p06_002"],
    },
    {
        "query_number": 5,
        "query": "Requirements Met and Task Solve Rate",
        "relevance_criterion": (
            "Must provide benchmark performance figures for both 'Requirements Met' and "
            "'Task Solve Rate' across frameworks (Table 2 baseline metrics: MetaGPT 5.67% / 0.0%, "
            "GPT-Pilot 29.25% / 1.82%, OpenHands 29.55% / 0.0%)."
        ),
        "ground_truth_chunks": ["chunk_p07_002"],
    },
    {
        "query_number": 6,
        "query": "OpenHands black-box alignment rate",
        "relevance_criterion": (
            "Must explicitly and unambiguously state the black-box alignment rate specifically "
            "for OpenHands (90.44% for Agent-as-a-Judge vs 60.38% for LLM-as-a-Judge), clearly "
            "distinguishing it from gray-box rates. Conflated/ambiguous prose does not qualify."
        ),
        "ground_truth_chunks": ["chunk_p09_002", "chunk_p09_001"],
    },
    {
        "query_number": 7,
        "query": "Agent-as-a-Judge component ablation",
        "relevance_criterion": (
            "Must present the component ablation setup or results analyzing the sequential "
            "impact of Agent-as-a-Judge modules (e.g., modular combination of (1), (2), (3), (5), (6), "
            "or Table 4 / Table 5 component ablations)."
        ),
        "ground_truth_chunks": ["chunk_p09_004", "chunk_p39_001", "chunk_p10_004"],
    },
    {
        "query_number": 8,
        "query": "Search module comparison",
        "relevance_criterion": (
            "Must present the comparative empirical evaluation of search algorithms used within "
            "the agent's search module (Table 6 comparison of BM25, grep, embeddings in Appendix K.2)."
        ),
        "ground_truth_chunks": ["chunk_p39_004"],
    },
    {
        "query_number": 9,
        "query": "SVM and LSTM architectures in DevAI",
        "relevance_criterion": (
            "Must provide evidence covering BOTH requested architecture terms (SVM and LSTM) "
            "within DevAI tasks or architecture distributions. A chunk with only one does not qualify."
        ),
        "ground_truth_chunks": ["chunk_p04_001"],
    },
    {
        "query_number": 10,
        "query": "Task 51 Requirement R1",
        "relevance_criterion": (
            "Must provide the specific content/specification of Requirement 1 (R1) for Task 51 "
            "(Figure 3 diagram: Requirement 1 Audio Loading)."
        ),
        "ground_truth_chunks": ["chunk_p05_001"],
    },
    {
        "query_number": 11,
        "query": "Human evaluator errors",
        "relevance_criterion": (
            "Must document or analyze errors, disagreements, or inconsistencies committed by "
            "human evaluators during assessment (Section 3.2 or Appendix H)."
        ),
        "ground_truth_chunks": ["chunk_p07_003", "chunk_p30_001", "chunk_p08_001", "chunk_p30_002"],
    },
    {
        "query_number": 12,
        "query": "Human-as-a-Judge limitations",
        "relevance_criterion": (
            "Must explicitly detail fundamental limitations of human evaluators (e.g., high labor "
            "cost, 86.5 hours total, $1,297.5 expense, fatigue, or substantial expertise requirement)."
        ),
        "ground_truth_chunks": ["chunk_p08_003", "chunk_p11_002"],
    },
]


def evaluate_benchmark():
    retriever = get_retriever()
    store = retriever.vector_store
    print(f"Auditing collection: {store.collection_name} ({store.count()} records)")

    audit_records = []
    top1_count = 0
    top3_count = 0
    top5_count = 0

    for spec in BENCHMARK_SPEC:
        qnum = spec["query_number"]
        query = spec["query"]
        crit = spec["relevance_criterion"]
        gt = set(spec["ground_truth_chunks"])

        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.chunk_id for r in results]

        # Determine strictly qualifying ranks
        qualifying_ranks = [
            r.rank for r in results if r.chunk_id in gt
        ]
        
        top1_hit = (results[0].chunk_id in gt)
        top3_hit = any(r.chunk_id in gt for r in results[:3])
        top5_hit = any(r.chunk_id in gt for r in results[:5])

        if top1_hit:
            top1_count += 1
        if top3_hit:
            top3_count += 1
        if top5_hit:
            top5_count += 1

        record = {
            "query_number": qnum,
            "query": query,
            "relevance_criterion": crit,
            "ground_truth_chunks": list(gt),
            "top1_success": top1_hit,
            "top3_success": top3_hit,
            "top5_success": top5_hit,
            "qualifying_ranks": qualifying_ranks,
            "results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "section": r.section,
                    "score": round(r.score, 4),
                    "distance": round(r.distance, 4),
                    "is_ground_truth": r.chunk_id in gt,
                    "preview": r.text.replace("\n", " ")[:160],
                }
                for r in results
            ],
        }
        audit_records.append(record)

    total = len(BENCHMARK_SPEC)
    metrics = {
        "total_queries": total,
        "top1_success_count": top1_count,
        "top1_success_rate": round(top1_count / total * 100, 1),
        "top3_success_count": top3_count,
        "top3_success_rate": round(top3_count / total * 100, 1),
        "top5_success_count": top5_count,
        "top5_success_rate": round(top5_count / total * 100, 1),
    }

    full_output = {
        "metrics": metrics,
        "evaluations": audit_records,
    }

    with open("retrieval_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"PHASE 6.1 AUDIT METRICS ({total} Queries)")
    print("=" * 60)
    print(f"Top-1 Success Rate: {metrics['top1_success_rate']}% ({top1_count}/{total})")
    print(f"Top-3 Success Rate: {metrics['top3_success_rate']}% ({top3_count}/{total})")
    print(f"Top-5 Success Rate: {metrics['top5_success_rate']}% ({top5_count}/{total})")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_benchmark()
