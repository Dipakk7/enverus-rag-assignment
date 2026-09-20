# Agent-as-a-Judge RAG — Enverus Technical Case Study

A locally deployable Retrieval-Augmented Generation system that answers questions about the paper **"Agent-as-a-Judge: Evaluate Agents with Agents"** ([arXiv:2410.10934](https://arxiv.org/abs/2410.10934)) and cites the exact page, section, and chunk each answer came from.

The design goal was not "get the LLM to answer." It was: **answer only when the retrieved evidence actually supports an answer, and abstain when it doesn't.** A deterministic safety layer sits between retrieval and generation, and again between generation and output.

Runs entirely on a local machine. No API keys, no external calls at inference time.

---

## At a Glance

| | |
| :--- | :--- |
| Corpus | 1 paper, 44 pages → 134 structure-aware chunks |
| Embeddings | `all-MiniLM-L6-v2`, 384-dim, L2-normalized |
| Vector store | ChromaDB, cosine (HNSW), persistent |
| Generation | `qwen2.5:1.5b` via Ollama, `temperature=0.0` |
| Retrieval (12-query eval set) | Top-1 **50.0%** · Top-3 **75.0%** · Top-5 **83.3%** |
| Benchmark | 12/12 questions grounded; 3 correctly abstained |
| Tests | 98 passing (`pytest`) |
| Dependencies | Pure Python + PyMuPDF + Sentence-Transformers + ChromaDB. No LangChain / LlamaIndex. |

---

## Quickstart

```bash
git clone <REPOSITORY_URL>
cd enverus-rag-assignment

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env               # Windows: Copy-Item .env.example .env

ollama serve                       # if not already running
ollama pull qwen2.5:1.5b

python build_index.py              # 44 pages → 134 chunks → chroma_db/
python test_real_rag.py            # end-to-end sanity run
```

Requires Python 3.10–3.13 (verified on 3.13), Ollama, and 8 GB+ RAM. CPU-only works; a GPU mainly cuts generation latency.

---

## Example Output

```python
from src.rag_pipeline import get_rag_pipeline

pipeline = get_rag_pipeline(default_top_k=5)
response = pipeline.answer(
    "How much time and cost does Agent-as-a-Judge save compared with human experts?"
)

print(response.answer)
for s in response.sources:
    print(f"[{s.rank}] {s.chunk_id} · p.{s.page_number} · {s.section} · {s.score:.4f}")
```

```
<paste your actual run output here>

Sources:
[1] chunk_p07_02 · p.7 · 4 Agent-as-a-Judge · 0.8412
[2] chunk_p06_04 · p.6 · 3 Human-as-a-Judge · 0.7903
...
```

And when the evidence doesn't support an answer:

```python
pipeline.answer("What is Requirement R1 of Task 51?")
# → "The provided evidence is insufficient to answer this question."
#    Safety layer halted before generation (missing entity: 'Task 51').
```

---

## Architecture

Offline indexing is decoupled from online retrieval — the PDF is processed once, then only the query is embedded per request.

```
OFFLINE                                    ONLINE
─────────────────────────────              ─────────────────────────────
agent_as_a_judge.pdf (44 pp)               User query
        │                                          │
  PyMuPDF page extraction                   Query embedding (384-dim)
  (src/ingest.py)                           (src/embeddings.py)
        │                                          │
  Structure-aware chunking  ──▶ 134                │
  (src/chunking.py)              chunks            │
        │                                          │
  Dense embeddings                                 │
  (src/embeddings.py)                              │
        │                                          │
        ▼                                          │
   ChromaDB  ◀────── cosine / HNSW top-k ──────────┘
   collection: agent_as_a_judge
   (src/vector_store.py)
        │
        ▼
   Top-5 evidence chunks (src/retriever.py)
        │
        ▼
   ┌───────────────────────────────────────┐
   │  PRE-GENERATION SAFETY (src/safety.py)│
   │   · entity sufficiency                │
   │   · table-context disambiguation      │
   │   · setting conflation                │
   └───────────────────────────────────────┘
        │                        │
   insufficient            sufficient
        │                        ▼
        │              Qwen 2.5 1.5B via Ollama
        │              (src/generator.py, temp 0.0)
        │                        │
        │                        ▼
        │              ┌──────────────────────────┐
        │              │ POST-GENERATION SAFETY   │
        │              │  · numerical audit       │
        │              └──────────────────────────┘
        │                        │
        ▼                        ▼
   Canonical abstention   Grounded answer + citations
```

A standalone diagram is in [`visualization/rag_workflow.png`](visualization/rag_workflow.png).

---

## Pipeline Components

**`src/ingest.py` — PDF ingestion.** Extracts text page-by-page with PyMuPDF into `PageRecord` objects, preserving 1-indexed page boundaries. Handles empty pages, file validation, and page-count verification.

**`src/chunking.py` — Structure-aware chunking.** Rather than fixed-length slicing, chunks respect section hierarchy (`1 Introduction`, `2 DevAI…`, `Appendix A–K`) and keep Markdown table structures intact (Tables 1, 2, 3, 4, 6). Target ~1000 chars, max ~1200, overlap ~150. Chunk IDs are deterministic: `chunk_p{page}_{index}`.

*Why this matters:* several benchmark questions depend on table values. If a table is split mid-row, retrieval can return `$0.12` with no indication of what it measures — a grounding failure the LLM cannot recover from.

**`src/embeddings.py` — Dense embeddings.** `all-MiniLM-L6-v2` maps text to 384-dim vectors with unit-length L2 normalization, so inner product equals cosine similarity. Batched for indexing.

**`src/vector_store.py` — ChromaDB.** Persistent local store at `chroma_db/`, collection `agent_as_a_judge`, `{"hnsw:space": "cosine"}`. Upserts are idempotent, so re-indexing doesn't duplicate entries.

**`src/retriever.py` — Dense retrieval.** Validates the query, embeds it, runs HNSW nearest-neighbor search, converts distance to similarity (`s = 1.0 − distance`), and returns `RetrievalResult` objects carrying rank, chunk ID, page, section, text, and score. Top-5 by default — one chunk often holds a metric while another holds its definition or evaluation setting.

**`src/generator.py` — Local generation.** Calls the Ollama daemon at `http://localhost:11434/api/generate` with `qwen2.5:1.5b` at `temperature=0.0`. The system prompt enforces a 10-point contract forbidding extrapolation, outside knowledge, and unsupported figures.

---

## The Safety Layer

`src/safety.py` runs deterministic checks — no model calls, no probabilistic judgement — on both sides of generation.

**Entity sufficiency (pre-gen).** If a query names a specific identifier (`Task 51`, `Requirement R1`) that does not appear in the retrieved evidence, generation is halted at 0.0s latency and the canonical fallback is returned:

> *"The provided evidence is insufficient to answer this question."*

This blocks the most common RAG failure mode: retrieval misses the target, the model recognises something adjacent, and produces a fluent wrong answer.

**Table-context disambiguation (pre-gen).** Detects queries asking generically for "results" or "metrics" when retrieved chunks span tables with different semantics — Table 2 (developer baselines) vs Table 3 (AI Judge evaluations). Prevents metric substitution across tables.

**Setting conflation (pre-gen).** Detects evidence presenting numbers under multiple evaluation settings (`gray-box`, `black-box`) without 1:1 attribution. Without this, an answer can state a figure without saying which setting it belongs to — technically present in the source, factually misleading.

**Numerical audit (post-gen).** Extracts every number and percentage from the generated answer and verifies each is explicitly supported by the retrieved text. Unsupported figures block the response. Research papers are dense with costs, percentages, and benchmark values, and a 1.5B model is exactly the size where those get transposed.

---

## Benchmark Results

All 12 assignment questions run through the live pipeline via `evaluate_question_bank.py`.

**Three of these questions are designed to be unanswerable from the retrieved evidence.** For Q05, Q06, and Q10, `TRIGGERED` is the correct outcome — the system declining to guess, not the system failing.

| ID | Query Concept | Target Context | Evidence | Grounding | Safety |
| :-: | :--- | :--- | :---: | :---: | :--- |
| Q01 | DevAI dataset statistics | DevAI baseline | SUFFICIENT | GROUNDED | PASS |
| Q02 | Agent-as-a-Judge savings | Human vs Agent-as-a-Judge | SUFFICIENT | GROUNDED | PASS |
| Q03 | Three agentic frameworks | AI developer baselines | SUFFICIENT | GROUNDED | PASS |
| Q04 | Average cost & time | Table 1 | SUFFICIENT | GROUNDED | PASS *(Table 1 resolved)* |
| Q05 | Requirements met & solve rate | Table 2 vs Table 3 | INSUFFICIENT_OR_AMBIGUOUS | GROUNDED | **TRIGGERED** *(Table 2 absent, Table 3 blocked)* |
| Q06 | OpenHands black-box alignment | Table 3 | INSUFFICIENT_OR_AMBIGUOUS | GROUNDED | **TRIGGERED** *(conflated settings)* |
| Q07 | Component ablations | Table 4 / §4.3 | SUFFICIENT | GROUNDED | PASS |
| Q08 | Search module comparison | Appendix K.2 / Table 6 | SUFFICIENT | GROUNDED | PASS |
| Q09 | SVM and LSTM architectures | Figure 2 | SUFFICIENT | GROUNDED | PASS |
| Q10 | Task 51 Requirement R1 | Figure 3 DAG | INSUFFICIENT_OR_AMBIGUOUS | GROUNDED | **TRIGGERED** *(missing entity 'Task 51')* |
| Q11 | Human evaluator errors | §3.2 / Appendix H | SUFFICIENT | GROUNDED | PASS |
| Q12 | Human-as-a-Judge limitations | §4 / §4.4 | SUFFICIENT | GROUNDED | PASS |

Zero ungrounded answers across all 12.

---

## Retrieval Evaluation

`eval_retrieval.py` measures whether the gold chunk appears in the retrieved set, writing results to `retrieval_evaluation_results.json`.

| Metric | Result |
| :--- | :---: |
| Top-1 | 50.0% |
| Top-3 | 75.0% |
| Top-5 | 83.3% |

**Read these as directional, not as a performance claim.** The evaluation set is the 12 benchmark queries, so each question moves the number by ~8 points and confidence intervals are wide. They are reported to show the retrieval layer is measured rather than assumed, and to justify the Top-5 default: Top-1 alone would miss half the gold chunks.

---

## Reproduction

```bash
python build_index.py               # rebuild vector store
python -m pytest -q                 # 98 passed in ~35s
python test_real_rag.py             # end-to-end representative queries
python evaluate_question_bank.py    # 12-question benchmark
python eval_retrieval.py            # retrieval metrics
```

Configuration lives in `.env`:

```ini
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=agent_as_a_judge
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TEMPERATURE=0.0
OLLAMA_TIMEOUT=60.0
```

---

## Repository Structure

```
enverus-rag-assignment/
├── data/agent_as_a_judge.pdf
├── src/
│   ├── ingest.py            # PyMuPDF page extraction
│   ├── chunking.py          # structure-aware chunking
│   ├── embeddings.py        # all-MiniLM-L6-v2
│   ├── vector_store.py      # ChromaDB
│   ├── retriever.py         # dense retrieval
│   ├── safety.py            # deterministic guardrails
│   ├── generator.py         # Ollama / Qwen 2.5 1.5B
│   └── rag_pipeline.py      # orchestration
├── tests/                   # 98 pytest cases
├── visualization/
│   ├── rag_workflow.png
│   └── README.md
├── build_index.py
├── evaluate_question_bank.py
├── eval_retrieval.py
├── test_real_rag.py
├── test_real_ollama.py
├── retrieval_evaluation_results.json
├── rag_e2e_results.json
├── requirements.txt
└── .env.example
```

---

## Design Decisions

**No orchestration framework.** The assignment turns on retrieval quality, grounding, and safety — all of which I needed to control explicitly. LangChain or LlamaIndex would have added abstraction between me and the exact logic being evaluated, with no capability I was missing.

**A 1.5B local model.** Qwen 2.5 1.5B runs on commodity hardware with no API dependency, which makes the whole system reproducible on a reviewer's laptop. The tradeoff is weaker reasoning — which is precisely why the safety layer is deterministic rather than model-based. The guardrails do not inherit the generator's weaknesses.

**Temperature 0.0.** Factual QA over a fixed corpus has no use for sampling diversity. Determinism also makes the benchmark reproducible run-to-run.

**No UI.** The deliverable is the pipeline and its evaluation. Interfaces are the Python API (`get_rag_pipeline()`), the CLI runner (`test_real_rag.py`), and the benchmark runner (`evaluate_question_bank.py`).

---

## Known Limitations

1. **Dense-only retrieval.** No BM25 sparse fusion. Rare alphanumeric tokens (`Task 51`, `R1`, specific table labels) are exactly where lexical matching outperforms embeddings, and exactly where Top-1 currently misses. Hybrid BM25 + dense ranking is the highest-value next change.
2. **Single-paper corpus.** Chunking heuristics are tuned to one academic paper. Multi-document retrieval would need document-level routing before chunk-level search.
3. **Conservative guardrails.** The safety layer prefers abstention over approximation. A larger model with looser checks would answer more questions — some of them wrongly. This tradeoff is intentional, but it is a tradeoff.
4. **CPU inference latency.** 8–15s for single-paragraph answers, up to ~40s for multi-point responses. CUDA or Apple Silicon MPS reduces this substantially.

---

## Source

Zhuge, M., Liu, C., Liu, H., You, J., et al. *Agent-as-a-Judge: Evaluate Agents with Agents*, October 2024. [arXiv:2410.10934](https://arxiv.org/abs/2410.10934)
