# Enverus Technical Case Study: RAG Chatbot

A locally deployable **Retrieval-Augmented Generation (RAG)** system built to answer questions from a research paper while maintaining strict source grounding, numerical accuracy, and safe handling of insufficient evidence.

## Overview

This project implements an end-to-end RAG pipeline over the supplied research paper **"Agent-as-a-Judge: Evaluate Agents with Agents."**

The system combines:

- Page-aware PDF ingestion
- Structure-aware document chunking
- Sentence-transformer embeddings
- ChromaDB vector storage
- Dense semantic retrieval
- Local LLM generation through Ollama
- Evidence sufficiency checks
- Numerical and table-grounding validation
- Reproducible retrieval and end-to-end evaluation

The design prioritizes traceability and grounded answers over unsupported generation. When retrieved evidence is insufficient or ambiguous, the system abstains instead of generating an unsupported answer.

---

## RAG Architecture

![RAG Architecture](visualization/rag_workflow.png)

The pipeline consists of two main stages:

**Offline Ingestion & Indexing**

Research Paper → Page Extraction → Structure-Aware Chunking → Embeddings → Vector Store

**Online Retrieval & Generation**

User Query → Semantic Retrieval → Safety & Grounding → Local LLM → Answer + Sources

---

## Technical Approach

### 1. Document Ingestion

The supplied PDF is processed page-by-page using PyMuPDF.

The ingestion pipeline preserves:

- Page boundaries
- Page numbers
- Source metadata
- Extracted text

The supplied document contains **44 pages**.

### 2. Structure-Aware Chunking

The extracted document is divided into retrieval-friendly chunks using paragraph and block boundaries rather than arbitrary character slicing.

Configuration:

- Target chunk size: approximately 1,000 characters
- Primary maximum: 1,200 characters
- Overlap: 150 characters
- Small adjacent blocks are consolidated when appropriate

The resulting index contains **134 chunks** with page and section metadata.

### 3. Embeddings

Each chunk is converted into a dense vector using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Embedding characteristics:

- 384-dimensional vectors
- L2 normalization
- Deterministic embedding generation

### 4. Vector Storage

ChromaDB is used as the persistent vector store.

Configuration:

- Collection: `agent_as_a_judge`
- Distance metric: cosine
- Persistent local storage
- Idempotent indexing

### 5. Semantic Retrieval

The baseline retriever performs dense semantic search using the query embedding.

Default configuration: `top_k = 5`

The implementation keeps retrieval modular so alternative retrieval strategies can be evaluated independently.

### 6. Safety & Grounding

Retrieved evidence is validated before the generated response is accepted.

The safety layer checks for:

- Insufficient entity evidence
- Numerical ambiguity
- Unsupported numerical claims
- Table-context requirements
- Evidence grounding

When the evidence is insufficient, the system can return a deterministic fallback rather than forcing the LLM to answer.

### 7. Local LLM Generation

Answer generation is performed locally through Ollama using `qwen2.5:1.5b`.

The generation layer uses a grounding-focused system prompt and provides retrieved evidence as context, keeping the core inference workflow locally executable without requiring an external LLM API.

---

## Evaluation

The RAG pipeline was evaluated against a representative benchmark derived from the supplied research document.

The evaluation covers:

- Factual retrieval
- Numerical and table-based information
- Multi-entity comparisons
- Architectural information
- Component-level results
- Source-grounding validation
- Ambiguous or insufficient-evidence queries

### Retrieval Results

| Metric | Result |
|---|---:|
| Top-1 Retrieval Accuracy | 50.0% |
| Top-3 Retrieval Accuracy | 75.0% |
| Top-5 Retrieval Accuracy | 83.3% |
| Automated Test Suite | 99 passed |
| Source Grounding | Enabled |
| Numerical Safety Validation | Enabled |

The retrieval benchmark evaluates the dense semantic retrieval baseline without hidden reranking or query-expansion logic.

### End-to-End Evaluation

The end-to-end pipeline validates:

- Query processing
- Evidence retrieval
- Evidence sufficiency
- LLM generation
- Numerical grounding
- Source attribution
- Safe abstention when evidence is insufficient

Machine-readable evaluation artifacts are included in the repository for reproducibility.

### Evaluation Artifacts

- `retrieval_evaluation_results.json` — retrieval benchmark results
- `rag_e2e_results.json` — end-to-end RAG execution results
- `eval_retrieval.py` — retrieval evaluation runner
- `evaluate_question_bank.py` — automated evaluation runner

---

## Setup & Reproduction

```bash
git clone https://github.com/Dipakk7/enverus-rag-assignment.git
cd enverus-rag-assignment

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), start the local service, then pull the model:
```bash
ollama pull qwen2.5:1.5b
```

Build the index (extracts the paper, embeds it, populates ChromaDB):
```bash
python build_index.py
```

Run and verify:
```bash
pytest -q                          # 99 passed
python eval_retrieval.py           # retrieval accuracy
python test_real_rag.py            # end-to-end RAG
python test_real_ollama.py         # local generation check
python evaluate_question_bank.py   # full benchmark
```

---

## Project Structure

```
enverus-rag-assignment/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── build_index.py
├── eval_retrieval.py
├── evaluate_question_bank.py
├── test_real_ollama.py
├── test_real_rag.py
│
├── retrieval_evaluation_results.json
├── rag_e2e_results.json
│
├── data/
│   └── agent_as_a_judge.pdf
│
├── visualization/
│   ├── rag_workflow.png
│   └── README.md
│
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingest.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── safety.py
│   └── vector_store.py
│
└── tests/
    ├── test_chunking.py
    ├── test_embeddings.py
    ├── test_generator.py
    ├── test_ingest.py
    ├── test_rag_pipeline.py
    ├── test_retriever.py
    ├── test_table_safety.py
    └── test_vector_store.py
```

---

## Design Decisions

- **Local-first:** the full path — PDF → Embeddings → ChromaDB → Retrieval → Ollama → Grounded Answer — runs locally with no external inference API required.
- **Source-first generation:** the LLM isn't the source of truth. Retrieved evidence is passed to generation, and the safety layer can return a fallback when that evidence doesn't support an answer.
- **Numerical integrity:** numerical and table-based questions get extra validation, since a retrieval or generation error there can materially change the meaning of an answer.
- **Reproducibility:** deterministic document processing, reproducible indexing, automated tests, and machine-readable evaluation artifacts.

---

## Limitations

- The current retriever uses dense semantic retrieval as its baseline strategy.
- Retrieval quality depends on chunking and embedding quality.
- Local LLM inference performance depends on available hardware.
- Ambiguous questions may intentionally result in abstention when retrieved evidence is insufficient.
- The supplied research document is the authoritative knowledge source for this implementation; information outside that document is not assumed to be factual evidence.

---

## Privacy & Source Material

Private assessment questions, confidential instructions, and other proprietary evaluation material are intentionally not reproduced in this README. The supplied research paper is the sole knowledge source for the pipeline.

---

## Summary

This project demonstrates a complete, locally executable RAG workflow emphasizing reliable document processing, semantic retrieval, source-grounded generation, numerical and table safety, deterministic abstention, and reproducible evaluation.

The implementation focuses on correctness, traceability, and reproducibility without unnecessary application-layer complexity.
