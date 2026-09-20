# Enverus Technical Case Study: RAG Chatbot

## Project Overview
This repository contains a production-grade, locally deployable Retrieval-Augmented Generation (RAG) system built for the **Enverus Product Intern — Machine Learning & Gen-AI** case study.

The system is designed to ingest, index, retrieve, and answer complex research queries about the seminal AI evaluation paper:
> **"Agent-as-a-Judge: Evaluate Agents with Agents"**  
> *Mingchen Zhuge, Changsheng Liu, Haozhe Liu, Jiaxuan You, et al. (October 2024)*  
> [arXiv:2410.10934](https://arxiv.org/abs/2410.10934)

The pipeline enables verifiable, source-grounded question answering over the paper's 44 pages, methodology, empirical findings, and complex benchmark tables, accompanied by explicit source chunk citations, page numbers, and cosine similarity scores.

---

## Assignment Objective & Principles
- **Lean & Modular Architecture:** Implemented with pure Python, PyMuPDF, Sentence-Transformers, ChromaDB, and Ollama without cumbersome orchestration frameworks (no LangChain / LlamaIndex overhead).
- **Strict Source Grounding:** Enforces a 10-point anti-hallucination prompt contract and a deterministic pre/post-generation Safety Layer (`src/safety.py`) to eliminate extrapolation.
- **Table-Context & Numerical Integrity:** Resolves nuanced table collisions (e.g. Table 1 developer costs vs Table 2/3 metrics) and verifies all numerical claims against retrieved evidence.
- **100% Local & Private Execution:** Operates entirely locally using open-weights embedding models (`all-MiniLM-L6-v2`) and a local SLM (`qwen2.5:1.5b` via Ollama) without external API costs or external telemetry.
- **Full Traceability:** Every response preserves full citation provenance (Chunk ID, 1-indexed page number, section title, and cosine similarity score).

---

## System Architecture & Workflow

The pipeline decouples **Offline Document Ingestion & Indexing** from **Online Grounded Retrieval & Generation**:

![RAG Architecture](visualization/rag_workflow.png)

*(A standalone architecture diagram is located in [`visualization/rag_workflow.png`](visualization/rag_workflow.png) with detailed notes in [`visualization/README.md`](visualization/README.md).)*

```
[ Research Paper PDF (44 pages) ]
              │
              ▼
[ Page Extraction: PyMuPDF (src/ingest.py) ]
              │
              ▼
[ Structure-Aware Chunking (src/chunking.py) ] ──▶ 134 Chunks with Headings & Tables
              │
              ▼
[ Dense Embeddings: all-MiniLM-L6-v2 (src/embeddings.py) ] ──▶ 384-dim Vectors (L2 norm)
              │
              ▼
[ Vector Store: ChromaDB (src/vector_store.py) ] ──▶ Collection: agent_as_a_judge
              ▲
              │ Cosine Nearest-Neighbor Search
              │
[ User Query ] ──▶ [ Dense Semantic Retrieval (src/retriever.py) ]
                          │
                          ▼
            [ Top-5 Retrieved Evidence Chunks ]
                          │
                          ▼
            [ Deterministic Safety Layer (src/safety.py) ]
              ├─ Pre-Gen Entity Sufficiency Check (e.g., Task 51 R1)
              ├─ Table-Context & Semantic Disambiguation (Table 2 vs 3)
              └─ Setting Conflation & Numerical Safety Audit
                          │
                          ▼
            [ Local LLM Generator: Qwen 2.5 1.5B (src/generator.py via Ollama) ]
                          │
                          ▼
            [ Validated Grounded Answer + Source Citations (RAGResponse) ]
```

---

## Core Pipeline Components

### 1. PDF Ingestion (`src/ingest.py`)
- Extracts full-text content page-by-page from `data/agent_as_a_judge.pdf` using PyMuPDF (`fitz`).
- Preserves exact 1-indexed document page boundaries into `PageRecord` objects.
- Handles edge cases including empty pages, file validation, and page count verification (44 pages total).

### 2. Structure-Aware Chunking (`src/chunking.py`)
- Transforms 44 page records into 134 semantically coherent `Chunk` objects.
- Preserves section hierarchies (`1 Introduction`, `2 DevAI...`, `3 Human-as-a-Judge...`, `Appendix A-K`).
- Identifies and retains Markdown tabular structures intact (e.g. Table 1, Table 2, Table 3, Table 4, Table 6).
- Formats deterministic identifiers (`chunk_p{page}_{index}`).

### 3. Dense Embedding Module (`src/embeddings.py`)
- Employs `sentence-transformers/all-MiniLM-L6-v2` locally to map text into 384-dimensional dense vectors.
- Applies unit-length L2 normalization to ensure that inner product matches cosine similarity.
- Supports batch embedding for efficient indexing.

### 4. ChromaDB Vector Store (`src/vector_store.py`)
- Manages a persistent local ChromaDB database in `chroma_db/`.
- Collection: `agent_as_a_judge` with `{"hnsw:space": "cosine"}` metric.
- Idempotent upserts prevent duplicate entries on re-indexing.

### 5. Dense Semantic Retriever (`src/retriever.py`)
- Validates query input and computes 384-dimensional query embedding.
- Queries ChromaDB using HNSW nearest-neighbor search to retrieve top-$k$ evidence chunks.
- Computes cosine similarity score: $s = 1.0 - \text{distance} \in [-1, 1]$.
- Returns structured `RetrievalResult` objects preserving rank, chunk ID, page number, section, and text.

### 6. Local LLM Generator (`src/generator.py`)
- Connects to local Ollama daemon via HTTP (`http://localhost:11434/api/generate`).
- Model: `qwen2.5:1.5b` with `temperature: 0.0` for deterministic factual generation.
- Enforces a 10-point system prompt contract forbidding extrapolation, outside knowledge, or fabricated figures.

### 7. Grounding, Numerical Safety & Table Disambiguation (`src/safety.py`)
A deterministic guardrail layer operating both pre- and post-generation:
- **Entity Sufficiency Check:** Checks if specific requested entities/identifiers (e.g., `Task 51`, `Requirement R1`) exist in the evidence. If missing, immediately halts generation (0.0s latency) with the canonical fallback:
  > *"The provided evidence is insufficient to answer this question."*
- **Table-Context Disambiguation:** Detects queries asking generally for benchmark results/metrics when retrieved evidence spans conflicting tables (e.g., Table 2 developer baselines vs Table 3 AI Judge evaluations), preventing incorrect metric substitution.
- **Setting Conflation Check:** Detects sentences presenting numbers across multiple evaluation settings (e.g. `gray-box` and `black-box`) without explicit 1:1 attribution, blocking arbitrary number guessing.
- **Numerical Hallucination Audit:** Extracts all percentages and numbers from candidate answers and verifies that each figure is explicitly supported in the retrieved text.

### 8. Application Interface Status
- **Implemented Interfaces:**
  - **Programmatic Python API:** [`get_rag_pipeline()`](src/rag_pipeline.py) provides full `answer(question, top_k)` capabilities.
  - **CLI Real RAG Runner:** [`test_real_rag.py`](test_real_rag.py) runs representative test queries against the live pipeline.
  - **Automated Benchmark Runner:** [`evaluate_question_bank.py`](evaluate_question_bank.py) executes the full 12-question assignment bank.
- *Streamlit UI Note:* In accordance with the assignment's modular requirements, an interactive graphical Streamlit UI was not implemented to keep dependencies minimal, clean, and focused on core pipeline reproducibility and grounding.

---

## Authoritative 12-Question Benchmark Results

All 12 assignment questions have been evaluated against the authoritative source paper:

| ID | Query Concept | Target Context | Evidence Status | Grounding Status | Safety Layer |
| :-: | :--- | :--- | :---: | :---: | :---: |
| **Q01** | DevAI Dataset Statistics | DevAI Dataset Baseline | SUFFICIENT | GROUNDED | PASS |
| **Q02** | Agent-as-a-Judge Savings | Human vs Agent-as-a-Judge | SUFFICIENT | GROUNDED | PASS |
| **Q03** | Three Agentic Frameworks | AI Developer Baselines | SUFFICIENT | GROUNDED | PASS |
| **Q04** | Average Cost & Time | Table 1 Developer Statistics | SUFFICIENT | GROUNDED | **PASS** *(Table 1 cost/time)* |
| **Q05** | Requirements Met & Solve Rate | Table 2 Baselines vs Table 3 Judges | INSUFFICIENT_OR_AMBIGUOUS | GROUNDED | **TRIGGERED** *(Table 2 absent, Table 3 blocked)* |
| **Q06** | OpenHands Black-Box Alignment | Table 3 AI Judges | INSUFFICIENT_OR_AMBIGUOUS | GROUNDED | **TRIGGERED** *(Conflated settings)* |
| **Q07** | Component Ablations | Table 4 / Section 4.3 | SUFFICIENT | GROUNDED | **PASS** *(Table 4 ablations)* |
| **Q08** | Search Module Comparison | Appendix K.2 / Table 6 | SUFFICIENT | GROUNDED | PASS |
| **Q09** | SVM and LSTM Architectures | Figure 2 Distribution | SUFFICIENT | GROUNDED | PASS |
| **Q10** | Task 51 Requirement R1 | Figure 3 Task 51 DAG | INSUFFICIENT_OR_AMBIGUOUS | GROUNDED | **TRIGGERED** *(Missing entity 'Task 51')* |
| **Q11** | Human Evaluator Errors | Section 3.2 / Appendix H | SUFFICIENT | GROUNDED | PASS |
| **Q12** | Human-as-a-Judge Limitations | Section 4 / 4.4 Constraints | SUFFICIENT | GROUNDED | PASS |

### Benchmark Deliverables
- **Evaluation Runner Script:** [`evaluate_question_bank.py`](evaluate_question_bank.py) (reproducibly executes the 12-question benchmark)

---

## Installation & Setup Guide

### 1. Prerequisites
- **Python:** 3.10 to 3.13 (Verified on Python 3.13)
- **Ollama:** Installed and running locally ([ollama.com](https://ollama.com/))
- **Hardware:** 8GB+ RAM; runs on standard CPU (GPU recommended for faster generation)

### 2. Clone Repository & Setup Virtual Environment
```bash
# Clone the repository
git clone <REPOSITORY_URL>
cd enverus-rag-assignment

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the template `.env.example` to `.env`:
```bash
# Windows (PowerShell):
Copy-Item .env.example .env
# Linux / macOS:
cp .env.example .env
```
Default configuration values:
```ini
# Vector Store
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=agent_as_a_judge

# Embeddings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Ollama LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TEMPERATURE=0.0
OLLAMA_TIMEOUT=60.0
```

### 4. Setup Local Ollama Model
```bash
# Start Ollama service (if not already running as a daemon)
ollama serve

# Pull the Qwen 2.5 1.5B model
ollama pull qwen2.5:1.5b

# Verify availability
ollama list
```

---

## Execution & Reproduction Instructions

### Step 1: Ingest PDF & Build ChromaDB Vector Store
If starting fresh or rebuilding the vector database:
```bash
python build_index.py
```
*Output: Extracts 44 pages from `data/agent_as_a_judge.pdf`, chunks into 134 records, embeds with `all-MiniLM-L6-v2`, and populates `chroma_db/`.*

### Step 2: Run Automated Tests
Execute the comprehensive 98-test pytest suite:
```bash
python -m pytest -q
```
*Expected result:* **`98 passed in ~35s`**

### Step 3: Run Interactive / Programmatic RAG Queries
Run the end-to-end RAG verification script:
```bash
python test_real_rag.py
```
Or use the Python API directly:
```python
from src.rag_pipeline import get_rag_pipeline

pipeline = get_rag_pipeline(default_top_k=5)
response = pipeline.answer("How much time and cost does Agent-as-a-Judge save compared with human experts?")

print("Answer:\n", response.answer)
print("\nSources:")
for src in response.sources:
    print(f" - [{src.rank}] {src.chunk_id} (Page {src.page_number}, {src.section}) - Score: {src.score:.4f}")
```

### Step 4: Reproduce 12-Question Benchmark Evaluation
Re-execute all 12 authoritative questions through the live pipeline:
```bash
python evaluate_question_bank.py
```
*Output: Evaluates all 12 benchmark questions against the local ChromaDB index and Ollama model.*

### Step 5: Reproduce Retrieval Evaluation Audit
Evaluate retrieval accuracy metrics across the 12 queries:
```bash
python eval_retrieval.py
```
*Output: Computes Top-1 (50.0%), Top-3 (75.0%), and Top-5 (83.3%) retrieval success rates in `retrieval_evaluation_results.json`.*

---

## Known Limitations
1. **Single-Paper Corpus:** Tailored specifically to academic paper ingestion for *Agent-as-a-Judge*. Multi-document retrieval across heterogeneous domains would benefit from hierarchical document-level routing.
2. **Dense Vector Search:** Uses dense semantic retrieval (`all-MiniLM-L6-v2`) without a sparse BM25 hybrid fusion layer. While achieving 83.3% Top-5 recall across benchmark questions, hybrid BM25 + dense ranking could further improve retrieval for rare alphanumeric tokens.
3. **Conservative Safety Guardrails:** The safety layer intentionally prioritizes factual grounding over conversational flexibility. Queries retrieving ambiguous or conflated multi-setting evidence return canonical insufficiency rather than attempting ungrounded approximations.
4. **Local CPU Inference Latency:** On standard multi-core CPUs, local Qwen 1.5B generation takes 8–15 seconds for single-paragraph responses and up to ~40 seconds for complex multi-point answers. Hardware acceleration (NVIDIA CUDA or Apple Silicon MPS) significantly reduces generation latency.

---

## Deliverables Summary
- **Source Code:** [`src/`](src/)
- **Test Suite:** [`tests/`](tests/)
- **Workflow Visualization:** [`visualization/rag_workflow.png`](visualization/rag_workflow.png) & [`visualization/README.md`](visualization/README.md)
- **Retrieval Evaluation Results:** [`retrieval_evaluation_results.json`](retrieval_evaluation_results.json)
