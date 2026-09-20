# Phase 6.1 — Retrieval Evaluation Audit Report

## 1. Overview & Evaluation Methodology

This audit establishes a **strict evidence-grounding standard** for evaluating the baseline dense semantic retriever (`src/retriever.py`, embedding model `all-MiniLM-L6-v2`, 134 chunks in ChromaDB collection `agent_as_a_judge`).

### The Strict Relevance Criterion
Under this standard:
> **A retrieval is classified as a SUCCESS only if the retrieved chunk contains sufficient, explicit evidence to answer the specific query directly.**  
> Broad topical overlap, adjacent chapter outlines, or ambiguous/conflated prose without the specific requested fact do **NOT** count as success.

---

## 2. Explicit Relevance Criteria by Query

| # | Query | Explicit Relevance Criterion | Designated Ground-Truth Chunks |
| :- | :--- | :--- | :--- |
| **1** | DevAI dataset statistics | Must state core quantitative dataset statistics for DevAI (e.g., 55 tasks, 365 requirements, 125 preferences, distribution of query lengths, or lines of code). | `chunk_p04_002`, `chunk_p04_001` |
| **2** | Agent-as-a-Judge evaluation time and cost savings | Must explicitly state quantitative savings in time and/or cost achieved by Agent-as-a-Judge compared to human evaluation (e.g., 97.72% time savings, 97.64% cost savings, or 86.5h / $1,297.5 vs 1.97h / $30.6). | `chunk_p11_002`, `chunk_p03_001` |
| **3** | Three open-source agentic frameworks | Must explicitly name all three open-source code agent frameworks benchmarked in the study (MetaGPT, GPT-Pilot, OpenHands). | `chunk_p02_006`, `chunk_p05_003` |
| **4** | Average cost and time of MetaGPT, GPT-Pilot and OpenHands | Must state the quantitative average cost and execution time figures for MetaGPT, GPT-Pilot, and OpenHands (e.g., Table 1 values: $1.19 / 775.29s, $3.92 / 1622.38s, $6.38 / 362.41s). | `chunk_p06_001`, `chunk_p06_002` |
| **5** | Requirements Met and Task Solve Rate | Must provide benchmark performance figures for both 'Requirements Met' and 'Task Solve Rate' across frameworks (Table 2 baseline metrics: MetaGPT 5.67% / 0.0%, GPT-Pilot 29.25% / 1.82%, OpenHands 29.55% / 0.0%). | `chunk_p07_002` |
| **6** | OpenHands black-box alignment rate | Must explicitly and unambiguously state the black-box alignment rate specifically for OpenHands (90.44% for Agent-as-a-Judge vs 60.38% for LLM-as-a-Judge), clearly distinguishing it from gray-box rates. Conflated/ambiguous prose does not qualify. | `chunk_p09_002`, `chunk_p09_001` |
| **7** | Agent-as-a-Judge component ablation | Must present the component ablation setup or results analyzing the sequential impact of Agent-as-a-Judge modules (e.g., modular combination of (1), (2), (3), (5), (6), or Table 4 / Table 5 component ablations). | `chunk_p09_004`, `chunk_p39_001`, `chunk_p10_004` |
| **8** | Search module comparison | Must present the comparative empirical evaluation of search algorithms used within the agent's search module (Table 6 comparison of BM25, grep, embeddings in Appendix K.2). | `chunk_p39_004` |
| **9** | SVM and LSTM architectures in DevAI | Must provide evidence covering BOTH requested architecture terms (SVM and LSTM) within DevAI tasks or architecture distributions. A chunk with only one does not qualify. | `chunk_p04_001` |
| **10** | Task 51 Requirement R1 | Must provide the specific content/specification of Requirement 1 (R1) for Task 51 (Figure 3 diagram: Requirement 1 Audio Loading). | `chunk_p05_001` |
| **11** | Human evaluator errors | Must document or analyze errors, disagreements, or inconsistencies committed by human evaluators during assessment (Section 3.2 or Appendix H). | `chunk_p07_003`, `chunk_p30_001`, `chunk_p08_001`, `chunk_p30_002` |
| **12** | Human-as-a-Judge limitations | Must explicitly detail fundamental limitations of human evaluators (e.g., high labor cost, 86.5 hours total, $1,297.5 expense, fatigue, or substantial expertise requirement). | `chunk_p08_003`, `chunk_p11_002` |

---

## 3. Detailed Query-by-Query Evaluation

### Query 1: "DevAI dataset statistics"
- **Criterion:** Core quantitative dataset statistics for DevAI.
- **Top 5 Results:**
  - Rank 1: `chunk_p04_002` (Score: 0.5254, Page 4, Section: `2.2 The DevAI Dataset`) — **Ground Truth** (55 tasks, 365 requirements, 125 preferences).
  - Rank 2: `chunk_p04_001` (Score: 0.5080, Page 4, Section: `2.1 Motivation`) — **Ground Truth** (Distribution of user queries, lines of code, requirements).
  - Rank 3: `chunk_p03_002` (Score: 0.4958, Page 3, Section: `2 DevAI: A Dataset for Automated AI Development`)
  - Rank 4: `chunk_p04_003` (Score: 0.4890, Page 4, Section: `2.2 The DevAI Dataset`)
  - Rank 5: `chunk_p04_005` (Score: 0.4763, Page 4, Section: `2.3 Preliminary Benchmark`)
- **Audit Verdict:** Top-1: **SUCCESS** | Top-3: **SUCCESS** | Top-5: **SUCCESS**

---

### Query 2: "Agent-as-a-Judge evaluation time and cost savings"
- **Criterion:** Quantitative evaluation time and cost savings vs. human baseline.
- **Top 5 Results:**
  - Rank 1: `chunk_p11_002` (Score: 0.7947, Page 11, Section: `4.4 Cost Analysis`) — **Ground Truth** (86.5h vs 1.97h, $1297.5 vs $30.6, saves 97.72% time and 97.64% cost).
  - Rank 2: `chunk_p08_003` (Score: 0.7709, Page 8, Section: `4 Agent-as-a-Judge`)
  - Rank 3: `chunk_p01_002` (Score: 0.7255, Page 1, Section: `[None]`)
  - Rank 4: `chunk_p10_003` (Score: 0.7063, Page 10, Section: `4.2 Judging Agent-as-a-Judge`)
  - Rank 5: `chunk_p10_002` (Score: 0.6577, Page 10, Section: `4.2 Judging Agent-as-a-Judge`)
- **Audit Verdict:** Top-1: **SUCCESS** | Top-3: **SUCCESS** | Top-5: **SUCCESS**

---

### Query 3: "Three open-source agentic frameworks"
- **Criterion:** Explicitly name all three frameworks: MetaGPT, GPT-Pilot, OpenHands.
- **Top 5 Results:**
  - Rank 1: `chunk_p18_001` (Score: 0.5993, Page 18, Section: `Appendix A: Outline`) — Outline only; does not name frameworks.
  - Rank 2: `chunk_p03_001` (Score: 0.5888, Page 3, Section: `1 Introduction`) — Cost summary; does not name frameworks.
  - Rank 3: `chunk_p01_003` (Score: 0.5876, Page 1, Section: `1 Introduction`) — Conceptual introduction; does not name frameworks.
  - Rank 4: `chunk_p02_006` (Score: 0.5823, Page 2, Section: `1 Introduction`) — **Ground Truth** (Explicitly introduces MetaGPT, GPT-Pilot, OpenHands).
  - Rank 5: `chunk_p05_003` (Score: 0.5652, Page 5, Section: `2.3 Preliminary Benchmark`) — **Ground Truth** (Benchmarked frameworks with formal citations).
- **Audit Verdict:** Top-1: **FAILURE** | Top-3: **FAILURE** | Top-5: **SUCCESS** (Appears at Rank 4)

---

### Query 4: "Average cost and time of MetaGPT, GPT-Pilot and OpenHands"
- **Criterion:** Quantitative cost and time figures for each framework.
- **Top 5 Results:**
  - Rank 1: `chunk_p06_002` (Score: 0.6225, Page 6, Section: `2.3 Preliminary Benchmark`) — **Ground Truth** (MetaGPT cost-effectiveness vs. GPT-Pilot $3.92 / 1622.38s and OpenHands $6.38 / 362.41s).
  - Rank 2: `chunk_p06_001` (Score: 0.5519, Page 6, Section: `2.3 Preliminary Benchmark`) — **Ground Truth** (Table 1: MetaGPT $1.19 / 775.29s, GPT-Pilot $3.92 / 1622.38s, OpenHands $6.38 / 362.41s).
  - Rank 3: `chunk_p07_002` (Score: 0.4721, Page 7, Section: `3.1 Benchmark Baselines`)
  - Rank 4: `chunk_p05_003` (Score: 0.4341, Page 5, Section: `2.3 Preliminary Benchmark`)
  - Rank 5: `chunk_p27_002` (Score: 0.4277, Page 27, Section: `Appendix F`)
- **Audit Verdict:** Top-1: **SUCCESS** | Top-3: **SUCCESS** | Top-5: **SUCCESS**

---

### Query 5: "Requirements Met and Task Solve Rate"
- **Criterion:** Baseline performance values for both metrics across evaluated frameworks.
- **Top 5 Results:**
  - Rank 1: `chunk_p04_004` (Score: 0.6028, Page 4, Section: `2.2 The DevAI Dataset`) — Structural description of DAG; no performance numbers.
  - Rank 2: `chunk_p09_002` (Score: 0.5398, Page 9, Section: `4.1 Proof-of-Concept`) — Table 3 (continued): Alignment & Shift metrics.
  - Rank 3: `chunk_p07_002` (Score: 0.5320, Page 7, Section: `3.1 Benchmark Baselines`) — **Ground Truth** (Table 2: Requirements Met: 5.67%, 29.25%, 29.55%; Task Solve Rate: 0.0%, 1.82%, 0.0%).
  - Rank 4: `chunk_p23_002` (Score: 0.5058, Page 23, Section: `Appendix E`)
  - Rank 5: `chunk_p05_002` (Score: 0.4886, Page 5, Section: `2.3 Preliminary Benchmark`)
- **Audit Verdict:** Top-1: **FAILURE** | Top-3: **SUCCESS** (Rank 3) | Top-5: **SUCCESS**

---

### Query 6: "OpenHands black-box alignment rate" — *RE-AUDITED*
- **Criterion:** Must unambiguously state the specific black-box alignment rate for OpenHands (90.44% for Agent-as-a-Judge vs. 60.38% for LLM-as-a-Judge), distinct from gray-box.
- **Top 5 Results:**
  - Rank 1: `chunk_p10_003` (Score: 0.4006, Page 10, Section: `4.2 Judging Agent-as-a-Judge and LLM-as-a-Judge`) — Conflated text: *"Agent-as-a-Judge reaches 92.07% and 90.44%, surpassing LLM-as-a-Judge's 70.76% and 60.38% in both gray-box and black-box settings."* It mixes MetaGPT's gray-box rate (92.07%) and OpenHands' black-box rate (90.44%) without explicit 1-to-1 attribute mapping.
  - Rank 2: `chunk_p21_003` (Score: 0.3580, Page 21, Section: `Appendix D`) — Related work discussion.
  - Rank 3: `chunk_p10_002` (Score: 0.3484, Page 10, Section: `4.2 Judging Agent-as-a-Judge and LLM-as-a-Judge`) — Definition of Judge Shift.
  - Rank 4: `chunk_p10_004` (Score: 0.3333, Page 10, Section: `4.3 Ablations For Agent-as-a-Judge`) — Ablation intro.
  - Rank 5: `chunk_p06_002` (Score: 0.3240, Page 6, Section: `2.3 Preliminary Benchmark`) — Cost stats.
  - *(Ground-Truth Table 3 appears at Rank 6 [`chunk_p09_001`] and Rank 7 [`chunk_p09_002`]).*
- **Audit Verdict:** Top-1: **FAILURE** | Top-3: **FAILURE** | Top-5: **FAILURE**  
  *Rationale:* `chunk_p10_003` does not provide an explicitly identifiable black-box value. The only explicit breakdown exists in Table 3 (`chunk_p09_002`), which was placed at Rank 7.

---

### Query 7: "Agent-as-a-Judge component ablation"
- **Criterion:** Component ablation methodology or sequential module ablation data.
- **Top 5 Results:**
  - Rank 1: `chunk_p09_004` (Score: 0.6617, Page 9, Section: `4.1 Proof-of-Concept`) — **Ground Truth** (Optimal modular combination (1), (2), (3), (5), and (6)).
  - Rank 2: `chunk_p39_001` (Score: 0.4713, Page 39, Section: `Appendix K: Ablations of Agent-as-a-Judge`) — **Ground Truth** (Table 5 Component Ablation Studies).
  - Rank 3: `chunk_p10_004` (Score: 0.4627, Page 10, Section: `4.3 Ablations For Agent-as-a-Judge`) — **Ground Truth** (Section 4.3 component ablation analysis).
  - Rank 4: `chunk_p08_003` (Score: 0.4169, Page 8, Section: `4 Agent-as-a-Judge`)
  - Rank 5: `chunk_p11_005` (Score: 0.3667, Page 11, Section: `6 Discussion and Conclusion`)
- **Audit Verdict:** Top-1: **SUCCESS** | Top-3: **SUCCESS** | Top-5: **SUCCESS**

---

### Query 8: "Search module comparison"
- **Criterion:** Empirical comparison of search algorithms used in the search module.
- **Top 5 Results:**
  - Rank 1: `chunk_p39_004` (Score: 0.6036, Page 39, Section: `Appendix K: Ablations of Agent-as-a-Judge`) — **Ground Truth** (Section K.2 and Table 6: Comparison of Search Algorithms).
  - Rank 2: `chunk_p40_001` (Score: 0.4849, Page 40, Section: `Appendix K`)
  - Rank 3: `chunk_p39_002` (Score: 0.4645, Page 39, Section: `Appendix K`)
  - Rank 4: `chunk_p10_001` (Score: 0.4271, Page 10, Section: `4.1 Proof-of-Concept`)
  - Rank 5: `chunk_p08_004` (Score: 0.4246, Page 8, Section: `4.1 Proof-of-Concept`)
- **Audit Verdict:** Top-1: **SUCCESS** | Top-3: **SUCCESS** | Top-5: **SUCCESS**

---

### Query 9: "SVM and LSTM architectures in DevAI" — *RE-AUDITED*
- **Criterion:** Must provide evidence covering BOTH requested architecture terms (SVM and LSTM) within DevAI.
- **Top 5 Results:**
  - Rank 1: `chunk_p24_002` (Score: 0.3996, Page 24, Section: `Appendix E`) — Covers CNN-LSTM in Task 25; **contains NO mention of SVM**.
  - Rank 2: `chunk_p04_003` (Score: 0.3891, Page 4, Section: `2.2 The DevAI Dataset`) — General domain techniques; no specific SVM or LSTM details.
  - Rank 3: `chunk_p04_001` (Score: 0.3822, Page 4, Section: `2.1 Motivation`) — **Ground Truth** (Figure 2 distribution explicitly includes BOTH SVM and LSTM architectures).
  - Rank 4: `chunk_p25_001` (Score: 0.3544, Page 25, Section: `Appendix E`)
  - Rank 5: `chunk_p04_002` (Score: 0.3473, Page 4, Section: `2.2 The DevAI Dataset`)
- **Audit Verdict:** Top-1: **FAILURE** | Top-3: **SUCCESS** (Rank 3) | Top-5: **SUCCESS**  
  *Rationale:* Rank 1 satisfies only one of the two requested architectures (LSTM). Only `chunk_p04_001` at Rank 3 contains evidence for BOTH SVM and LSTM architectures.

---

### Query 10: "Task 51 Requirement R1"
- **Criterion:** Exact specification of Requirement 1 for Task 51 (from Figure 3 diagram).
- **Top 5 Results:**
  - Rank 1: `chunk_p04_004` (Score: 0.4556, Page 4, Section: `2.2 The DevAI Dataset`) — Generic DAG structure; wrong task.
  - Rank 2: `chunk_p05_002` (Score: 0.4195, Page 5, Section: `2.3 Preliminary Benchmark`) — Figure 3 caption; mentions Devin demo, lacks R1 text.
  - Rank 3: `chunk_p23_002` (Score: 0.3921, Page 23, Section: `Appendix E`)
  - Rank 4: `chunk_p23_003` (Score: 0.3776, Page 23, Section: `Appendix E`)
  - Rank 5: `chunk_p04_002` (Score: 0.3768, Page 4, Section: `2.2 The DevAI Dataset`)
  - *(Ground-Truth Figure 3 diagram chunk `chunk_p05_001` is at **Rank 8**, Score: 0.3409).*
- **Audit Verdict:** Top-1: **FAILURE** | Top-3: **FAILURE** | Top-5: **FAILURE**  
  *Rationale:* Clear demonstration of pure dense retrieval failing to match exact alphanumeric entity identifiers ("Task 51", "R1"). General methodological passages score higher than the exact diagram chunk.

---

### Query 11: "Human evaluator errors"
- **Criterion:** Documentation/analysis of human evaluator errors, disagreements, or inconsistencies.
- **Top 5 Results:**
  - Rank 1: `chunk_p07_003` (Score: 0.6686, Page 7, Section: `3.2 Judging Human-as-a-Judge`) — **Ground Truth** (Observed disagreement and errors among human evaluators in Figure 4).
  - Rank 2: `chunk_p30_001` (Score: 0.6541, Page 30, Section: `Appendix H`) — **Ground Truth** (Human evaluation procedure and error rectification).
  - Rank 3: `chunk_p07_001` (Score: 0.6146, Page 7, Section: `3.1 Benchmark Baselines`)
  - Rank 4: `chunk_p08_001` (Score: 0.6054, Page 8, Section: `3.2 Judging Human-as-a-Judge`) — **Ground Truth** (Error acknowledgment after consensus).
  - Rank 5: `chunk_p30_002` (Score: 0.5981, Page 30, Section: `Appendix H`) — **Ground Truth** (Second round correction of human errors).
- **Audit Verdict:** Top-1: **SUCCESS** | Top-3: **SUCCESS** | Top-5: **SUCCESS**

---

### Query 12: "Human-as-a-Judge limitations"
- **Criterion:** Fundamental limitations of human evaluators (cost, time, fatigue, subjectivity).
- **Top 5 Results:**
  - Rank 1: `chunk_p19_002` (Score: 0.5574, Page 19, Section: `Appendix B`) — Experimental participant table; does not detail limitations.
  - Rank 2: `chunk_p08_003` (Score: 0.5124, Page 8, Section: `4 Agent-as-a-Judge`) — **Ground Truth** (*"Human evaluation, while somewhat reliable, is time-consuming and requires substantial expertise..."*).
  - Rank 3: `chunk_p18_002` (Score: 0.5107, Page 18, Section: `Appendix A`)
  - Rank 4: `chunk_p11_002` (Score: 0.5055, Page 11, Section: `4.4 Cost Analysis`) — **Ground Truth** (86.5 hours, $1,297.5 financial burden).
  - Rank 5: `chunk_p02_007` (Score: 0.4967, Page 2, Section: `1 Introduction`)
- **Audit Verdict:** Top-1: **FAILURE** | Top-3: **SUCCESS** (Rank 2) | Top-5: **SUCCESS**

---

## 4. Metric Comparison: Phase 6 vs. Phase 6.1 Audit

| Metric | Phase 6 (Initial) | Phase 6.1 (Strict Audit) | Delta | Classification Changes |
| :--- | :---: | :---: | :---: | :--- |
| **Top-1 Success Rate** | **66.7%** (8/12) | **50.0%** (6/12) | **-16.7%** | **Q6** changed from Success to Failure (ambiguous prose).<br>**Q9** changed from Success to Failure (Rank 1 lacked SVM). |
| **Top-3 Success Rate** | **83.3%** (10/12) | **75.0%** (9/12) | **-8.3%** | **Q6** changed from Success to Failure (ground truth Table 3 is at Rank 7). |
| **Top-5 Success Rate** | **91.7%** (11/12) | **83.3%** (10/12) | **-8.4%** | **Q6** changed from Success to Failure (neither Table 3 chunk is in Top 5). |

### Summary of Changed Query Classifications

1. **Query 6 ("OpenHands black-box alignment rate"):**
   - *Previous Label:* Top-1: YES, Top-3: YES, Top-5: YES
   - *Revised Label:* Top-1: **NO**, Top-3: **NO**, Top-5: **NO**
   - *Exact Reason:* In `chunk_p10_003`, the sentence *"Agent-as-a-Judge reaches 92.07% and 90.44% ... in both gray-box and black-box settings"* fails to attribute 90.44% specifically to black-box, and conflates MetaGPT's gray-box figure (92.07%). The unambiguous ground truth is Table 3 (`chunk_p09_002`), which retrieved at Rank 7.
2. **Query 9 ("SVM and LSTM architectures in DevAI"):**
   - *Previous Label:* Top-1: YES, Top-3: YES, Top-5: YES
   - *Revised Label:* Top-1: **NO**, Top-3: **YES**, Top-5: **YES**
   - *Exact Reason:* Rank 1 (`chunk_p24_002`) only describes a CNN-LSTM task and completely omits SVM. Only `chunk_p04_001` at Rank 3 satisfies both terms by detailing the distributions of both SVM and LSTM architectures in DevAI.

---

## 5. Architectural Diagnosis for Downstream Phases

1. **Alphanumeric Entity Blindspot:**  
   Pure dense retrieval using `all-MiniLM-L6-v2` excels at semantic concepts (ablation, evaluation time/cost, human errors), but degrades on specific alphanumeric entity queries like `"Task 51 Requirement R1"` (placed at Rank 8).
2. **Tabular vs. Narrative Bias:**  
   Dense vector models embed natural language sentences with higher affinity than markdown tables. In Query 6, prose mentioning alignment ranked at Rank 1, while the structured table containing the definitive data rows landed at Rank 7.
3. **Future Retrieval Enhancements (Phase 7+):**  
   These findings provide concrete, empirical justification for exploring sparse keyword matching (BM25) or a cross-encoder reranker in later pipeline stages without preemptively bloating the current baseline.
