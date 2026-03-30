# Multimodal Research Assistant (MRA)

> **Does augmenting a vision-language model with retrieved paper text improve its ability to answer questions about scientific figures — and does this effect vary by question type?**
> 

Corpus: Attention Is All You Need · ResNet · BERT · 30 figure-question pairs across 3 question categories

Evaluation: 3-way pipeline comparison · LLM-as-judge scoring · Failure mode taxonomy

---

## Abstract

We build and evaluate a three-pipeline system for question answering over scientific figures, using the Attention, ResNet, and BERT papers as evaluation corpus. The system compares: (A) a vision-language model answering from image alone, (B) a text retrieval system answering from retrieved paper passages alone, and (C) a multimodal RAG system that fuses both visual understanding and retrieved text into a structured prompt.

We evaluate across 30 figure-question pairs stratified into three categories — factual, visual trend, and cross-verification — and score each pipeline on correctness, groundedness, and hallucination rate using a fixed LLM-as-judge protocol.

Our results show that **Pipeline C (Multimodal RAG) outperforms both baselines on all three question types**, with the largest gains on cross-verification questions (+49% over vision-only, +51% over RAG-only on correctness). Vision-only performance is consistently weakest, and RAG-only fails entirely on visual trend questions (0.000 correctness), confirming that text retrieval cannot substitute for visual understanding. These findings suggest that structured multimodal fusion is necessary for robust scientific figure Q&A, and that neither modality alone is sufficient.

---

## Results

### Correctness

| Question Type | Vision-Only (A) | RAG-Only (B) | Multimodal RAG (C) | Winner |
| --- | --- | --- | --- | --- |
| Factual | 0.182 | 0.266 | **0.336** | C |
| Visual Trend | 0.056 | 0.000 | **0.511** | C |
| Cross-Verification | 0.230 | 0.210 | **0.720** | C |

### Groundedness

| Question Type | Vision-Only (A) | RAG-Only (B) | Multimodal RAG (C) |
| --- | --- | --- | --- |
| Factual | 0.182 | 0.345 | **0.318** |
| Visual Trend | 0.111 | 0.222 | **0.622** |
| Cross-Verification | 0.000 | 0.170 | **0.780** |

### Failure Mode Summary (30 items)

| Failure Mode | Count |
| --- | --- |
| FM-1: Retrieval Noise | 1 / 30 |
| FM-2: Visual Misinterpretation | 1 / 30 |
| FM-3: Modality Conflict | 1 / 30 |

---

## What These Numbers Mean

**Pipeline C wins every question category.** The gains are not uniform — they are largest on cross-verification (+49pp over A, +51pp over B) and visual trend (+45pp over A, +51pp over B), and smallest on factual (+15pp over A, +7pp over B). This pattern is interpretable and consistent with the pre-stated hypotheses.

**RAG-only scores 0.000 on visual trend questions.** This is the clearest finding in the results. Questions requiring interpretation of curve slopes, relative ordering, or spatial visual relationships cannot be answered from paper text alone — the relevant information simply does not appear in the text corpus. RAG-only is not just worse than the multimodal system on these questions; it is completely uninformative.

**Vision-only is weakest on factual questions.** The BLIP-2 vision backbone struggles to reliably extract specific numerical values from tables and figures, scoring 0.182 on factual questions. Retrieved paper text substantially improves factual precision, with Pipeline B scoring 0.266 and Pipeline C 0.336.

**Groundedness of Pipeline C is highest on cross-verification (0.780).** This is consistent with the structured fusion prompt design: the [IMAGE UNDERSTANDING] + [RETRIEVED CONTEXT] + [INSTRUCTION] structure gives the model explicit evidential channels to cite, reducing hallucination on questions that require both modalities.

**The precision-groundedness pattern from Project 2 generalises.** Project 2 found that cross-encoder reranking improves groundedness at a cost to domain accuracy. Here, multimodal RAG improves correctness and groundedness simultaneously on cross-verification, but the factual groundedness of Pipeline C (0.318) is slightly lower than RAG-only (0.345) — suggesting that visual context occasionally introduces noise for questions fully answerable from text.

---

## System Architecture

```
User Input: PDF + Figure + Question
              |
    ┌─────────┴──────────┐
    │                    │
DOCUMENT PIPELINE   VISION PIPELINE
PyPDFLoader         ImageProcessor (PIL)
→ Chunker           → BLIP-2 / LLaVA
  (512 / 50)          caption generation
→ bge-small embed   → PaddleOCR
→ ChromaDB            text extraction
  (per-session)     → Vision-only answer
    │                    │
    └─────────┬──────────┘
              │
         FUSION LAYER
    [IMAGE UNDERSTANDING]
    [TEXT IN FIGURE (OCR)]
    [RETRIEVED PAPER CONTEXT]
    [QUESTION] + [INSTRUCTION]
    → Llama-3-8B (Groq, temp=0.1)
              │
         OUTPUT LAYER
    Pipeline A: Vision-only answer
    Pipeline B: RAG-only answer
    Pipeline C: Multimodal RAG answer  ← primary system
    Evidence:   caption + OCR + passages
```

---

## The Three Pipelines

**Pipeline A — Vision-Only (Baseline 1)**

LLaVA / BLIP-2 receives the figure and question directly. No paper text. No retrieved passages. Establishes the upper bound of pure visual reasoning.

**Pipeline B — RAG-Only (Baseline 2)**

The question is embedded and used to retrieve top-4 passages from a session-scoped ChromaDB built from the uploaded PDF. Llama-3-8B generates an answer from retrieved text only. The figure is never seen. Establishes the upper bound of text-only retrieval.

**Pipeline C — Multimodal RAG (Primary System)**

Fuses three evidence sources in a structured prompt:

1. LLaVA/BLIP-2 caption of the figure
2. PaddleOCR text extracted from the figure
3. Top-4 retrieved paper passages

The prompt uses explicit section headers ([IMAGE UNDERSTANDING], [RETRIEVED CONTEXT]) and instructs the model to cite each claim's source and surface contradictions between modalities.

---

## Evaluation Design

**Benchmark:** 30 figure-question-answer triples from three landmark ML papers (Attention Is All You Need, ResNet, BERT). 10 items per question category.

| Category | Definition |
| --- | --- |
| Factual | Answerable by reading a specific value from figure or paper |
| Visual Trend | Answerable by interpreting a visual pattern not stated in text |
| Cross-Verification | Requires comparing a visual observation against a textual claim |

**Scoring:** Fixed LLM-as-judge prompt (Llama-3-8B, temp=0.0) scoring three dimensions per answer:

- **Correctness** (0–1): factual accuracy vs reference answer
- **Groundedness** (0–1): claims supported by provided evidence
- **Hallucination-free rate** (0–1): absence of unsupported claims (1 − hallucination)

Same judge prompt applied identically to all three pipelines. Results are reproducible.

**Failure mode taxonomy:**

- FM-1 Retrieval Noise: Pipeline C underperforms A and figure not mentioned in retrieved passages
- FM-2 Visual Misinterpretation: Pipeline C underperforms B, suggesting caption error
- FM-3 Modality Conflict: Figure and paper text contain contradictory information

---

## Connection to Project 2 (RAG Pipeline)

This project extends Project 2's RAG pipeline with a single architectural change: ChromaDB is built dynamically per session from the uploaded PDF instead of a static corpus. All retrieval parameters are held constant — bge-small-en-v1.5 embeddings, ms-marco-MiniLM cross-encoder reranker, chunk size 512/50, Llama-3-8B generation at temp=0.1.

This design decision isolates the effect of adding the vision modality from confounding changes to the retrieval stack, making the comparison between Project 2 and Project 3 findings methodologically consistent.

Project 2 finding: cross-encoder reranking improves groundedness (+2.09%) at a cost to domain accuracy (-4.0%) in multi-domain corpora.

Project 3 finding: multimodal RAG improves correctness and groundedness on cross-verification questions, but shows marginal groundedness degradation on factual questions relative to RAG-only — consistent with visual context occasionally acting as noise for text-answerable questions.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Vision-Language Model | BLIP-2-flan-t5-xl (local) / LLaVA-1.6-Mistral-7B (deployed) |
| Text Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Store | ChromaDB (session-scoped) |
| Generation LLM | llama3-8b-8192 via Groq API (temp=0.1) |
| OCR | PaddleOCR (confidence ≥ 0.7) |
| PDF Processing | LangChain PyPDFLoader |
| Backend | FastAPI + uvicorn |
| Frontend | Gradio 4.x |
| Deployment | HuggingFace Spaces (Docker, T4 GPU) |

---

## Project Structure

```
multimodal-research-assistant/
├── backend/
│   ├── main.py
│   ├── core/          config.py · session_store.py
│   ├── routes/        upload.py · ask.py
│   ├── rag/           ingest.py · retriever.py · generator.py
│   ├── vision/        image_processor.py · llava_model.py · ocr.py
│   ├── fusion/        multimodal_pipeline.py
│   └── eval/          evaluator.py
├── frontend/          app.py
├── data/benchmark/    eval_set.json · results.json
├── Dockerfile
└── .env.example
```

---
## UI (WORKING DEMO SCREENSHOT)

<img width="1900" height="878" alt="image" src="https://github.com/user-attachments/assets/d7109ade-73c8-470b-b51c-aca40a9e32b4" />


## Reproducibility

```bash
# Install
conda create -n mra python=3.10 -y
conda activate mra
pip install -r requirements.txt

# Configure
cp .env.example .env
# Set GROQ_API_KEY in .env

# Run backend
uvicorn backend.main:app --port 8000

# Run frontend
python frontend/app.py

# Run evaluation
python backend/eval/evaluator.py
```

**Key hyperparameters**

| Parameter | Value |
| --- | --- |
| Embedding model | BAAI/bge-small-en-v1.5 (384-dim) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Stage-1 candidates | 20 |
| Final top-k passages | 4 |
| Chunk size | 512 characters |
| Chunk overlap | 50 characters |
| Generation LLM | llama3-8b-8192 via Groq |
| Temperature | 0.1 |
| Judge model | llama3-8b-8192, temp=0.0 |
| Eval set size | 30 items (10 per question type) |
| Vision backbone (local) | BLIP-2-flan-t5-xl |
| Vision backbone (deployed) | LLaVA-1.6-Mistral-7B |

---

## Limitations

- **Benchmark scale:** 30 items provides directional findings, not statistically significant estimates. Results are preliminary and should not be over-interpreted.
- **Single vision backbone:** BLIP-2 used locally due to GPU constraints. LLaVA-1.6 used on HuggingFace Spaces. Results may differ between backbones.
- **Judge bias:** The judge model (Llama-3-8B) is the same as the generation model in Pipelines B and C, introducing potential self-evaluation bias.
- **Figure types:** Benchmark biased toward line charts, bar charts, and architecture tables from ML papers. Findings may not generalise to other figure types.
- **Single reranker:** Only ms-marco-MiniLM evaluated. Domain-adapted rerankers not tested.

---

## Future Work

- Scale benchmark to 100+ items with confidence intervals reported
- Evaluate LLaVA-1.6 and GPT-4o as alternative vision backbones and compare against BLIP-2
- Domain-adapted reranker fine-tuned on scientific figure QA pairs
- Extend to other paper domains (biology, physics, economics)
- Full RAGAs evaluation (faithfulness, answer relevancy, context recall, context precision)
