# import time
# from PIL import Image
# from groq import Groq
# from backend.vision.llava_model import vision_only_answer, generate_figure_caption
# from backend.vision.ocr import extract_text_from_figure
# from backend.rag.retriever import retrieve          # your Project 2 retriever
# from backend.rag.generator import generate_answer  # your Project 2 generator
# from backend.core.config import settings

# client = Groq(api_key=settings.groq_api_key)


# # ---------------------------------------------------------------------------
# # PIPELINE A — Vision-Only
# # ---------------------------------------------------------------------------

# def run_pipeline_a(image: Image.Image, question: str) -> dict:
#     """
#     Pipeline A: Vision-only baseline.
#     Input:  image + question
#     Cannot see: paper text, retrieved passages
#     Purpose: establishes VLM upper bound on visual reasoning alone.
#     """
#     start = time.time()
#     answer = vision_only_answer(image, question)
#     return {
#         "pipeline": "A_vision_only",
#         "answer": answer,
#         "latency": round(time.time() - start, 2),
#         "evidence": {}
#     }


# # ---------------------------------------------------------------------------
# # PIPELINE B — RAG-Only
# # ---------------------------------------------------------------------------

# def run_pipeline_b(question: str, vectorstore) -> dict:
#     """
#     Pipeline B: Text retrieval baseline.
#     Input:  question + ChromaDB vectorstore (built from uploaded PDF)
#     Cannot see: the figure at all
#     Purpose: establishes text-retrieval upper bound on factual questions.
#     Uses identical retrieval + generation stack as Project 2.
#     """
#     start = time.time()
#     passages = retrieve(question, vectorstore, top_k=settings.rerank_top_k)
#     answer = generate_answer(question, passages)
#     return {
#         "pipeline": "B_rag_only",
#         "answer": answer,
#         "latency": round(time.time() - start, 2),
#         "evidence": {
#             "retrieved_passages": passages
#         }
#     }


# # ---------------------------------------------------------------------------
# # PIPELINE C — Multimodal RAG (Primary System)
# # ---------------------------------------------------------------------------

# FUSION_PROMPT_TEMPLATE = """\
# [IMAGE UNDERSTANDING]
# {visual_description}

# [TEXT EXTRACTED FROM FIGURE (OCR)]
# {ocr_text}

# [RETRIEVED PAPER CONTEXT]
# Passage 1: {passage_1}
# ---
# Passage 2: {passage_2}
# ---
# Passage 3: {passage_3}
# ---
# Passage 4: {passage_4}

# [QUESTION]
# {question}

# [INSTRUCTION]
# You are a research assistant analysing a scientific figure alongside its source paper.
# Answer the question using all available sources above.

# For every factual claim in your answer, cite its source:
#   [VISUAL]     - information from the image description
#   [OCR]        - text directly extracted from the figure
#   [PAPER:1]    - Passage 1 from retrieved paper context
#   [PAPER:2]    - Passage 2 from retrieved paper context
#   [PAPER:3]    - Passage 3 from retrieved paper context
#   [PAPER:4]    - Passage 4 from retrieved paper context

# If information from different sources contradicts each other, state the
# contradiction explicitly. Do not silently resolve it. Indicate which
# source you are prioritising and why.

# If none of the sources contain sufficient information to answer the
# question, state this explicitly rather than speculating.
# """


# def run_pipeline_c(image: Image.Image, question: str, vectorstore) -> dict:
#     """
#     Pipeline C: Multimodal RAG — primary system.
#     Input:  image + question + ChromaDB vectorstore
#     Can see: LLaVA caption + OCR text + top-4 retrieved paper passages

#     Fusion order:
#       1. LLaVA generates structured figure caption [IMAGE UNDERSTANDING]
#       2. PaddleOCR extracts visible text [OCR]
#       3. Question embeds → ChromaDB ANN → cross-encoder rerank → top-4 passages
#       4. Structured prompt assembled with fixed section headers
#       5. Llama-3-8B (Groq, temp=0.1) generates cited answer
#     """
#     start = time.time()

#     # Step 1: Visual understanding
#     visual_description = generate_figure_caption(image)

#     # Step 2: OCR
#     ocr_text = extract_text_from_figure(image)

#     # Step 3: Retrieval (same as Pipeline B)
#     passages = retrieve(question, vectorstore, top_k=settings.rerank_top_k)
#     # Pad to exactly 4 if fewer returned
#     while len(passages) < 4:
#         passages.append("[No passage retrieved]")

#     # Step 4: Build structured fusion prompt
#     prompt = FUSION_PROMPT_TEMPLATE.format(
#         visual_description=visual_description,
#         ocr_text=ocr_text,
#         passage_1=passages[0],
#         passage_2=passages[1],
#         passage_3=passages[2],
#         passage_4=passages[3],
#         question=question
#     )

#     # Step 5: Generate
#     response = client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1,
#         max_tokens=800
#     )
#     answer = response.choices[0].message.content.strip()

#     return {
#         "pipeline": "C_multimodal_rag",
#         "answer": answer,
#         "latency": round(time.time() - start, 2),
#         "evidence": {
#             "visual_description": visual_description,
#             "ocr_text": ocr_text,
#             "retrieved_passages": passages
#         }
#     }


# # ---------------------------------------------------------------------------
# # Orchestrator — runs all three
# # ---------------------------------------------------------------------------

# def run_all_pipelines(image: Image.Image, question: str, vectorstore) -> dict:
#     """
#     Runs Pipeline A, B, C for a single image + question.
#     Returns combined result dict for API response and evaluation.
#     """
#     result_a = run_pipeline_a(image, question)
#     result_b = run_pipeline_b(question, vectorstore)
#     result_c = run_pipeline_c(image, question, vectorstore)

#     return {
#         "question": question,
#         "pipeline_a_vision_only": result_a,
#         "pipeline_b_rag_only": result_b,
#         "pipeline_c_multimodal_rag": result_c
#     }


#GPT!#
import time
from PIL import Image
from groq import Groq

from backend.vision.llava_model import vision_only_answer, generate_figure_caption
from backend.vision.ocr import extract_text_from_figure

# ✅ FIXED IMPORT
from backend.rag.retriever import RAGRetriever
from backend.rag.generator import generate_answer

from backend.core.config import settings


# Initialize clients/models ONCE
client = Groq(api_key=settings.groq_api_key)

# ✅ Create retriever instance
# retriever = RAGRetriever(use_reranker=True)


# ---------------------------------------------------------------------------
# PIPELINE A — Vision-Only
# ---------------------------------------------------------------------------

def run_pipeline_a(image: Image.Image, question: str) -> dict:
    start = time.time()

    answer = vision_only_answer(image, question)

    return {
        "pipeline": "A_vision_only",
        "answer": answer,
        "latency": round(time.time() - start, 2),
        "evidence": {}
    }


# ---------------------------------------------------------------------------
# PIPELINE B — RAG-Only
# ---------------------------------------------------------------------------

# def run_pipeline_b(question: str) -> dict:
#     start = time.time()

#     # ✅ FIXED CALL
#     passages = retriever.retrieve(
#         query=question,
#         top_k=settings.rerank_top_k
#     )

#     answer = generate_answer(question, passages)

#     return {
#         "pipeline": "B_rag_only",
#         "answer": answer,
#         "latency": round(time.time() - start, 2),
#         "evidence": {
#             "retrieved_passages": passages
#         }
#     }


# ---------------------------------------------------------------------------
# PIPELINE C — Multimodal RAG
# ---------------------------------------------------------------------------

FUSION_PROMPT_TEMPLATE = """\
[IMAGE UNDERSTANDING]
{visual_description}

[TEXT EXTRACTED FROM FIGURE (OCR)]
{ocr_text}

[RETRIEVED PAPER CONTEXT]
Passage 1: {passage_1}
---
Passage 2: {passage_2}
---
Passage 3: {passage_3}
---
Passage 4: {passage_4}

[QUESTION]
{question}

[INSTRUCTION]
You are a research assistant analysing a scientific figure alongside its source paper.
Answer the question using all available sources above.

For every factual claim in your answer, cite its source:
  [VISUAL]
  [OCR]
  [PAPER:1]
  [PAPER:2]
  [PAPER:3]
  [PAPER:4]

If sources contradict each other, state it explicitly.
If insufficient information, say so instead of guessing.
"""


# def run_pipeline_c(image: Image.Image, question: str) -> dict:
#     start = time.time()

#     # Step 1: Vision
#     visual_description = generate_figure_caption(image)

#     # Step 2: OCR
#     ocr_text = extract_text_from_figure(image)

#     # Step 3: Retrieval
#     passages = retriever.retrieve(
#         query=question,
#         top_k=settings.rerank_top_k
#     )

#     # Ensure exactly 4 passages
#     while len(passages) < 4:
#         passages.append("[No passage retrieved]")

#     # Convert Document → text if needed
#     passages_text = [
#         p.page_content if hasattr(p, "page_content") else str(p)
#         for p in passages
#     ]

#     # Step 4: Prompt
#     prompt = FUSION_PROMPT_TEMPLATE.format(
#         visual_description=visual_description,
#         ocr_text=ocr_text,
#         passage_1=passages_text[0],
#         passage_2=passages_text[1],
#         passage_3=passages_text[2],
#         passage_4=passages_text[3],
#         question=question
#     )

#     # Step 5: Generate (Groq)
#     response = client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1,
#         max_tokens=800
#     )

#     answer = response.choices[0].message.content.strip()

#     return {
#         "pipeline": "C_multimodal_rag",
#         "answer": answer,
#         "latency": round(time.time() - start, 2),
#         "evidence": {
#             "visual_description": visual_description,
#             "ocr_text": ocr_text,
#             "retrieved_passages": passages_text
#         }
#     }


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

# def run_all_pipelines(image: Image.Image, question: str, vectorstore=None) -> dict:
#     """
#     vectorstore arg kept for compatibility (ignored now)
#     """

#     result_a = run_pipeline_a(image, question)
#     result_b = run_pipeline_b(question)
#     result_c = run_pipeline_c(image, question)

#     return {
#         "question": question,
#         "pipeline_a_vision_only": result_a,
#         "pipeline_b_rag_only": result_b,
#         "pipeline_c_multimodal_rag": result_c
#     }

def run_pipeline_b(question: str, vectorstore) -> dict:
    start = time.time()

    retriever = RAGRetriever(vectorstore=vectorstore, use_reranker=True)

    passages = retriever.retrieve(
        query=question,
        top_k=settings.rerank_top_k
    )

    result = generate_answer(question, passages)
    answer = result["answer"]

    return {
        "pipeline": "B_rag_only",
        "answer": answer,
        "latency": round(time.time() - start, 2),
        "evidence": {
            "retrieved_passages": passages
        }
    }

def run_pipeline_c(image: Image.Image, question: str, vectorstore) -> dict:
    start = time.time()

    retriever = RAGRetriever(vectorstore=vectorstore, use_reranker=True)

    visual_description = generate_figure_caption(image)
    ocr_text = extract_text_from_figure(image)

    passages = retriever.retrieve(
        query=question,
        top_k=settings.rerank_top_k
    )

    while len(passages) < 4:
        passages.append("[No passage retrieved]")

    passages_text = [
        p.page_content if hasattr(p, "page_content") else str(p)
        for p in passages
    ]

    prompt = FUSION_PROMPT_TEMPLATE.format(
        visual_description=visual_description,
        ocr_text=ocr_text,
        passage_1=passages_text[0],
        passage_2=passages_text[1],
        passage_3=passages_text[2],
        passage_4=passages_text[3],
        question=question
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800
    )

    answer = response.choices[0].message.content.strip()

    return {
        "pipeline": "C_multimodal_rag",
        "answer": answer,
        "latency": round(time.time() - start, 2),
        "evidence": {
            "visual_description": visual_description,
            "ocr_text": ocr_text,
            "retrieved_passages": passages_text
        }
    }

def run_all_pipelines(image: Image.Image, question: str, vectorstore) -> dict:
    return {
        "question": question,
        "pipeline_a_vision_only": run_pipeline_a(image, question),
        "pipeline_b_rag_only": run_pipeline_b(question, vectorstore),
        "pipeline_c_multimodal_rag": run_pipeline_c(image, question, vectorstore)
    }