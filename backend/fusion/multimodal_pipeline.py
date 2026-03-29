# import time
# from PIL import Image
# from groq import Groq

# from backend.vision.llava_model import vision_only_answer, generate_figure_caption
# from backend.vision.ocr import extract_text_from_figure

# # ✅ FIXED IMPORT
# from backend.rag.retriever import RAGRetriever
# from backend.rag.generator import generate_answer

# from backend.core.config import settings


# # Initialize clients/models ONCE
# client = Groq(api_key=settings.groq_api_key)

# # ✅ Create retriever instance
# # retriever = RAGRetriever(use_reranker=True)


# # ---------------------------------------------------------------------------
# # PIPELINE A — Vision-Only
# # ---------------------------------------------------------------------------

# def run_pipeline_a(image: Image.Image, question: str) -> dict:
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

# # def run_pipeline_b(question: str) -> dict:
# #     start = time.time()

# #     # ✅ FIXED CALL
# #     passages = retriever.retrieve(
# #         query=question,
# #         top_k=settings.rerank_top_k
# #     )

# #     answer = generate_answer(question, passages)

# #     return {
# #         "pipeline": "B_rag_only",
# #         "answer": answer,
# #         "latency": round(time.time() - start, 2),
# #         "evidence": {
# #             "retrieved_passages": passages
# #         }
# #     }


# # ---------------------------------------------------------------------------
# # PIPELINE C — Multimodal RAG
# # ---------------------------------------------------------------------------

# # FUSION_PROMPT_TEMPLATE = """\
# # [IMAGE UNDERSTANDING]
# # {visual_description}

# # [TEXT EXTRACTED FROM FIGURE (OCR)]
# # {ocr_text}

# # [RETRIEVED PAPER CONTEXT]
# # Passage 1: {passage_1}
# # ---
# # Passage 2: {passage_2}
# # ---
# # Passage 3: {passage_3}
# # ---
# # Passage 4: {passage_4}

# # [QUESTION]
# # {question}

# # [INSTRUCTION]
# # You are a research assistant analysing a scientific figure alongside its source paper.
# # Answer the question using all available sources above.

# # For every factual claim in your answer, cite its source:
# #   [VISUAL]
# #   [OCR]
# #   [PAPER:1]
# #   [PAPER:2]
# #   [PAPER:3]
# #   [PAPER:4]

# # If sources contradict each other, state it explicitly.
# # If insufficient information, say so instead of guessing.
# # """
# FUSION_PROMPT_TEMPLATE = """\
# You are an expert research assistant performing multimodal reasoning.

# You are given:
# - Visual description of a figure
# - OCR text extracted from the figure
# - Retrieved passages from the paper

# ----------------------------------------
# [VISUAL DESCRIPTION]
# {visual_description}

# ----------------------------------------
# [OCR TEXT]
# {ocr_text}

# ----------------------------------------
# [RETRIEVED CONTEXT]
# [PAPER 1] {passage_1}
# [PAPER 2] {passage_2}
# [PAPER 3] {passage_3}
# [PAPER 4] {passage_4}

# ----------------------------------------
# [QUESTION]
# {question}

# ----------------------------------------
# [INSTRUCTION]

# Internally reason using all sources (visual, OCR, paper).
# DO NOT show your reasoning steps.

# ----------------------------------------
# [OUTPUT FORMAT]

# Final Answer:
# <clear, concise explanation in 5-8 sentences>

# Evidence:
# - [VISUAL]&#58; ...
# - [OCR]&#58; ...
# - [PAPER:1]&#58; ...
# """

# def run_pipeline_b(question: str, vectorstore) -> dict:
#     start = time.time()

#     retriever = RAGRetriever(vectorstore=vectorstore, use_reranker=True)

#     passages = retriever.retrieve(
#         query=question,
#         top_k=settings.rerank_top_k
#     )

#     result = generate_answer(question, passages)
#     answer = result["answer"]

#     return {
#         "pipeline": "B_rag_only",
#         "answer": answer,
#         "latency": round(time.time() - start, 2),
#         "evidence": {
#             "retrieved_passages": passages
#         }
#     }

# def run_pipeline_c(image: Image.Image, question: str, vectorstore) -> dict:
#     start = time.time()

#     retriever = RAGRetriever(vectorstore=vectorstore, use_reranker=True)

#     visual_description = generate_figure_caption(image)
#     ocr_text = extract_text_from_figure(image)

#     passages = retriever.retrieve(
#         query=question,
#         top_k=settings.rerank_top_k
#     )

#     while len(passages) < 4:
#         passages.append("[No passage retrieved]")

#     passages_text = [
#         p.page_content if hasattr(p, "page_content") else str(p)
#         for p in passages
#     ]

#     prompt = FUSION_PROMPT_TEMPLATE.format(
#         visual_description=visual_description,
#         ocr_text=ocr_text,
#         passage_1=passages_text[0],
#         passage_2=passages_text[1],
#         passage_3=passages_text[2],
#         passage_4=passages_text[3],
#         question=question
#     )

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1,
#         max_tokens=300
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

# def run_all_pipelines(image: Image.Image, question: str, vectorstore) -> dict:
#     return {
#         "question": question,
#         "pipeline_a_vision_only": run_pipeline_a(image, question),
#         "pipeline_b_rag_only": run_pipeline_b(question, vectorstore),
#         "pipeline_c_multimodal_rag": run_pipeline_c(image, question, vectorstore)
#     }

import time
from PIL import Image
from groq import Groq

from backend.vision.llava_model import vision_only_answer, generate_figure_caption
from backend.vision.ocr import extract_text_from_figure

# ✅ USE WRAPPERS (IMPORTANT)
from backend.rag.retriever import retrieve
from backend.rag.generator import generate_rag_answer

from backend.core.config import settings

client = Groq(api_key=settings.groq_api_key)


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
# FUSION PROMPT
# ---------------------------------------------------------------------------

FUSION_PROMPT_TEMPLATE = """\
You are an expert research assistant performing multimodal reasoning.

You are given:
- Visual description of a figure
- OCR text extracted from the figure
- Retrieved passages from the paper

----------------------------------------
[VISUAL DESCRIPTION]
{visual_description}

----------------------------------------
[OCR TEXT]
{ocr_text}

----------------------------------------
[RETRIEVED CONTEXT]
[PAPER 1] {passage_1}
[PAPER 2] {passage_2}
[PAPER 3] {passage_3}
[PAPER 4] {passage_4}

----------------------------------------
[QUESTION]
{question}

----------------------------------------
[INSTRUCTION]

Internally reason using all sources (visual, OCR, paper).
DO NOT show reasoning steps.

----------------------------------------
[OUTPUT FORMAT]

Final Answer:
<clear explanation in 5-8 sentences>

Evidence:
- [VISUAL]&#58; ...
- [OCR]&#58; ...
- [PAPER:1]&#58; ...
"""


# ---------------------------------------------------------------------------
# PIPELINE B — RAG-Only
# ---------------------------------------------------------------------------

def run_pipeline_b(question: str, vectorstore) -> dict:
    start = time.time()

    passages = retrieve(question, vectorstore, settings.rerank_top_k)
    answer = generate_rag_answer(question, passages)

    return {
        "pipeline": "B_rag_only",
        "answer": answer,
        "latency": round(time.time() - start, 2),
        "evidence": {
            "retrieved_passages": passages
        }
    }


# ---------------------------------------------------------------------------
# PIPELINE C — Multimodal RAG
# ---------------------------------------------------------------------------

def run_pipeline_c(image: Image.Image, question: str, vectorstore) -> dict:
    start = time.time()

    visual_description = generate_figure_caption(image)
    ocr_text = extract_text_from_figure(image)

    passages = retrieve(question, vectorstore, settings.rerank_top_k)

    while len(passages) < 4:
        passages.append("[No passage retrieved]")

    prompt = FUSION_PROMPT_TEMPLATE.format(
        visual_description=visual_description,
        ocr_text=ocr_text,
        passage_1=passages[0],
        passage_2=passages[1],
        passage_3=passages[2],
        passage_4=passages[3],
        question=question
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )

    answer = response.choices[0].message.content.strip()

    return {
        "pipeline": "C_multimodal_rag",
        "answer": answer,
        "latency": round(time.time() - start, 2),
        "evidence": {
            "visual_description": visual_description,
            "ocr_text": ocr_text,
            "retrieved_passages": passages
        }
    }


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_all_pipelines(image: Image.Image, question: str, vectorstore) -> dict:
    return {
        "question": question,
        "pipeline_a_vision_only": run_pipeline_a(image, question),
        "pipeline_b_rag_only": run_pipeline_b(question, vectorstore),
        "pipeline_c_multimodal_rag": run_pipeline_c(image, question, vectorstore)
    }