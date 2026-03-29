"""
src/generator.py

Takes retrieved document chunks + user query and calls the LLM
to produce a grounded answer.

Key design: the system prompt FORCES the model to answer only from
the provided context. This is what makes answers faithful and measurable.
"""

from groq import Groq
from langchain_core.documents import Document
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from .env file

# ---- Configuration -------------------------------------------------------
MODEL_NAME   = "llama-3.1-8b-instant"   # fast, free on Groq
TEMPERATURE  = 0.1                 # low = deterministic, faithful answers
MAX_TOKENS   = 1024               # max answer length
# --------------------------------------------------------------------------

# System prompt: the most important part of the generator
# This instruction is what separates a RAG system from a hallucinating chatbot
SYSTEM_PROMPT = """You are a precise, expert question-answering assistant.

You are given CONTEXT passages extracted from research documents.
Your job:
1. Answer the question using ONLY information present in the CONTEXT.
2. If the context does not contain the answer, respond exactly: "The provided context does not contain enough information to answer this question."
3. Do NOT use your general knowledge. Do NOT guess. Do NOT add information not in the context.
4. Be concise and direct. Cite which passage your answer comes from when possible."""


def build_context_string(docs: List[Document]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    This makes it easier for the LLM to cite sources.
    """
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Passage {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str,
                    context_docs: List[Document]) -> Dict:
    """
    Generate a grounded answer given query and retrieved docs.

    Returns a dict with:
    - answer: str
    - context: str (the formatted context passed to LLM)
    - model: str
    - usage: token usage stats
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_str = build_context_string(context_docs)

    user_message = f"""CONTEXT:
{context_str}

QUESTION: {query}

Answer based ONLY on the context above:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    return {
        "answer":  response.choices[0].message.content.strip(),
        "context": context_str,
        "model":   MODEL_NAME,
        "usage":   response.usage
    }