import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.session_store import store
from backend.fusion.multimodal_pipeline import run_all_pipelines

router = APIRouter()


class AskRequest(BaseModel):
    session_id: str
    question: str


@router.post("/ask")
async def ask(req: AskRequest):
    """
    Run all three pipelines on the uploaded image + question.
    Returns side-by-side answers from:
      - Pipeline A: vision-only (LLaVA)
      - Pipeline B: RAG-only (text retrieval)
      - Pipeline C: multimodal RAG (fusion, primary system)
    """
    session = store.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. POST to /upload first.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    try:
        result = run_all_pipelines(
            session.image,
            req.question,
            session.vectorstore
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    session.add_result(result)
    return result


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    return {"session_id": session_id, "history": session.history}