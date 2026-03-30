import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.vision.image_processor import ImageProcessor
from backend.rag.ingest import build_vectorstore_from_pdf
from backend.core.session_store import store

router = APIRouter()
processor = ImageProcessor()


@router.post("/upload")
async def upload_files(
    pdf: UploadFile = File(..., description="Research paper PDF"),
    figure: UploadFile = File(..., description="Figure image from the paper")
):
    """
    Upload a paper PDF and one of its figures together.
    Creates a session-scoped ChromaDB from the PDF.
    Returns session_id used for all subsequent /ask calls.
    """
    if pdf.content_type != "application/pdf":
        raise HTTPException(400, "First file must be a PDF")
    if not figure.content_type.startswith("image/"):
        raise HTTPException(400, "Second file must be an image")

    figure_bytes = await figure.read()
    if len(figure_bytes) > 15 * 1024 * 1024:
        raise HTTPException(413, "Figure image too large. Max 15MB.")

    # Process figure
    try:
        figure_result = processor.process(figure_bytes)
    except ValueError as e:
        raise HTTPException(422, f"Invalid figure: {e}")

    # Build session-scoped vectorstore from PDF
    pdf_bytes = await pdf.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        session_id = store.create(figure_result["image"], None)  # placeholder
        vectorstore = build_vectorstore_from_pdf(tmp_path, session_id)
        store.get(session_id).vectorstore = vectorstore
    except Exception as e:
        raise HTTPException(500, f"Failed to process PDF: {e}")
    finally:
        os.unlink(tmp_path)

    return {
        "session_id": session_id,
        "figure_size": figure_result["processed_size"],
        "figure_preview_base64": figure_result["base64"],
        "message": "Upload successful. Send questions to /ask."
    }