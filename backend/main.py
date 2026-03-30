from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.upload import router as upload_router
from backend.routes.ask import router as ask_router

app = FastAPI(
    title="Multimodal Research Assistant API",
    description="Three-pipeline VQA over scientific papers and figures. "
                "Evaluates whether retrieval-augmented context improves "
                "vision-language model performance by question type.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(upload_router)
app.include_router(ask_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# @app.on_event("startup")
# async def startup():
#     # Pre-warm LLaVA on startup so first request is not slow
#     from backend.vision.llava_model import _load
#     _load()