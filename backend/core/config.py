from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = ""
    model_backend: str = "llava"
    device: str = "cuda"
    max_image_size: int = 1024
    chroma_persist_dir: str = "./chroma_sessions"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 20
    rerank_top_k: int = 4

    class Config:
        env_file = ".env"
        protected_namespaces = ()

settings = Settings()