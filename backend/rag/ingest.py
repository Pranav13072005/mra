import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from backend.core.config import settings


def build_vectorstore_from_pdf(pdf_path: str, session_id: str) -> Chroma:
    """
    Identical to Project 2's ingest.py, but session-scoped.
    Each uploaded PDF gets its own ChromaDB collection.
    Embedding model: BAAI/bge-small-en-v1.5 (same as Project 2).
    Chunk size: 512, overlap: 50 (same as Project 2).
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": settings.device}
    )

    persist_dir = os.path.join(settings.chroma_persist_dir, session_id)
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=f"session_{session_id}"
    )
    vectorstore.persist()
    print(f"[ingest] Built vectorstore: {len(chunks)} chunks | session={session_id}")
    return vectorstore