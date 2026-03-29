"""
src/retriever.py

Two-stage retriever:
Stage 1 - Bi-encoder (ChromaDB): Fast ANN search. Gets top-20 candidates.
Stage 2 - Cross-encoder (ms-marco-MiniLM): Slow but precise. Reranks top-20
           and returns the best top_k.

The bi-encoder embeds query and docs separately (fast).
The cross-encoder takes (query, doc) TOGETHER and scores them jointly (slow, precise).
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List
import os

# ---- Configuration -------------------------------------------------------
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DIR    = "./chroma_db"
CANDIDATES    = 20      # how many to fetch in stage 1
FINAL_TOP_K   = 4       # how many to return after reranking
# --------------------------------------------------------------------------


class RAGRetriever:
    """
    Wraps both stages of retrieval.
    use_reranker=True  -> two-stage (bi-encoder + cross-encoder)
    use_reranker=False -> single-stage (bi-encoder only)
    """

    # def __init__(self, use_reranker: bool = True):
    #     self.use_reranker = use_reranker

    #     # Stage 1: bi-encoder embedder (same model as ingestion)
    #     print(f"Loading bi-encoder: {EMBED_MODEL}")
    #     self.embeddings = HuggingFaceEmbeddings(
    #         model_name=EMBED_MODEL,
    #         model_kwargs={"device": "cpu"},
    #         encode_kwargs={"normalize_embeddings": True}
    #     )

    #     # Stage 1: ChromaDB vectorstore
    #     if not os.path.exists(CHROMA_DIR):
    #         raise FileNotFoundError(
    #             f"Vectorstore not found at {CHROMA_DIR}. Run src/ingest.py first."
    #         )
    #     # if not os.path.exists(CHROMA_DIR):
    #     #     print("Vector DB not found. Running ingestion pipeline...")
    #     #     from backend.rag.ingest import build_vectorstore_from_pdf            
    #     #     build_vectorstore_from_pdf()
    #     #     print("Vector DB built successfully.")
    #     self.vectorstore = Chroma(
    #         persist_directory=CHROMA_DIR,
    #         embedding_function=self.embeddings,
    #         collection_name="rag_collection"
    #     )

    #     # Stage 2: cross-encoder reranker (only loaded if needed)
    #     if use_reranker:
    #         print(f"Loading cross-encoder: {RERANK_MODEL}")
    #         self.reranker = CrossEncoder(
    #             RERANK_MODEL,
    #             max_length=512
    #         )

    def __init__(self, vectorstore=None, use_reranker: bool = True):
        self.use_reranker = use_reranker

        print(f"Loading bi-encoder: {EMBED_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # ✅ USE PASSED VECTORSTORE
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            if not os.path.exists(CHROMA_DIR):
                raise FileNotFoundError(
                    f"Vectorstore not found at {CHROMA_DIR}."
                )

            self.vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )

        if use_reranker:
            print(f"Loading cross-encoder: {RERANK_MODEL}")
            self.reranker = CrossEncoder(
                RERANK_MODEL,
                max_length=512
            )

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> List[Document]:
        """
        Retrieve the top_k most relevant document chunks for a query.
        """
        # ---- Stage 1: Bi-encoder ANN search ----------------------------
        candidates: List[Document] = self.vectorstore.similarity_search(
            query, k=CANDIDATES
        )
        if len(candidates) == 0:
            print("ERROR: No candidates retrieved from Chroma.")
            return []
        if not self.use_reranker:
            # Single-stage: just return the top_k from bi-encoder
            return candidates[:top_k]

        # ---- Stage 2: Cross-encoder reranking --------------------------
        # Build list of (query, passage) pairs for the cross-encoder
        
        # scores is a numpy array of floats, one per candidate
        pairs = [(query, doc.page_content) for doc in candidates]

        if len(pairs) == 0:
            print("ERROR: No pairs for reranker.")
            return []

        scores = self.reranker.predict(pairs)

        # Zip scores with documents, sort descending, return top_k
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        return [doc for _, doc in ranked[:top_k]]

    def get_vectorstore(self) -> Chroma:
        """Expose vectorstore for direct access if needed."""
        return self.vectorstore
    
def retrieve(query: str, vectorstore, top_k: int = 4):
    retriever = RAGRetriever(vectorstore=vectorstore, use_reranker=True)

    docs = retriever.retrieve(query, top_k)

    return [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in docs
    ]