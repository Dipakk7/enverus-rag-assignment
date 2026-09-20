"""
Source package for the Enverus RAG assignment.
"""

from src.chunking import Chunk, chunk_page_records
from src.embeddings import (
    EmbeddedChunk,
    EmbeddingManager,
    embed_chunks,
    embed_query,
    load_embedding_model,
)
from src.generator import (
    DEFAULT_SYSTEM_PROMPT,
    LLMGenerator,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    build_rag_prompt,
    get_generator,
)
from src.ingest import PageRecord, extract_pages_from_pdf
from src.rag_pipeline import (
    RAGPipeline,
    RAGResponse,
    RAGSource,
    get_rag_pipeline,
)
from src.retriever import (
    RetrievalResult,
    SemanticRetriever,
    get_retriever,
    retrieve,
)
from src.vector_store import StoredRecord, VectorStore, get_vector_store

__all__ = [
    "PageRecord",
    "extract_pages_from_pdf",
    "Chunk",
    "chunk_page_records",
    "EmbeddedChunk",
    "EmbeddingManager",
    "embed_chunks",
    "embed_query",
    "load_embedding_model",
    "StoredRecord",
    "VectorStore",
    "get_vector_store",
    "RetrievalResult",
    "SemanticRetriever",
    "get_retriever",
    "retrieve",
    "LLMGenerator",
    "get_generator",
    "build_rag_prompt",
    "DEFAULT_SYSTEM_PROMPT",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "OllamaResponseError",
    "RAGPipeline",
    "RAGResponse",
    "RAGSource",
    "get_rag_pipeline",
]
