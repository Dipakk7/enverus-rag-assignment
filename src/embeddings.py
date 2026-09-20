"""
Embedding module for the Enverus RAG pipeline.

Generates dense vector embeddings for document chunks and user queries
using local SentenceTransformers models.
"""

from dataclasses import dataclass
import os
from typing import List, Optional
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    Associates a document Chunk with its generated dense embedding vector.

    Attributes:
        chunk: The source Chunk containing chunk_id, page_number, text, and section.
        embedding: The dense vector representation as a list of floats.
    """
    chunk: Chunk
    embedding: List[float]

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def section(self) -> Optional[str]:
        return self.chunk.section


class EmbeddingManager:
    """
    Manages model initialization, caching, and inference for vector embeddings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initializes the embedding manager.

        Args:
            model_name: HuggingFace model identifier or local model path.
                        Defaults to EMBEDDING_MODEL_NAME env var or 'all-MiniLM-L6-v2'.
            normalize_embeddings: Whether to L2-normalize output vectors to unit length.
                                 L2 normalization enables cosine similarity to be computed
                                 efficiently via standard dot product and prevents document
                                 length variations from biasing retrieval scores.
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.normalize_embeddings = normalize_embeddings
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily loads and returns the SentenceTransformer model instance."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self._embedding_dim = self._model.get_embedding_dimension()
            else:
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Returns the vector dimensionality detected from the loaded model."""
        if self._embedding_dim is None:
            _ = self.model
        return self._embedding_dim

    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single text string.

        Args:
            text: Input text string.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ValueError: If the input text is empty or contains only whitespace.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        vector = self.model.encode(
            text.strip(),
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vector.tolist() if isinstance(vector, np.ndarray) else list(vector)

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a user query using the same pipeline as document chunks.

        Args:
            query: The user search query.

        Returns:
            Dense embedding vector as a list of floats.
        """
        return self.embed_text(query)

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 32,
    ) -> List[EmbeddedChunk]:
        """
        Generates embeddings for a list of document chunks while preserving metadata.

        Args:
            chunks: List of Chunk objects from src.chunking.
            batch_size: Number of texts per forward pass.

        Returns:
            List of EmbeddedChunk items strictly aligned with the input chunk order.

        Raises:
            ValueError: If any chunk has empty or whitespace-only text.
            RuntimeError: If the output vector count mismatches the input chunk count.
        """
        if not chunks:
            return []

        texts: List[str] = []
        for idx, chunk in enumerate(chunks):
            clean_text = chunk.text.strip()
            if not clean_text:
                raise ValueError(
                    f"Chunk at index {idx} ({chunk.chunk_id}) contains empty or whitespace-only text."
                )
            texts.append(clean_text)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        vector_list: List[List[float]]
        if isinstance(vectors, np.ndarray):
            vector_list = vectors.tolist()
        else:
            vector_list = [list(v) for v in vectors]

        if len(vector_list) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(chunks)}, but generated {len(vector_list)}"
            )

        return [
            EmbeddedChunk(chunk=chunk, embedding=vec)
            for chunk, vec in zip(chunks, vector_list)
        ]


def load_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Convenience helper to load the SentenceTransformer model."""
    manager = EmbeddingManager(model_name=model_name)
    return manager.model


def embed_chunks(
    chunks: List[Chunk],
    model_name: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32,
) -> List[EmbeddedChunk]:
    """Convenience helper to embed a list of Chunk objects."""
    manager = EmbeddingManager(model_name=model_name, normalize_embeddings=normalize)
    return manager.embed_chunks(chunks, batch_size=batch_size)


def embed_query(
    query: str,
    model_name: Optional[str] = None,
    normalize: bool = True,
) -> List[float]:
    """Convenience helper to embed a single query string."""
    manager = EmbeddingManager(model_name=model_name, normalize_embeddings=normalize)
    return manager.embed_query(query)
