"""
Vector store module for the Enverus RAG pipeline.

Manages persistent ChromaDB vector storage, collection lifecycle,
and metadata indexing for embedded document chunks.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from src.embeddings import EmbeddedChunk

load_dotenv()

DEFAULT_CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agent_as_a_judge")
DISTANCE_METRIC = "cosine"


@dataclass(frozen=True)
class StoredRecord:
    """
    Structured representation of a record retrieved from ChromaDB.

    Attributes:
        chunk_id: Unique document chunk identifier.
        document: Raw text payload stored in the database.
        page_number: 1-indexed source document page number.
        section: Source section or appendix heading (or None).
        embedding: Vector embedding components as a list of floats (optional).
    """
    chunk_id: str
    document: str
    page_number: int
    section: Optional[str]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "page_number": self.page_number,
            "section": self.section,
            "embedding": self.embedding,
        }


class VectorStore:
    """
    Manages local persistent storage and indexing of embedded chunks in ChromaDB.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initializes the ChromaDB vector store.

        Args:
            persist_dir: Directory path for ChromaDB files.
                         Defaults to CHROMA_PERSIST_DIR env var or 'chroma_db'.
            collection_name: Name of the vector collection.
                             Defaults to CHROMA_COLLECTION_NAME env var or 'agent_as_a_judge'.
        """
        self.persist_dir = Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PERSIST_DIR))
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)

        # Ensure persistence directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazily creates and returns the persistent ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """
        Lazily gets or creates the target collection with cosine distance.
        Cosine distance is chosen to match normalized embeddings from Phase 4.
        """
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": DISTANCE_METRIC},
            )
        return self._collection

    def count(self) -> int:
        """Returns the total number of records in the collection."""
        return self.collection.count()

    def get_all_ids(self) -> List[str]:
        """Returns a list of all chunk IDs currently stored in the collection."""
        res = self.collection.get(include=[])
        return res.get("ids", [])

    def upsert_embedded_chunks(
        self,
        embedded_chunks: List[EmbeddedChunk],
        batch_size: int = 64,
    ) -> int:
        """
        Inserts or updates embedded chunks in ChromaDB.

        Deterministic and idempotent: re-indexing the same chunks will update
        existing records without creating duplicates.

        Args:
            embedded_chunks: List of EmbeddedChunk objects to index.
            batch_size: Batch size for database upserts.

        Returns:
            The number of records upserted.

        Raises:
            ValueError: If embedded_chunks is empty or contains invalid data.
        """
        if not embedded_chunks:
            return 0

        total_upserted = 0
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i : i + batch_size]

            ids: List[str] = []
            documents: List[str] = []
            embeddings: List[List[float]] = []
            metadatas: List[Dict[str, Any]] = []

            for idx, item in enumerate(batch):
                if not item.chunk_id or not item.chunk_id.strip():
                    raise ValueError(f"Record at batch index {idx} has invalid chunk_id.")
                if not item.text or not item.text.strip():
                    raise ValueError(f"Record {item.chunk_id} has empty document text.")
                if not item.embedding:
                    raise ValueError(f"Record {item.chunk_id} is missing embedding.")

                ids.append(item.chunk_id)
                documents.append(item.text)
                embeddings.append(item.embedding)

                # Safely encode metadata (ChromaDB does not permit None values)
                metadatas.append(
                    {
                        "chunk_id": item.chunk_id,
                        "page_number": item.page_number,
                        "section": item.section if item.section is not None else "",
                    }
                )

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_upserted += len(batch)

        return total_upserted

    def get_by_id(
        self,
        chunk_id: str,
        include_embedding: bool = True,
    ) -> Optional[StoredRecord]:
        """
        Retrieves a single stored record by its chunk_id.

        Args:
            chunk_id: The ID of the chunk to retrieve.
            include_embedding: Whether to return the vector embedding.

        Returns:
            A StoredRecord object if found, otherwise None.
        """
        if not chunk_id or not chunk_id.strip():
            return None

        include: List[str] = ["documents", "metadatas"]
        if include_embedding:
            include.append("embeddings")

        result = self.collection.get(
            ids=[chunk_id.strip()],
            include=include,
        )

        ids = result.get("ids", [])
        if not ids:
            return None

        doc = result["documents"][0]
        meta = result["metadatas"][0]

        # Restore None for empty section string
        raw_section = meta.get("section", "")
        section = raw_section if raw_section != "" else None

        emb = None
        if include_embedding and result.get("embeddings") is not None:
            raw_emb = result["embeddings"][0]
            emb = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)

        return StoredRecord(
            chunk_id=ids[0],
            document=doc,
            page_number=meta.get("page_number", 0),
            section=section,
            embedding=emb,
        )

    def get_by_ids(
        self,
        chunk_ids: List[str],
        include_embedding: bool = False,
    ) -> List[StoredRecord]:
        """
        Retrieves multiple records by their chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers.
            include_embedding: Whether to include embedding vectors.

        Returns:
            List of StoredRecord items found.
        """
        if not chunk_ids:
            return []

        clean_ids = [cid.strip() for cid in chunk_ids if cid and cid.strip()]
        if not clean_ids:
            return []

        include: List[str] = ["documents", "metadatas"]
        if include_embedding:
            include.append("embeddings")

        result = self.collection.get(
            ids=clean_ids,
            include=include,
        )

        records: List[StoredRecord] = []
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        embs = result.get("embeddings")

        for idx, cid in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            raw_sec = meta.get("section", "")
            sec = raw_sec if raw_sec != "" else None

            emb = None
            if include_embedding and embs is not None and idx < len(embs):
                raw_emb = embs[idx]
                emb = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)

            records.append(
                StoredRecord(
                    chunk_id=cid,
                    document=docs[idx] if idx < len(docs) else "",
                    page_number=meta.get("page_number", 0),
                    section=sec,
                    embedding=emb,
                )
            )

        return records

    def reset_collection(self) -> None:
        """
        Deletes and recreates the target collection.
        Affects ONLY the target ChromaDB collection; does not touch files.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection might not exist yet
        self._collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )


def get_vector_store(
    persist_dir: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
) -> VectorStore:
    """Factory helper to obtain a VectorStore instance."""
    return VectorStore(persist_dir=persist_dir, collection_name=collection_name)
