"""
Build and index script for the Enverus RAG system.
Extracts pages from the research paper, chunks them, embeds them,
and populates the persistent ChromaDB collection.
"""

import sys
import time
from pathlib import Path
from src.ingest import extract_pages_from_pdf
from src.chunking import chunk_page_records
from src.embeddings import EmbeddingManager
from src.vector_store import get_vector_store


def build_index(pdf_path: str = "data/agent_as_a_judge.pdf", reset: bool = False):
    print("=" * 70)
    print("ENVERUS RAG: OFFLINE DOCUMENT INGESTION & VECTOR INDEXING")
    print("=" * 70)

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"ERROR: Target PDF not found at {pdf.resolve()}")
        print("Please place 'agent_as_a_judge.pdf' in the data/ directory.")
        sys.exit(1)

    print(f"1. Extracting pages from: {pdf.name}")
    start_t = time.perf_counter()
    pages = extract_pages_from_pdf(pdf)
    print(f"   -> Extracted {len(pages)} pages in {time.perf_counter() - start_t:.2f}s")

    print(f"2. Performing structure-aware chunking...")
    start_t = time.perf_counter()
    chunks = chunk_page_records(pages)
    print(f"   -> Created {len(chunks)} chunks in {time.perf_counter() - start_t:.2f}s")

    print(f"3. Loading embedding model and embedding chunks...")
    start_t = time.perf_counter()
    manager = EmbeddingManager()
    embedded_chunks = manager.embed_chunks(chunks)
    print(f"   -> Embedded {len(embedded_chunks)} chunks (384-dim) in {time.perf_counter() - start_t:.2f}s")

    print(f"4. Indexing into persistent ChromaDB store...")
    start_t = time.perf_counter()
    store = get_vector_store()
    if reset:
        print("   -> Resetting existing collection...")
        store.reset_collection()
    upserted = store.upsert_embedded_chunks(embedded_chunks)
    total_count = store.count()
    print(f"   -> Successfully indexed {upserted} chunks in {time.perf_counter() - start_t:.2f}s")
    print(f"   -> Total records in collection '{store.collection_name}': {total_count}")
    print("=" * 70)
    print("INDEXING COMPLETE AND READY FOR RETRIEVAL.")
    print("=" * 70)


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    build_index(reset=reset_flag)
