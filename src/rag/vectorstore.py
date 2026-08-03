"""
Thin wrapper around the vector store (Chroma).

This is the ONLY module that imports chromadb or the embedding model.
Everything else calls similarity_search() and gets plain dicts back, which
means the store can be swapped without touching an agent.

The embedding model is loaded here and only here, which is what guarantees
the classic RAG footgun cannot happen: a query embedded with a different
model than the corpus would return nearest neighbours that are nearest in
the wrong space, and the failure is silent — you get confident, plausible,
wrong chunks. Ingestion and query both go through embed().
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src import config


@lru_cache(maxsize=1)
def get_client():
    """Persistent Chroma client rooted at config.VECTOR_STORE_DIR."""
    import chromadb

    return chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)


@lru_cache(maxsize=1)
def get_embedder():
    """The one sentence-transformers model used for ingest AND query."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings with the corpus embedding model."""
    vectors = get_embedder().encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [vector.tolist() for vector in vectors]


def get_or_create_collection(name: str | None = None):
    """Return the guideline collection, creating it if needed.

    Cosine distance rather than the L2 default: the embeddings are
    normalised, and cosine keeps the distance in [0, 2] so it converts to a
    readable 0-1 similarity score for the UI.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=name or config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(name: str | None = None):
    """Drop and recreate the collection. Used by a full re-ingest."""
    client = get_client()
    collection_name = name or config.COLLECTION_NAME
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # nothing to delete on a first run
    return get_or_create_collection(collection_name)


def collection_count(name: str | None = None) -> int:
    """Number of chunks currently indexed. 0 means ingestion hasn't run."""
    try:
        return get_or_create_collection(name).count()
    except Exception:
        return 0


def similarity_search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Embed `query` and return the top-k chunks with metadata and score.

    Returns [] rather than raising when the collection is empty, so a
    missing corpus surfaces as "no guidance found" downstream instead of a
    crash mid-graph.
    """
    k = k or config.TOP_K_RETRIEVED
    collection = get_or_create_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=embed([query]),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        chunks.append(
            {
                "text": text,
                "source_doc": metadata.get("source_doc", "unknown"),
                "section": metadata.get("section", ""),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "page": int(metadata.get("page", 0)),
                # cosine distance -> similarity, clamped for display
                "score": round(max(0.0, 1.0 - float(distance)), 4),
            }
        )
    return chunks
