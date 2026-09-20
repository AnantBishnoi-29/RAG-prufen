from __future__ import annotations

import atexit
import chromadb
from pathlib import Path
import pickle
import threading
from typing import Any, Literal
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore

# Base database directory
ROOT = Path(__file__).resolve().parent.parent
DATABASES_DIR = ROOT / "databases"
CHROMA_DIR = DATABASES_DIR / "chroma"
FAISS_DIR = DATABASES_DIR / "faiss"
QDRANT_DIR = DATABASES_DIR / "qdrant"
BM25_DIR = DATABASES_DIR / "bm25"

# Process-wide singleton cache for embedded Qdrant clients to avoid storage folder locking errors
_QDRANT_CLIENT_CACHE: dict[str, Any] = {}
_qdrant_lock = threading.Lock()


def get_qdrant_client(path: Path | str | None = None):
    """
    Returns a process-wide singleton QdrantClient instance for the given storage path.
    Reusing the embedded client prevents 'Storage folder is already accessed by another instance'
    file-lock collisions across queries and evaluation runs in the same process.
    """
    from qdrant_client import QdrantClient

    target_path = str(Path(path or QDRANT_DIR).resolve())
    with _qdrant_lock:
        if target_path not in _QDRANT_CLIENT_CACHE:
            _QDRANT_CLIENT_CACHE[target_path] = QdrantClient(path=target_path)
        return _QDRANT_CLIENT_CACHE[target_path]


def close_qdrant_clients():
    """Cleanly closes any cached QdrantClient instances to release folder locks."""
    with _qdrant_lock:
        for p, client in list(_QDRANT_CLIENT_CACHE.items()):
            try:
                client.close()
            except Exception:
                pass
        _QDRANT_CLIENT_CACHE.clear()


atexit.register(close_qdrant_clients)


def create_chroma_store(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    **kwargs,
) -> Chroma:
    """
    Creates or rebuilds a persistent Chroma vector store.
    If the collection already exists in Chroma, it is deleted first to prevent
    duplicate document appending on rebuilds.
    """
    target_dir = str(persist_directory or CHROMA_DIR)
    try:
        client = chromadb.PersistentClient(path=target_dir)
        existing_cols = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
        if collection_name in existing_cols:
            client.delete_collection(name=collection_name)
    except Exception:
        pass

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=target_dir,
        **kwargs,
    )


def create_faiss_store(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    index_name: str = "index",
    **kwargs,
) -> FAISS:
    """
    Creates an in-memory FAISS vector store and saves it locally to disk under collection_name.
    """
    base_dir = Path(persist_directory or FAISS_DIR)
    target_dir = base_dir / collection_name
    target_dir.mkdir(parents=True, exist_ok=True)

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        **kwargs,
    )
    vector_store.save_local(folder_path=str(target_dir), index_name=index_name)
    return vector_store


def create_qdrant_store(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    path: Path | str | None = None,
    **kwargs,
) -> QdrantVectorStore:
    """
    Creates an embedded Qdrant vector store on local disk (no Docker required).
    Reuses the singleton QdrantClient to prevent file-locking conflicts.
    If the collection already exists, it is deleted and recreated to prevent duplicate points on rebuilds.
    """
    from qdrant_client import models

    target_path = Path(path or QDRANT_DIR).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    client = get_qdrant_client(target_path)

    try:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    # Measure embedding dimension with probe query
    sample_emb = embeddings.embed_query("dimension_probe")
    dim = len(sample_emb)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    valid_keys = {
        "retrieval_mode",
        "vector_name",
        "content_payload_key",
        "metadata_payload_key",
        "distance",
        "sparse_embedding",
        "sparse_vector_name",
        "validate_embeddings",
        "validate_collection_config",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        **filtered_kwargs,
    )
    vector_store.add_documents(documents)
    return vector_store


class ThresholdBM25Retriever(BM25Retriever):
    """
    BM25 retriever with score thresholding:
    - Zero-match floor: drops any document with BM25 score <= 0.0 (zero matching terms).
    - Relative normalization: norm_score = score / max_score
      Drops documents with norm_score < score_threshold.
    - Preserves and enriches Document metadata with 'bm25_score' and 'relevance_score'.
    """
    score_threshold: float = 0.0

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)

        positive_indices = [i for i, s in enumerate(scores) if s > 0.0]
        if not positive_indices:
            return []

        sorted_indices = sorted(positive_indices, key=lambda i: scores[i], reverse=True)
        max_score = float(scores[sorted_indices[0]])

        results: list[Document] = []
        for idx in sorted_indices:
            raw_score = float(scores[idx])
            norm_score = raw_score / max_score if max_score > 0 else 0.0

            if self.score_threshold > 0.0 and norm_score < self.score_threshold:
                continue

            orig_doc = self.docs[idx]
            doc_meta = dict(orig_doc.metadata) if orig_doc.metadata else {}
            doc_meta["bm25_score"] = round(raw_score, 4)
            doc_meta["relevance_score"] = round(norm_score, 4)
            results.append(Document(page_content=orig_doc.page_content, metadata=doc_meta))

            if len(results) >= self.k:
                break

        return results


def create_bm25_retriever(
    documents: list[Document],
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    k: int = 3,
    score_threshold: float = 0.0,
    **kwargs,
) -> ThresholdBM25Retriever:
    """
    Creates an in-memory BM25 sparse keyword retriever and saves it to disk via pickle.
    """
    target_dir = Path(persist_directory or BM25_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{collection_name}.pkl"

    retriever = ThresholdBM25Retriever.from_documents(documents=documents, k=k, **kwargs)
    retriever.score_threshold = score_threshold
    with open(target_file, "wb") as f:
        pickle.dump(retriever, f)
    return retriever


def get_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    store_type: Literal["chroma", "faiss", "qdrant"] = "chroma",
    collection_name: str = "rag_collection",
    **kwargs,
) -> VectorStore:
    """
    Unified router to create and persist a vector store based on selected engine.
    """
    engine = store_type.lower().strip()

    if engine in ("chroma", "1"):
        return create_chroma_store(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name,
            **kwargs,
        )
    elif engine in ("faiss", "2"):
        return create_faiss_store(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name,
            **kwargs,
        )
    elif engine in ("qdrant", "3"):
        return create_qdrant_store(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown vector store: '{store_type}'. Supported options: 'chroma', 'faiss', 'qdrant'"
        )


# ---------------------------------------------------------------------------
# Store Existence Checkers (for Instant Cache Hit Detection)
# ---------------------------------------------------------------------------
def chroma_store_exists(
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
) -> bool:
    """Checks if a Chroma collection exists and contains indexed documents."""
    target_dir = str(persist_directory or CHROMA_DIR)
    if not Path(target_dir).exists():
        return False
    try:
        client = chromadb.PersistentClient(path=target_dir)
        col = client.get_collection(name=collection_name)
        return col.count() > 0
    except Exception:
        return False


def faiss_store_exists(
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    index_name: str = "index",
) -> bool:
    """Checks if a persistent FAISS index folder exists on disk."""
    base_dir = Path(persist_directory or FAISS_DIR)
    target_dir = base_dir / collection_name
    return (target_dir / f"{index_name}.faiss").exists() and (target_dir / f"{index_name}.pkl").exists()


def qdrant_store_exists(
    collection_name: str = "rag_collection",
    path: Path | str | None = None,
) -> bool:
    """Checks if a local Qdrant collection exists and has points."""
    target_path = Path(path or QDRANT_DIR).resolve()
    if not target_path.exists():
        return False
    try:
        client = get_qdrant_client(target_path)
        if client.collection_exists(collection_name=collection_name):
            return client.count(collection_name=collection_name).count > 0
        return False
    except Exception:
        return False


def bm25_store_exists(
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
) -> bool:
    """Checks if a pickled BM25 retriever index exists on disk."""
    target_dir = Path(persist_directory or BM25_DIR)
    target_file = target_dir / f"{collection_name}.pkl"
    return target_file.exists() and target_file.stat().st_size > 0


def vector_store_exists(
    store_type: str = "chroma",
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
) -> bool:
    """Unified check whether a collection already exists with data on disk."""
    engine = store_type.lower().strip()
    if engine in ("chroma", "1"):
        return chroma_store_exists(collection_name, persist_directory)
    elif engine in ("faiss", "2"):
        return faiss_store_exists(collection_name, persist_directory)
    elif engine in ("qdrant", "3"):
        return qdrant_store_exists(collection_name, persist_directory)
    elif engine in ("bm25", "4"):
        return bm25_store_exists(collection_name, persist_directory)
    return False


# ---------------------------------------------------------------------------
# Store Loaders (Mounts Existing Store in <50ms without Re-embedding)
# ---------------------------------------------------------------------------
def load_chroma_store(
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    **kwargs,
) -> Chroma:
    """Mounts an existing persistent Chroma store from disk."""
    target_dir = str(persist_directory or CHROMA_DIR)
    return Chroma(
        collection_name=collection_name,
        persist_directory=target_dir,
        embedding_function=embeddings,
        **kwargs,
    )


def load_faiss_store(
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    index_name: str = "index",
    **kwargs,
) -> FAISS:
    """Loads an existing saved FAISS index from disk."""
    base_dir = Path(persist_directory or FAISS_DIR)
    target_dir = base_dir / collection_name
    return FAISS.load_local(
        folder_path=str(target_dir),
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
        **kwargs,
    )


def load_qdrant_store(
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    path: Path | str | None = None,
    **kwargs,
) -> QdrantVectorStore:
    """Mounts an existing embedded Qdrant collection from disk using the singleton client."""
    target_path = Path(path or QDRANT_DIR).resolve()
    client = get_qdrant_client(target_path)

    valid_keys = {
        "retrieval_mode",
        "vector_name",
        "content_payload_key",
        "metadata_payload_key",
        "distance",
        "sparse_embedding",
        "sparse_vector_name",
        "validate_embeddings",
        "validate_collection_config",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        **filtered_kwargs,
    )


def load_bm25_retriever(
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    k: int = 3,
    score_threshold: float = 0.0,
) -> ThresholdBM25Retriever:
    """Loads a persisted BM25 retriever index from disk in <2ms."""
    target_dir = Path(persist_directory or BM25_DIR)
    target_file = target_dir / f"{collection_name}.pkl"
    if not target_file.exists():
        raise FileNotFoundError(f"BM25 index '{collection_name}' not found in {target_dir}")
    with open(target_file, "rb") as f:
        retriever = pickle.load(f)
    if not isinstance(retriever, ThresholdBM25Retriever):
        retriever.__class__ = ThresholdBM25Retriever
    retriever.k = k
    retriever.score_threshold = score_threshold
    return retriever


def load_vector_store(
    embeddings: Embeddings,
    store_type: str = "chroma",
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    **kwargs,
) -> VectorStore:
    """Unified router to load any existing vector store from disk."""
    engine = store_type.lower().strip()
    if engine in ("chroma", "1"):
        return load_chroma_store(embeddings, collection_name, persist_directory, **kwargs)
    elif engine in ("faiss", "2"):
        return load_faiss_store(embeddings, collection_name, persist_directory, **kwargs)
    elif engine in ("qdrant", "3"):
        return load_qdrant_store(embeddings, collection_name, persist_directory, **kwargs)
    else:
        raise ValueError(
            f"Unknown vector store: '{store_type}'. Supported options: 'chroma', 'faiss', 'qdrant'"
        )


def get_retriever(
    vector_store: VectorStore,
    k: int = 3,
    search_type: str = "similarity",
    score_threshold: float | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
    **kwargs,
) -> VectorStoreRetriever:
    """
    Returns a retriever interface from any LangChain VectorStore.
    Supports search_type:
    - 'similarity': standard k-NN top-k search
    - 'similarity_score_threshold': filters chunks where score >= score_threshold
    - 'mmr': maximal marginal relevance balancing relevance and chunk diversity
    """
    search_kwargs: dict[str, Any] = {"k": k, **kwargs}
    if search_type == "similarity_score_threshold" and score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    elif search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k if fetch_k is not None else max(k * 3, 6)
        if lambda_mult is not None:
            search_kwargs["lambda_mult"] = lambda_mult

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def main():
    import sys
    sys.path.append(str(ROOT))
    from components.embeddings import get_embeddings
    from components.retriever import retrieve_contexts

    print("""Vector Store Test:
    1. Chroma (databases/chroma)
    2. FAISS (databases/faiss)
    3. Qdrant Embedded (databases/qdrant)
    """)
    choice = input("Enter choice [1-3] (default: 1): ").strip() or "1"

    sample_docs = [
        Document(page_content="LangChain simplifies building context-aware LLM applications."),
        Document(page_content="Chroma, FAISS, and Qdrant are popular vector databases for RAG."),
        Document(page_content="Retrieval-Augmented Generation grounds LLM answers with relevant context."),
    ]

    print("Initializing embedding model...")
    embeddings = get_embeddings("openai")

    print(f"Creating vector store ({choice})...")
    vstore = get_vector_store(sample_docs, embeddings, store_type=choice)
    retriever = get_retriever(vstore, k=2)

    query = "Which vector databases are used for RAG?"
    print(f"\nQuery: {query}")
    results = retrieve_contexts(retriever, query)
    for i, res in enumerate(results, 1):
        print(f"  [{i}] {res}")


if __name__ == "__main__":
    main()
