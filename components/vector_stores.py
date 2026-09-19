from __future__ import annotations

import chromadb
from pathlib import Path
import pickle
from typing import Literal
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore

# Base database directory
ROOT = Path(__file__).resolve().parent.parent
DATABASES_DIR = ROOT / "databases"
CHROMA_DIR = DATABASES_DIR / "chroma"
FAISS_DIR = DATABASES_DIR / "faiss"
QDRANT_DIR = DATABASES_DIR / "qdrant"
BM25_DIR = DATABASES_DIR / "bm25"


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
    If the collection already exists, it is deleted first to prevent duplicate points on rebuilds.
    """
    target_path = str(path or QDRANT_DIR)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(path=target_path)
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=target_path,
        collection_name=collection_name,
        **kwargs,
    )


def create_bm25_retriever(
    documents: list[Document],
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    k: int = 3,
    **kwargs,
) -> BM25Retriever:
    """
    Creates an in-memory BM25 sparse keyword retriever and saves it to disk via pickle.
    """
    target_dir = Path(persist_directory or BM25_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{collection_name}.pkl"

    retriever = BM25Retriever.from_documents(documents=documents, k=k, **kwargs)
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
    target_path = str(path or QDRANT_DIR)
    if not Path(target_path).exists():
        return False
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(path=target_path)
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
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
    """Mounts an existing embedded Qdrant collection from disk."""
    target_path = str(path or QDRANT_DIR)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        path=target_path,
        **kwargs,
    )


def load_bm25_retriever(
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    k: int = 3,
) -> BM25Retriever:
    """Loads a persisted BM25 retriever index from disk in <2ms."""
    target_dir = Path(persist_directory or BM25_DIR)
    target_file = target_dir / f"{collection_name}.pkl"
    if not target_file.exists():
        raise FileNotFoundError(f"BM25 index '{collection_name}' not found in {target_dir}")
    with open(target_file, "rb") as f:
        retriever = pickle.load(f)
    retriever.k = k
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
    **kwargs,
) -> VectorStoreRetriever:
    """
    Returns a retriever interface from any LangChain VectorStore.
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, **kwargs},
    )


def retrieve_contexts(retriever: VectorStoreRetriever, query: str) -> list[str]:
    """
    Invokes the retriever with a query and returns plain string contexts.
    """
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def main():
    import sys
    sys.path.append(str(ROOT))
    from components.embeddings import get_embeddings

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
