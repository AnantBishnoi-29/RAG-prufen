import hashlib
from pathlib import Path
import re
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from components.embeddings import get_embeddings
from components.loaders import load_documents
from components.rerankers import rerank
from components.text_splitters import split_documents
from components.vector_stores import (
    bm25_store_exists,
    create_bm25_retriever,
    get_retriever as _get_retriever,
    get_vector_store,
    load_bm25_retriever,
    load_vector_store,
    vector_store_exists,
)


def get_collection_name(
    doc_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_provider: str = "openai",
) -> str:
    """
    Computes a deterministic, valid collection slug based on document and chunking settings.
    Ensures safe characters (alphanumeric and underscore) conforming to Chroma/Qdrant standards.
    """
    doc_stem = Path(doc_path).stem
    cleaned_stem = re.sub(r"[^a-zA-Z0-9_]", "_", doc_stem)
    clean_slug = cleaned_stem.strip("_")[:24].rstrip("_") or "doc"

    param_str = f"{Path(doc_path).name}_{chunk_size}_{chunk_overlap}_{embedding_provider}"
    param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:8]

    return f"c_{clean_slug}_{chunk_size}_{param_hash}"


def reciprocal_rank_fusion(
    doc_lists: list[list[Document]],
    c: int = 60,
    top_k: int = 3,
) -> list[Document]:
    """
    Combines multiple ranked document lists using Reciprocal Rank Fusion (RRF).
    Formula: score(d) = sum(1 / (c + rank_i))
    Deduplicates by document page_content and preserves original metadata.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content.strip()
            scores[key] = scores.get(key, 0.0) + (1.0 / (c + rank))
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:top_k]]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that executes dense vector search and BM25 sparse keyword
    search concurrently, fusing candidates using Reciprocal Rank Fusion (RRF).
    """
    dense_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    c: int = 60
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        fetch_k = max(self.k * 2, 6)
        for r in (self.dense_retriever, self.bm25_retriever):
            if hasattr(r, "search_kwargs"):
                r.search_kwargs["k"] = fetch_k
            elif hasattr(r, "k"):
                r.k = fetch_k

        dense_docs = self.dense_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        return reciprocal_rank_fusion(
            doc_lists=[dense_docs, bm25_docs],
            c=self.c,
            top_k=self.k,
        )


def build_retriever(
    doc_path: str,
    loader_type: int | str = "auto",
    splitter_type: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_provider: str = "openai",
    vector_store_type: str = "chroma",
    collection_name: str | None = None,
    k: int = 3,
    force_rebuild: bool = False,
    use_hybrid: bool = False,
) -> BaseRetriever:
    """
    End-to-end retriever orchestrator with intelligent index caching:
    - If an index already exists for the given document + chunk parameters, mounts from disk in <30ms.
    - Otherwise (or if force_rebuild=True), loads, chunks, embeds (or indexes), and saves the new index to disk.
    - Supports dense vector stores ('chroma', 'faiss', 'qdrant') and sparse keyword retrieval ('bm25').
    - Supports dense vector stores ('chroma', 'faiss', 'qdrant'), sparse keyword retrieval ('bm25'),
      and hybrid retrieval (dense + BM25 via Reciprocal Rank Fusion).
    """
    engine = vector_store_type.lower().strip()
    is_bm25 = engine in ("bm25", "4")

    # Handle Hybrid Retrieval: Query dense store + BM25 and fuse via RRF
    if use_hybrid and not is_bm25:
        dense_retriever = build_retriever(
            doc_path=doc_path,
            loader_type=loader_type,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            vector_store_type=vector_store_type,
            collection_name=collection_name,
            k=k,
            force_rebuild=force_rebuild,
            use_hybrid=False,
        )
        bm25_retriever = build_retriever(
            doc_path=doc_path,
            loader_type=loader_type,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            vector_store_type="bm25",
            collection_name=None,
            k=k,
            force_rebuild=force_rebuild,
            use_hybrid=False,
        )
        print(f"[Hybrid Retriever Ready] Combining '{vector_store_type}' (Dense) + BM25 (Sparse) via RRF (k={k})")
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            k=k,
        )

    # 1. Resolve deterministic collection name
    if collection_name is None or collection_name == "rag_collection":
        target_collection = get_collection_name(
            doc_path=doc_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider="bm25" if is_bm25 else embedding_provider,
        )
    else:
        target_collection = collection_name

    # Handle BM25 (Sparse Keyword Search)
    if is_bm25:
        if not force_rebuild and bm25_store_exists(collection_name=target_collection):
            print(f"[Retriever Cache HIT] Loading existing BM25 index for '{Path(doc_path).name}' ({target_collection})")
            return load_bm25_retriever(collection_name=target_collection, k=k)

        print(f"[Retriever Cache MISS] Building new BM25 index for '{Path(doc_path).name}' ({target_collection})...")
        docs = load_documents(path=doc_path, choice=loader_type)
        chunks = split_documents(
            documents=docs,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return create_bm25_retriever(
            documents=chunks,
            collection_name=target_collection,
            k=k,
        )

    # Handle Dense Vector Stores (Chroma, FAISS, Qdrant)
    embeddings = get_embeddings(provider=embedding_provider)

    # 2. Check cache hit (instant disk mount)
    if not force_rebuild and vector_store_exists(store_type=vector_store_type, collection_name=target_collection):
        print(f"[Retriever Cache HIT] Loading existing '{vector_store_type}' index for '{Path(doc_path).name}' ({target_collection})")
        vector_store = load_vector_store(
            embeddings=embeddings,
            store_type=vector_store_type,
            collection_name=target_collection,
        )
        return _get_retriever(vector_store, k=k)

    # 3. Cache miss: Load, chunk, embed, and persist
    print(f"[Retriever Cache MISS] Building new '{vector_store_type}' index for '{Path(doc_path).name}' ({target_collection})...")
    docs = load_documents(path=doc_path, choice=loader_type)

    chunks = split_documents(
        documents=docs,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    vector_store = get_vector_store(
        documents=chunks,
        embeddings=embeddings,
        store_type=vector_store_type,
        collection_name=target_collection,
    )


    return _get_retriever(vector_store, k=k)


def retrieve(
    retriever: BaseRetriever,
    query: str,
    use_reranker: bool = False,
    top_k: int = 3,
    rerank_top_n: int | None = None,
) -> list[Document]:
    """
    Retrieves matching documents for a query, with optional cross-encoder reranking.
    """
    effective_top_n = rerank_top_n if rerank_top_n is not None else top_k
    # If using reranker, fetch 3x candidates first to let reranker filter the best
    fetch_k = effective_top_n * 3 if use_reranker else effective_top_n
    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = fetch_k
    elif hasattr(retriever, "k"):
        retriever.k = fetch_k

    docs = retriever.invoke(query)

    if use_reranker and docs:
        return rerank(query=query, documents=docs, top_n=effective_top_n)

    return docs[:effective_top_n]


def retrieve_contexts(
    retriever: BaseRetriever,
    query: str,
    use_reranker: bool = False,
    top_k: int = 3,
    rerank_top_n: int | None = None,
    return_documents: bool = False,
) -> list[str] | list[Document]:
    """
    Retrieves matching contexts. By default returns plain strings (for generators or evaluations).
    If return_documents=True, returns full LangChain Document objects preserving metadata.
    """
    docs = retrieve(
        retriever=retriever,
        query=query,
        use_reranker=use_reranker,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
    )
    if return_documents:
        return docs
    return [doc.page_content for doc in docs]


# Backwards-compatibility helpers for main.py and test_rag.py
def create_vector_store(path: str, **kwargs):
    return build_retriever(doc_path=path, **kwargs)


def get_retriever(vector_store_or_retriever, k: int = 3) -> BaseRetriever:
    if hasattr(vector_store_or_retriever, "as_retriever"):
        return _get_retriever(vector_store_or_retriever, k=k)
    return vector_store_or_retriever