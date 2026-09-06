from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from components.embeddings import get_embeddings
from components.loaders import load_documents
from components.rerankers import rerank
from components.text_splitters import split_documents
from components.vector_stores import (
    get_retriever as _get_retriever,
    get_vector_store,
)


def build_retriever(
    doc_path: str,
    loader_type: int | str = "pdf",
    splitter_type: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_provider: str = "openai",
    vector_store_type: str = "chroma",
    collection_name: str = "rag_collection",
    k: int = 3,
) -> VectorStoreRetriever:
    """
    End-to-end retriever orchestrator:
    1. Loads document (components/loaders.py)
    2. Chunks document (components/text_splitters.py)
    3. Initializes embeddings (components/embeddings.py)
    4. Indexes into vector store (components/vector_stores.py)
    5. Returns configured retriever
    """
    # 1. Load document
    choice = 2 if loader_type in ("pdf", "pypdf", 2) else loader_type
    docs = load_documents(choice=choice, path=doc_path)

    # 2. Split into chunks
    chunks = split_documents(
        documents=docs,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. Embedding model
    embeddings = get_embeddings(provider=embedding_provider)

    # 4. Vector store
    vector_store = get_vector_store(
        documents=chunks,
        embeddings=embeddings,
        store_type=vector_store_type,
        collection_name=collection_name,
    )

    # 5. Return retriever
    return _get_retriever(vector_store, k=k)


def retrieve(
    retriever: VectorStoreRetriever,
    query: str,
    use_reranker: bool = False,
    top_k: int = 3,
    rerank_top_n: int = 3,
) -> list[Document]:
    """
    Retrieves matching documents for a query, with optional cross-encoder reranking.
    """
    # If using reranker, fetch 3x candidates first to let reranker filter the best
    fetch_k = top_k * 3 if use_reranker else top_k
    retriever.search_kwargs["k"] = fetch_k

    docs = retriever.invoke(query)

    if use_reranker and docs:
        return rerank(query=query, documents=docs, top_n=rerank_top_n)

    return docs[:top_k]


def retrieve_contexts(
    retriever: VectorStoreRetriever,
    query: str,
    use_reranker: bool = False,
    top_k: int = 3,
    rerank_top_n: int = 3,
) -> list[str]:
    """
    Retrieves matching contexts as plain strings (for generators or evaluations).
    """
    docs = retrieve(
        retriever=retriever,
        query=query,
        use_reranker=use_reranker,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
    )
    return [doc.page_content for doc in docs]


# Backwards-compatibility helpers for main.py and test_rag.py
def create_vector_store(path: str, **kwargs):
    return build_retriever(doc_path=path, **kwargs)


def get_retriever(vector_store_or_retriever, k: int = 3) -> VectorStoreRetriever:
    if hasattr(vector_store_or_retriever, "as_retriever"):
        return _get_retriever(vector_store_or_retriever, k=k)
    return vector_store_or_retriever