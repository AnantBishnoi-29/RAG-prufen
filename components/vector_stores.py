from pathlib import Path
from typing import Literal
from langchain_chroma import Chroma
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


def create_chroma_store(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
    persist_directory: Path | str | None = None,
    **kwargs,
) -> Chroma:
    """
    Creates or updates a persistent Chroma vector store.
    """
    target_dir = str(persist_directory or CHROMA_DIR)
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
    persist_directory: Path | str | None = None,
    index_name: str = "index",
    **kwargs,
) -> FAISS:
    """
    Creates an in-memory FAISS vector store and saves it locally to disk.
    """
    target_dir = Path(persist_directory or FAISS_DIR)
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
    """
    target_path = str(path or QDRANT_DIR)
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=target_path,
        collection_name=collection_name,
        **kwargs,
    )


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
