from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Cache loaded models so weights aren't reloaded repeatedly
_MODEL_CACHE: dict[str, CrossEncoder] = {}


def get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cpu",
) -> CrossEncoder:
    """
    Returns a cached CrossEncoder instance.
    Defaults to the lightweight ms-marco-MiniLM-L-6-v2 model (~80MB).
    """
    cache_key = f"{model_name}_{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = CrossEncoder(model_name, device=device)
    return _MODEL_CACHE[cache_key]


def rerank(
    query: str,
    documents: list[Document],
    top_n: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cpu",
) -> list[Document]:
    """
    Reranks candidate Document objects against a query using a Cross-Encoder.
    Returns the top_n most relevant Document objects with metadata preserved.
    """
    if not documents:
        return []

    model = get_cross_encoder(model_name=model_name, device=device)

    # Clean direct access — no isinstance checks
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    # Sort documents descending by cross-encoder score
    scored_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_documents[:top_n]]


def main():
    print("Cross-Encoder Reranker Test:")
    sample_query = "How to cancel a subscription without paying a fee?"

    # Chunks retrieved from a hypothetical Stage 1 vector search
    candidates = [
        Document(
            page_content="Our standard annual subscription fee is $120, billed recurringly on January 1st.",
            metadata={"source": "pricing.pdf", "page": 1},
        ),
        Document(
            page_content="You can terminate your account under Settings at any time with no penalties applied.",
            metadata={"source": "terms.pdf", "page": 4},
        ),
        Document(
            page_content="To upgrade your subscription to enterprise tier, contact our sales department.",
            metadata={"source": "upgrade.pdf", "page": 2},
        ),
        Document(
            page_content="Subscription fees are non-refundable after 30 days of purchase.",
            metadata={"source": "refunds.pdf", "page": 3},
        ),
    ]

    print(f"\nQuery: {sample_query}")
    print(f"\nCandidate Documents before Reranking ({len(candidates)} items):")
    for i, doc in enumerate(candidates, 1):
        print(f"  [{i}] ({doc.metadata['source']}): {doc.page_content}")

    print("\nReranking with cross-encoder/ms-marco-MiniLM-L-6-v2...")
    results = rerank(query=sample_query, documents=candidates, top_n=2)

    print("\nTop 2 Reranked Documents:")
    for rank, doc in enumerate(results, 1):
        print(f"  Rank {rank} ({doc.metadata['source']}): {doc.page_content}")


if __name__ == "__main__":
    main()


