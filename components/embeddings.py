from typing import Literal
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def openai_embeddings(
    model_name: str = "text-embedding-3-small",
    **kwargs,
) -> OpenAIEmbeddings:
    """Returns OpenAI embedding model (requires OPENAI_API_KEY)."""
    return OpenAIEmbeddings(model=model_name, **kwargs)


def huggingface_local_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    **kwargs,
) -> HuggingFaceEmbeddings:
    """Returns local HuggingFace embedding model running on CPU/GPU."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        **kwargs,
    )


def huggingface_cloud_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs,
) -> HuggingFaceEndpointEmbeddings:
    """Returns HuggingFace serverless inference endpoint embedding model."""
    return HuggingFaceEndpointEmbeddings(model=model_name, **kwargs)


def get_embeddings(
    provider: Literal["openai", "huggingface_local", "huggingface_cloud"] = "openai",
    model_name: str | None = None,
    **kwargs,
) -> Embeddings:
    """
    Unified router to instantiate and return an embedding model.
    """
    provider_normalized = provider.lower().strip()

    if provider_normalized in ("openai", "1"):
        model = model_name or "text-embedding-3-small"
        return openai_embeddings(model_name=model, **kwargs)

    elif provider_normalized in ("huggingface_local", "local", "hf_local", "2"):
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return huggingface_local_embeddings(model_name=model, **kwargs)

    elif provider_normalized in ("huggingface_cloud", "cloud", "hf_cloud", "3"):
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return huggingface_cloud_embeddings(model_name=model, **kwargs)

    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Supported: 'openai', 'huggingface_local', 'huggingface_cloud'"
        )


def main():
    print("""Embedding Model Test:
    1. OpenAI (text-embedding-3-small)
    2. HuggingFace Local (all-MiniLM-L6-v2)
    3. HuggingFace Cloud Endpoint
    """)
    choice = input("Enter choice [1-3] (default: 1): ").strip() or "1"
    embeddings = get_embeddings(provider=choice)

    sample_query = "What is Retrieval-Augmented Generation?"
    vector = embeddings.embed_query(sample_query)
    print(f"\nGenerated vector with {len(vector)} dimensions.")
    print(f"Sample values (first 3): {vector[:3]}")


if __name__ == "__main__":
    main()
