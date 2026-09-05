from typing import Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def get_recursive_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    **kwargs,
) -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter.
    Recursively splits by delimiters (paragraph, newline, space) to keep context coherent.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )


def get_character_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separator: str = "\n\n",
    **kwargs,
) -> CharacterTextSplitter:
    """
    Returns a CharacterTextSplitter.
    Splits strictly based on a specified delimiter string.
    """
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        **kwargs,
    )


def get_token_splitter(
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    **kwargs,
) -> TokenTextSplitter:
    """
    Returns a TokenTextSplitter.
    Splits text directly by BPE token count.
    """
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )


def split_documents(
    documents: list[Document],
    splitter_type: Literal["recursive", "character", "token"] = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    **kwargs,
) -> list[Document]:
    """
    Splits a list of Documents into chunks using the specified splitter configuration.

    Parameters:
    - documents: List of LangChain Document objects (e.g. from loaders)
    - splitter_type: 'recursive', 'character', or 'token'
    - chunk_size: Target size of each chunk (characters for recursive/character, tokens for token)
    - chunk_overlap: Overlap between consecutive chunks
    """
    splitter_type_normalized = splitter_type.lower().strip()

    if splitter_type_normalized in ("recursive", "recursive_character", "1"):
        splitter = get_recursive_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    elif splitter_type_normalized in ("character", "char", "2"):
        splitter = get_character_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    elif splitter_type_normalized in ("token", "tokens", "3"):
        splitter = get_token_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    else:
        raise ValueError(
            f"Unknown splitter_type: '{splitter_type}'. Supported: 'recursive', 'character', 'token'"
        )

    return splitter.split_documents(documents)


def main():
    print("""Text Splitter Test:
    1. RecursiveCharacterTextSplitter (Recommended for general text/PDFs)
    2. CharacterTextSplitter (Splits on delimiter)
    3. TokenTextSplitter (Splits by raw token counts)
    """)
    sample_text = (
        "Retrieval-Augmented Generation (RAG) combines search retrieval with generative AI. "
        "By grounding LLM outputs in verified external knowledge, RAG drastically reduces hallucinations. "
        "Choosing the right chunk size and overlap is critical to ensure the retriever provides adequate context."
    )
    sample_doc = [Document(page_content=sample_text, metadata={"source": "sample"})]

    choice = input("Enter splitter choice [1-3] (default: 1): ").strip() or "1"
    chunk_size = int(input("Enter chunk_size (default: 100): ").strip() or "100")
    chunk_overlap = int(input("Enter chunk_overlap (default: 20): ").strip() or "20")

    chunks = split_documents(
        documents=sample_doc,
        splitter_type=choice,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"\nGenerated {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] ({len(chunk.page_content)} chars): {chunk.page_content}")


if __name__ == "__main__":
    main()