from typing import List, Sequence, Union
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "databases" / "chroma"

# def create_vector_store(
#     data: list[Document],
#     embedding_model: OpenAIEmbeddings | None = None,
#     collection_name: str = "rag_collection",
# ) -> Chroma:
#     """
#     Creates an in-memory Chroma vector store from raw strings or LangChain Documents.
#     """
#     if embedding_model is None:
#         embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#     documents = data

#     vector_store = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         collection_name=collection_name,
#     )
#     return vector_store


def get_retriever(vector_store: Chroma, k: int = 3):
    """
    Returns a standard retriever with top-k similarity search.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})



def retrieve_contexts(retriever, query: str) -> List[str]:
    """
    Retrieves matching documents and returns their text contents as a list of strings.
    """
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]



def pdf_directory_loader(path):
    """
    Loads all the pdfs from a directory
    """
    loader = PyPDFDirectoryLoader(path=path)

    docu = loader.load()
    return docu

def pdf_loader(path):
    """
    This function loads a single pdf when given path
    """

    loader = PyPDFLoader(file_path=path)

    docu = loader.load()

    return docu


def create_vector_store(path):
    """
    creates a vector store from the path of the pdf given
    """

    docu = pdf_loader(path)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    chunks = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap= 20).split_documents(docu)

    return Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory=str(DB_DIR))