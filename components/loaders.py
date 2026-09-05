from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document


def pypdf_directory_loader(path: str) -> list[Document]:
    """Loads all PDFs from a given directory."""
    loader = PyPDFDirectoryLoader(path=path)
    return loader.load()


def pypdf_loader(path: str) -> list[Document]:
    """Loads a single PDF file."""
    loader = PyPDFLoader(file_path=path)
    return loader.load()


def text_loader(path: str, encoding: str = "utf-8") -> list[Document]:
    """Loads a text or markdown file."""
    loader = TextLoader(file_path=path, encoding=encoding)
    return loader.load()


def load_documents(choice: int, path: str) -> list[Document]:
    """Dispatches loading based on user selection."""
    if choice == 1:
        return pypdf_directory_loader(path)
    elif choice == 2:
        return pypdf_loader(path)
    elif choice == 3:
        return text_loader(path)
    else:
        raise ValueError(f"Invalid choice: {choice}. Must be 1, 2, or 3.")


def main():
    print("""Available Loaders:
    1. PyPDFDirectoryLoader (Loads folder of PDFs)
    2. PyPDFLoader (Loads single PDF)
    3. TextLoader (Loads .txt / .md file)
    """)
    user_choice = int(input("Enter your choice of loader (1-3): "))
    path = input("Enter the file or directory path: ")

    docs = load_documents(user_choice, path)
    print(f"\nSuccessfully loaded {len(docs)} document chunk(s)/page(s).")


if __name__ == "__main__":
    main()

