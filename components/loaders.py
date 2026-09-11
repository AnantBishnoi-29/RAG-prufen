from pathlib import Path
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


def auto_detect_loader(path: str) -> list[Document]:
    """
    Automatically detects the document format from the path and dispatches
    to the appropriate loader.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")

    if path_obj.is_dir():
        return pypdf_directory_loader(str(path_obj))

    ext = path_obj.suffix.lower()
    if ext == ".pdf":
        return pypdf_loader(str(path_obj))
    elif ext in (".txt", ".md", ".json", ".csv", ".yaml", ".yml"):
        return text_loader(str(path_obj))
    else:
        # Fallback to text loader
        return text_loader(str(path_obj))


def load_documents(
    choice_or_path: int | str | None = None,
    path: str | None = None,
    choice: int | str | None = None,
) -> list[Document]:
    """
    Unified document loader supporting automatic file-type detection as well as
    explicit choice selections (by int 1-3 or string name).

    Flexible invocations:
    - load_documents("docs/sample.pdf")               -> auto-detect
    - load_documents(path="docs/sample.pdf")          -> auto-detect
    - load_documents(2, "docs/sample.pdf")            -> choice=2 (PDF)
    - load_documents(choice="pdf", path="docs/sample.pdf")
    - load_documents(choice=1, path="docs/")
    """
    actual_path = path
    actual_choice = choice

    if actual_path is None:
        if choice_or_path is not None:
            actual_path = str(choice_or_path)
        else:
            raise ValueError("A document path must be provided to load_documents.")
    else:
        if actual_choice is None and choice_or_path is not None:
            actual_choice = choice_or_path

    # If no choice is specified or 'auto' requested, auto-detect from path
    if actual_choice in (None, "auto", "0", 0):
        return auto_detect_loader(actual_path)

    choice_str = str(actual_choice).lower().strip()

    if choice_str in ("1", "directory", "dir", "folder", "pypdf_directory"):
        return pypdf_directory_loader(actual_path)
    elif choice_str in ("2", "pdf", "pypdf"):
        return pypdf_loader(actual_path)
    elif choice_str in ("3", "text", "txt", "md", "markdown"):
        return text_loader(actual_path)
    else:
        return auto_detect_loader(actual_path)


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

