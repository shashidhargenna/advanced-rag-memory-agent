from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_directory(data_dir: str | Path) -> list[Document]:
    """
    Load all PDF files from the specified directory.

    Args:
        data_dir: Directory containing PDF documents.

    Returns:
        A list of LangChain Document objects.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}"
        )

    if not data_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {data_path}"
        )

    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_path}"
        )

    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))

        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name

        documents.extend(pages)

        print(f"Loaded {pdf_file.name}: {len(pages)} pages")

    print(f"\nTotal pages loaded: {len(documents)}")

    return documents