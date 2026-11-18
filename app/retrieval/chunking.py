from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Parent chunk configuration
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)


# Child chunk configuration
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)


def create_parent_chunks(
    documents: list[Document]
) -> list[Document]:
    """
    Split extracted PDF documents into larger parent chunks.
    """

    parent_chunks = parent_splitter.split_documents(documents)

    for index, chunk in enumerate(parent_chunks):
        chunk.metadata["parent_id"] = f"parent_{index}"

    return parent_chunks


def create_child_chunks(
    parent_chunks: list[Document]
) -> list[Document]:
    """
    Split parent chunks into smaller child chunks.
    """

    child_chunks = []

    for parent in parent_chunks:

        parent_id = parent.metadata["parent_id"]

        chunks = child_splitter.split_documents([parent])

        for index, child in enumerate(chunks):

            child.metadata["parent_id"] = parent_id

            child.metadata["child_id"] = (
                f"{parent_id}_child_{index}"
            )

            child_chunks.append(child)

    return child_chunks