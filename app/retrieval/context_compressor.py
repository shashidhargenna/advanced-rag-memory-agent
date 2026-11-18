import os

import cohere

from langchain_core.documents import Document


def compress_context(
    query: str,
    parent_documents: list[Document],
    top_k: int = 5
) -> list[Document]:
    """
    Compress retrieved context by selecting the most
    relevant parent documents using Cohere Rerank.

    The first reranking stage has already filtered the
    child chunks.

    Therefore, this stage only ranks the retrieved
    parent documents and selects the top_k documents.

    It does not apply another relevance threshold
    that could eliminate all retrieved context.
    """

    # -------------------------------------------------
    # STEP 1: Validate input
    # -------------------------------------------------

    if not parent_documents:
        return []

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    # -------------------------------------------------
    # STEP 2: Initialize Cohere
    # -------------------------------------------------

    co = cohere.Client(
        api_key=os.getenv("COHERE_API_KEY")
    )

    # -------------------------------------------------
    # STEP 3: Prepare documents for reranking
    # -------------------------------------------------

    documents = [
        document.page_content
        for document in parent_documents
    ]

    # -------------------------------------------------
    # STEP 4: Rerank parent documents
    # -------------------------------------------------

    response = co.rerank(
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
        model="rerank-v3.5"
    )

    # -------------------------------------------------
    # STEP 5: Select top-ranked parent documents
    # -------------------------------------------------

    compressed_documents = []

    for result in response.results:

        document = parent_documents[result.index]

        compressed_document = Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "compression_score": result.relevance_score
            }
        )

        compressed_documents.append(
            compressed_document
        )

    # -------------------------------------------------
    # STEP 6: Logging
    # -------------------------------------------------

    print(
        "\n========== COHERE CONTEXT COMPRESSION =========="
    )

    print(
        f"Parent documents received: "
        f"{len(parent_documents)}"
    )

    print(
        f"Documents retained: "
        f"{len(compressed_documents)}"
    )

    print(
        f"Top K: {top_k}"
    )

    print(
        "Relevance threshold: None"
    )

    return compressed_documents