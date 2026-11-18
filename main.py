from app.retrieval.ingestion import load_pdf_directory

from app.retrieval.chunking import (
    create_parent_chunks,
    create_child_chunks
)

from app.retrieval.parent_retriever import (
    save_parent_chunks,
    retrieve_parent
)

from app.retrieval.vector_store import (
    connect_pinecone,
    upsert_child_chunks
)


if __name__ == "__main__":

    # Step 1: Load PDF documents
    documents = load_pdf_directory("data")

    # Step 2: Create parent chunks
    parent_chunks = create_parent_chunks(documents)

    # Step 3: Create child chunks
    child_chunks = create_child_chunks(parent_chunks)

    # Step 4: Save parent chunks
    save_parent_chunks(parent_chunks)

    # Step 5: Select a sample child
    sample_child = child_chunks[0]

    parent_id = sample_child.metadata["parent_id"]

    print("\n========== CHILD DOCUMENT ==========")

    print("Child ID:", sample_child.metadata["child_id"])
    print("Parent ID:", parent_id)

    # Step 6: Retrieve corresponding parent
    retrieved_parent = retrieve_parent(parent_id)

    print("\n========== RETRIEVED PARENT ==========")

    print("Parent ID:", retrieved_parent.metadata["parent_id"])

    print("\nParent Content:")
    print(retrieved_parent.page_content[:500])

    # Step 7: Connect to Pinecone
    index = connect_pinecone()

    # Step 8: Upload child chunks
    uploaded_count = upsert_child_chunks(
        index=index,
        child_chunks=child_chunks
    )

    print("\n========== FINAL SUMMARY ==========")

    print("PDF pages loaded:", len(documents))
    print("Parent chunks created:", len(parent_chunks))
    print("Child chunks created:", len(child_chunks))
    print("Child chunks uploaded:", uploaded_count)