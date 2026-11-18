from app.retrieval.ingestion import load_pdf_directory

from app.retrieval.chunking import (
    create_parent_chunks,
    create_child_chunks
)

from app.retrieval.vector_store import (
    connect_pinecone
)

from app.retrieval.hybrid_search import (
    build_bm25_index,
    hybrid_search
)


if __name__ == "__main__":

    # Step 1: Load PDFs
    documents = load_pdf_directory("data")

    # Step 2: Create parent chunks
    parent_chunks = create_parent_chunks(documents)

    # Step 3: Create child chunks
    child_chunks = create_child_chunks(parent_chunks)

    # Step 4: Connect to Pinecone
    index = connect_pinecone()

    # Step 5: Build BM25 index
    bm25, child_chunks = build_bm25_index(child_chunks)

    # Step 6: Enter query
    query = "What is the company's leave policy?"

    # Step 7: Perform hybrid search
    results = hybrid_search(
        query=query,
        bm25=bm25,
        child_chunks=child_chunks,
        index=index,
        top_k=5
    )

    # Step 8: Display results
    print("\n========== FINAL SEARCH RESULTS ==========")

    for result in results:

     print("\nChild ID:", result["child_id"])

    print("Parent ID:", result["parent_id"])

    print("Source:", result["source"])

    print("RRF Score:", result["rrf_score"])

    print("Rerank Score:", result["rerank_score"])

    print("Content:", result["chunk_text"][:300])

    print("-" * 60)
        