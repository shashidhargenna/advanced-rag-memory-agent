import os
import cohere


def rerank_results(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    min_relevance_score: float = 0.30
) -> list[dict]:

    """
    Rerank hybrid search results using Cohere.

    Discard documents below the minimum
    relevance-score threshold.
    """

    if not documents:
        return []

    if not 0 <= min_relevance_score <= 1:

        raise ValueError(
            "Minimum relevance score must be between 0 and 1."
        )

    co = cohere.Client(
        api_key=os.getenv("COHERE_API_KEY")
    )

    texts = [
        document["chunk_text"]
        for document in documents
    ]

    response = co.rerank(
        query=query,
        documents=texts,
        top_n=len(texts),
        model="rerank-v3.5"
    )

    reranked_results = []

    print("\n========== COHERE RERANKING SCORES ==========")

    for result in response.results:

        document = documents[result.index].copy()

        document["rerank_score"] = result.relevance_score

        # Display every document's score
        print(
            f"Score: {result.relevance_score:.4f} "
            f"| Source: {document.get('source', 'unknown')}"
        )

        # Apply minimum relevance threshold
        if result.relevance_score < min_relevance_score:

            continue

        reranked_results.append(document)

    # Keep only the highest-scoring results
    reranked_results = reranked_results[:top_k]

    print("\n========== COHERE RERANKING SUMMARY ==========")

    print(f"Input documents: {len(documents)}")

    print(f"Minimum relevance score: {min_relevance_score}")

    print(f"Documents after filtering: {len(reranked_results)}")

    return reranked_results