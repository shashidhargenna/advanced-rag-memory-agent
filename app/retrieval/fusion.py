from langchain_core.documents import Document


def reciprocal_rank_fusion(
    bm25_results: list,
    semantic_results: list,
    k: int = 60,
    top_k: int = 5
) -> list[dict]:
    """
    Combine BM25 and semantic search results
    using Reciprocal Rank Fusion.
    """

    fused_scores = {}

    documents = {}

    result_lists = [
        bm25_results,
        semantic_results
    ]

    for result_list in result_lists:

        for rank, result in enumerate(result_list):

            if isinstance(result, Document):

                child_id = result.metadata["child_id"]

                documents[child_id] = {
                    "child_id": child_id,
                    "parent_id": result.metadata["parent_id"],
                    "chunk_text": result.page_content,
                    "source": result.metadata.get("source", "unknown"),
                    "page": result.metadata.get("page", 0)
                }

            else:

                child_id = result["child_id"]

                documents[child_id] = result

            fused_scores[child_id] = (
                fused_scores.get(child_id, 0)
                + 1 / (k + rank + 1)
            )

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    final_results = []

    for child_id, score in ranked_results[:top_k]:

        result = documents[child_id].copy()

        result["rrf_score"] = score

        final_results.append(result)

    print("\n========== RECIPROCAL RANK FUSION ==========")

    print(f"BM25 results: {len(bm25_results)}")

    print(f"Semantic results: {len(semantic_results)}")

    print(f"Fused results: {len(final_results)}")

    return final_results