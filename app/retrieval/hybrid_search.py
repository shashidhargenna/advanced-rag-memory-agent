"""
Hybrid Retrieval Module

This module handles:

1. BM25 keyword retrieval.
2. Pinecone semantic retrieval.
3. Reciprocal Rank Fusion (RRF).
4. Cohere reranking.
5. Retrieval and reranking latency measurement.
"""

import re
from time import perf_counter

from pinecone import EmbedQuery
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_results


# =====================================================
# STEP 1: TEXT TOKENIZATION
# =====================================================

def tokenize_text(text: str) -> list[str]:
    """
    Convert text into lowercase tokens.
    """

    return re.findall(r"\b\w+\b", text.lower())


# =====================================================
# STEP 2: BUILD BM25 INDEX
# =====================================================

def build_bm25_index(
    child_chunks: list[Document]
):
    """
    Build a BM25 index from child chunks.

    Args:
        child_chunks: List of child documents.

    Returns:
        BM25 index and corresponding child documents.
    """

    if not child_chunks:
        raise ValueError(
            "No child chunks available for BM25 indexing."
        )

    tokenized_corpus = [
        tokenize_text(chunk.page_content)
        for chunk in child_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("\n========== BM25 INDEXING ==========")

    print(f"Total documents indexed: {len(child_chunks)}")

    return bm25, child_chunks


# =====================================================
# STEP 3: BM25 KEYWORD SEARCH
# =====================================================

def bm25_search(
    query: str,
    bm25,
    child_chunks: list[Document],
    top_k: int = 5
) -> list[Document]:
    """
    Retrieve relevant child chunks using BM25.
    """

    tokenized_query = tokenize_text(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        if scores[index] <= 0:
            continue

        results.append(child_chunks[index])

        if len(results) == top_k:
            break

    print("\n========== BM25 SEARCH ==========")

    print(f"Query: {query}")

    print(f"Results retrieved: {len(results)}")

    return results


# =====================================================
# STEP 4: PINECONE SEMANTIC SEARCH
# =====================================================

def pinecone_search(
    query: str,
    index,
    top_k: int = 5,
    namespace: str = "technova-knowledge-base"
) -> list[dict]:
    """
    Retrieve relevant child chunks using
    Pinecone's integrated semantic search.

    Uses the configured embedding model:
    text-embedding-3-small.
    """

    response = index.documents.search(
        namespace=namespace,

        top_k=top_k,

        score_by=[
            EmbedQuery(
                field="chunk_text",
                query=query
            )
        ],

        include_fields=[
            "chunk_text",
            "parent_id",
            "child_id",
            "source",
            "page"
        ]
    )

    results = []

    for match in response.matches:

        results.append({

            "child_id": match.id,

            "parent_id": match.get("parent_id"),

            "chunk_text": match.get("chunk_text", ""),

            "source": match.get("source", "unknown"),

            "page": match.get("page", 0),

            "score": match.score

        })

    print("\n========== PINECONE SEMANTIC SEARCH ==========")

    print(f"Query: {query}")

    print(f"Results retrieved: {len(results)}")

    return results


# =====================================================
# STEP 5: HYBRID SEARCH + RECIPROCAL RANK FUSION
# =====================================================

def hybrid_search(
    query: str,
    bm25,
    child_chunks: list[Document],
    index,
    top_k: int = 5,
    telemetry: dict | None = None
) -> list[dict]:
    """
    Perform hybrid retrieval using:

    1. BM25 keyword search.
    2. Pinecone semantic search.
    3. Reciprocal Rank Fusion.
    4. Cohere reranking.

    Measures:

    - Retrieval latency.
    - Reranking latency.

    Args:
        query: User's search query.
        bm25: BM25 index.
        child_chunks: Indexed child documents.
        index: Pinecone index.
        top_k: Number of final results.
        telemetry: Optional dictionary for latency metrics.

    Returns:
        Final reranked search results.
    """

    if telemetry is None:
        telemetry = {}

    # -----------------------------------------------
    # STEP 1: Measure retrieval latency
    # -----------------------------------------------

    retrieval_start = perf_counter()

    # BM25 keyword retrieval
    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        child_chunks=child_chunks,
        top_k=20
    )

    # Pinecone semantic retrieval
    semantic_results = pinecone_search(
        query=query,
        index=index,
        top_k=20
    )

    # Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        semantic_results=semantic_results,
        top_k=20
    )

    retrieval_latency = (
        perf_counter() - retrieval_start
    ) * 1000

    telemetry["retrieval_latency_ms"] = round(
        retrieval_latency,
        2
    )

    # -----------------------------------------------
    # STEP 2: Measure Cohere reranking latency
    # -----------------------------------------------

    rerank_start = perf_counter()

    reranked_results = rerank_results(
        query=query,
        documents=fused_results,
        top_k=top_k
    )

    rerank_latency = (
        perf_counter() - rerank_start
    ) * 1000

    telemetry["rerank_latency_ms"] = round(
        rerank_latency,
        2
    )

    # -----------------------------------------------
    # STEP 3: Display retrieval results
    # -----------------------------------------------

    print("\n========== FINAL HYBRID RETRIEVAL ==========")

    print(
        f"Final results returned: "
        f"{len(reranked_results)}"
    )

    print(
        f"Retrieval latency: "
        f"{telemetry['retrieval_latency_ms']} ms"
    )

    print(
        f"Reranking latency: "
        f"{telemetry['rerank_latency_ms']} ms"
    )

    return reranked_results