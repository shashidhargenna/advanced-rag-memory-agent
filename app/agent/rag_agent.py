import os
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

from app.retrieval.context_compressor import compress_context
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.parent_retriever import retrieve_parents

from app.memory.mem0_manager import (
    add_memory,
    search_memory
)


# =====================================================
# ENVIRONMENT CONFIGURATION
# =====================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing from the .env file."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY
)

MODEL_NAME = "gpt-6-luna"


# =====================================================
# STEP 1: GENERATE ANSWER USING GPT-6 LUNA
# =====================================================

def generate_answer(
    query: str,
    parent_documents: list,
    user_memories: list | None = None,
    telemetry: dict | None = None
) -> str:
    """
    Generate an answer using GPT-6 Luna.

    Uses:
    1. Retrieved parent documents.
    2. Relevant conversational memories from Mem0.

    Measures GPT generation latency.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not parent_documents:
        return (
            "I could not find relevant information "
            "in the knowledge base."
        )

    if telemetry is None:
        telemetry = {}

    # -------------------------------------------------
    # Prepare retrieved document context
    # -------------------------------------------------

    context_parts = []

    for document in parent_documents:

        source = document.metadata.get(
            "source", "Unknown"
        )

        page = document.metadata.get(
            "page", "Unknown"
        )

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # -------------------------------------------------
    # Prepare conversational memory context
    # -------------------------------------------------

    memory_context = ""

    if user_memories:

        memory_context = "\n".join(
            f"- {memory.get('memory', '')}"
            for memory in user_memories
            if memory.get("memory")
        )

    # -------------------------------------------------
    # LLM instructions
    # -------------------------------------------------

    instructions = """
    You are an AI knowledge assistant.

    Answer the user's question using the retrieved
    document context as your primary source of facts.

    Use relevant user memories to personalize the
    response when appropriate.

    Instructions:

    1. Provide accurate and relevant answers.
    2. Do not invent information.
    3. Do not use conversational memories as evidence
       for facts about the knowledge base.
    4. If the retrieved documents do not contain
       the answer, clearly say that the information
       is unavailable.
    5. Include source names and page numbers
       when supporting your answer.
    6. Adapt the explanation to relevant user
       preferences stored in memory.
    7. Keep the answer clear and well structured.

    Relevant User Memories:

    """

    instructions += (
        memory_context
        if memory_context
        else "No relevant memories found."
    )

    # -------------------------------------------------
    # Measure GPT generation latency
    # -------------------------------------------------

    generation_start = perf_counter()

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=f"""
        Retrieved Document Context:

        {context}

        User Question:

        {query}
        """
    )

    generation_latency = (
        perf_counter() - generation_start
    ) * 1000

    telemetry["generation_latency_ms"] = round(
        generation_latency,
        2
    )

    answer = response.output_text

    print("\n========== GPT-6 LUNA RESPONSE ==========")

    print(answer)

    print(
        f"\nGeneration latency: "
        f"{telemetry['generation_latency_ms']} ms"
    )

    return answer


# =====================================================
# STEP 2: COMPLETE RAG + MEM0 PIPELINE
# =====================================================

def run_rag_pipeline(
    query: str,
    bm25,
    child_chunks: list,
    index,
    user_id: str,
    run_id: str | None = None,
    top_k: int = 5,
    telemetry: dict | None = None
) -> str:
    """
    Execute the complete RAG pipeline with Mem0.

    Steps:

    1. Retrieve relevant conversational memories.
    2. Perform hybrid search.
    3. Rerank results using Cohere.
    4. Extract parent IDs.
    5. Retrieve parent documents.
    6. Compress retrieved context.
    7. Generate an answer using GPT-6 Luna.
    8. Store the conversation in Mem0.

    Measures retrieval, reranking, generation,
    and total pipeline latency.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not user_id.strip():
        raise ValueError("User ID cannot be empty.")

    if telemetry is None:
        telemetry = {}

    total_start = perf_counter()

    print(
        "\n========== RAG + MEM0 PIPELINE STARTED =========="
    )

    # -------------------------------------------------
    # STEP 1: Retrieve relevant user memories
    # -------------------------------------------------

    print("\n========== MEM0 MEMORY RETRIEVAL ==========")

    memory_response = search_memory(
        query=query,
        user_id=user_id,
        limit=5
    )

    user_memories = memory_response.get(
        "results", []
    )

    print(
        f"Relevant memories retrieved: "
        f"{len(user_memories)}"
    )

    # -------------------------------------------------
    # STEP 2: Hybrid search and Cohere reranking
    # -------------------------------------------------

    reranked_results = hybrid_search(
        query=query,
        bm25=bm25,
        child_chunks=child_chunks,
        index=index,
        top_k=top_k,
        telemetry=telemetry
    )

    if not reranked_results:

        answer = (
            "I could not find relevant information "
            "in the knowledge base."
        )

        telemetry["generation_latency_ms"] = 0.0

        telemetry["total_latency_ms"] = round(
            (perf_counter() - total_start) * 1000,
            2
        )

        return answer

    # -------------------------------------------------
    # STEP 3: Extract parent IDs
    # -------------------------------------------------

    parent_ids = [
        result["parent_id"]
        for result in reranked_results
    ]

    # -------------------------------------------------
    # STEP 4: Retrieve parent documents
    # -------------------------------------------------

    parent_documents = retrieve_parents(
        parent_ids=parent_ids
    )

    if not parent_documents:

        answer = (
            "I could not retrieve relevant "
            "parent documents."
        )

        telemetry["generation_latency_ms"] = 0.0

        telemetry["total_latency_ms"] = round(
            (perf_counter() - total_start) * 1000,
            2
        )

        return answer

    # -------------------------------------------------
    # STEP 5: Compress retrieved context using Cohere
    # -------------------------------------------------

        # -------------------------------------------------
    # STEP 5: Compress retrieved context using Cohere
    # -------------------------------------------------

    compressed_documents = compress_context(
        query=query,
        parent_documents=parent_documents,
        top_k=5
    )

    print("\n========== COMPRESSION RESULTS ==========")

    print(
        f"Parent documents before compression: "
        f"{len(parent_documents)}"
    )

    print(
        f"Documents after compression: "
        f"{len(compressed_documents)}"
    )

    if not compressed_documents:

        answer = (
            "I could not find sufficiently relevant "
            "information in the knowledge base."
        )

        telemetry["generation_latency_ms"] = 0.0

        telemetry["total_latency_ms"] = round(
            (perf_counter() - total_start) * 1000,
            2
        )

        return answer

    # -------------------------------------------------
    # STEP 6: Generate answer using GPT-6 Luna
    # -------------------------------------------------

    answer = generate_answer(
        query=query,
        parent_documents=compressed_documents,
        user_memories=user_memories,
        telemetry=telemetry
    )

    # -------------------------------------------------
    # STEP 7: Store conversation in Mem0
    # -------------------------------------------------

    print("\n========== MEM0 MEMORY STORAGE ==========")

    add_response = add_memory(
        user_message=query,
        assistant_response=answer,
        user_id=user_id,
        run_id=run_id
    )

    print("Conversation submitted to Mem0.")

    print(add_response)

    # -------------------------------------------------
    # STEP 8: Calculate total pipeline latency
    # -------------------------------------------------

    telemetry["total_latency_ms"] = round(
        (perf_counter() - total_start) * 1000,
        2
    )

    print("\n========== PIPELINE TELEMETRY ==========")

    for metric, value in telemetry.items():
        print(f"{metric}: {value} ms")

    print(
        "\n========== RAG + MEM0 PIPELINE COMPLETED =========="
    )

    return answer