from app.retrieval.ingestion import load_pdf_directory
from langchain_core.documents import Document
from app.agent.rag_agent import generate_answer
from app.memory.mem0_manager import search_memory
from app.retrieval.chunking import (
    create_parent_chunks,
    create_child_chunks
)

from app.retrieval.parent_retriever import save_parent_chunks

from app.retrieval.hybrid_search import build_bm25_index

from app.retrieval.vector_store import connect_pinecone

from app.agent.rag_agent import run_rag_pipeline


def test_complete_rag_pipeline():

    # Step 1: Load PDF documents
    documents = load_pdf_directory("data")

    # Step 2: Create parent chunks
    parent_chunks = create_parent_chunks(documents)

    # Step 3: Create child chunks
    child_chunks = create_child_chunks(parent_chunks)

    # Step 4: Save parent chunks
    save_parent_chunks(parent_chunks)

    # Step 5: Build BM25 index
    bm25, child_chunks = build_bm25_index(child_chunks)

    # Step 6: Connect to Pinecone
    index = connect_pinecone()

    # Step 7: Run complete RAG pipeline
    query = "What is the company's leave policy?"

    answer = run_rag_pipeline(
    query=query,
    bm25=bm25,
    child_chunks=child_chunks,
    index=index,
    user_id="rag_test_user_001",
    run_id="rag_session_001",
    top_k=5
)

    # Step 8: Display final answer
    print("\n========== FINAL RAG ANSWER ==========")

    print(answer)

def test_persona_adaptation_new_session():

    user_id = "usr_99"
    run_id = "session_2"

    # Retrieve preferences saved in session 1
    memory_response = search_memory(
        query="What programming language does the user prefer, and how should responses be formatted?",
        user_id=user_id,
        limit=5
    )

    user_memories = memory_response.get("results", [])

    assert user_memories, "No user preferences retrieved."

    # Provide relevant knowledge context
    parent_documents = [
        Document(
            page_content=(
                "Merge sort is a divide-and-conquer sorting "
                "algorithm. It divides an array into smaller "
                "subarrays, sorts them recursively, and merges "
                "the sorted subarrays."
            ),
            metadata={
                "source": "algorithm_guide",
                "page": 1
            }
        )
    ]

    # Generate response using the retrieved preferences
    answer = generate_answer(
        query="Write a quick merge sort routine.",
        parent_documents=parent_documents,
        user_memories=user_memories
    )

    print("\n========== NEW SESSION PERSONA TEST ==========")
    print("User ID:", user_id)
    print("Run ID:", run_id)
    print("Retrieved preferences:", user_memories)
    print("Generated answer:", answer)

    assert "def " in answer, "Expected a Python function."

if __name__ == "__main__":
    test_complete_rag_pipeline()