from app.memory.mem0_manager import (
    add_memory,
    search_memory,
    get_all_memories,
    delete_memory
)


USER_ID = "rag_test_user_001"


def test_mem0_operations():

    # Step 1: Add a test memory
    print("\n========== ADD MEMORY ==========")

    add_response = add_memory(
        user_message=(
            "I am learning LangGraph "
            "and prefer Python examples."
        ),
        assistant_response=(
            "I will use Python examples "
            "when explaining technical concepts."
        ),
        user_id=USER_ID,
        run_id="test_session_001",
        ttl_days=7
    )

    print(add_response)


    # Step 2: Search memories
    print("\n========== SEARCH MEMORY ==========")

    search_response = search_memory(
        query="What programming examples does the user prefer?",
        user_id=USER_ID
    )

    print(search_response)


    # Step 3: Retrieve all memories
    print("\n========== GET ALL MEMORIES ==========")

    all_memories = get_all_memories(
        user_id=USER_ID
    )

    print(all_memories)


 # Step 4: Delete the test memory
    print("\n========== DELETE MEMORY ==========")

    memory_results = all_memories.get("results", [])

    if memory_results:

        memory_id = memory_results[0]["id"]

        delete_response = delete_memory(
            memory_id=memory_id
        )

        print(delete_response)

    else:
        print("No memories available to delete.")

# =====================================================
# PERSONA ADAPTATION TEST
# =====================================================

def test_store_persona_preferences():

    user_id = "usr_99"

    user_message = (
        "Hello! I work in Python and prefer "
        "brief, code-only responses."
    )

    assistant_response = (
        "Understood. I can tailor future responses "
        "to your preferences."
    )

    response = add_memory(
        user_message=user_message,
        assistant_response=assistant_response,
        user_id=user_id,
        run_id="session_1"
    )

    print("\n========== PERSONA MEMORY STORAGE ==========")
    print(response)

def test_retrieve_persona_preferences():

    user_id = "usr_99"

    response = search_memory(
        query="What programming language does the user prefer, and how should responses be formatted?",
        user_id=user_id,
        limit=5
    )

    print("\n========== PERSONA MEMORY RETRIEVAL ==========")

    print(response)

    results = response.get("results", [])

    assert results, "No persona memories were retrieved."

    print("\nPersona preference successfully retrieved!")

if __name__ == "__main__":
    test_mem0_operations()
    
    
   