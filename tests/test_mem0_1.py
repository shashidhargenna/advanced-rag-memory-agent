import time

from app.memory.mem0_manager import (
    search_memory,
    get_all_memories
)


USER_ID = "rag_test_user_001"


def test_persistent_memory():

    print("\nWaiting for Mem0 processing...")

    time.sleep(5)

    memories = get_all_memories(
        user_id=USER_ID
    )

    print("\nStored memories:")
    print(memories)

    results = search_memory(
        query="What information has the user shared previously?",
        user_id=USER_ID
    )

    print("\nRetrieved memories:")
    print(results)


if __name__ == "__main__":
    test_persistent_memory()