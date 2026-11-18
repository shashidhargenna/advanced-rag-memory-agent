import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from mem0 import MemoryClient


load_dotenv()


# Initialize Mem0 client
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

if not MEM0_API_KEY:
    raise ValueError(
        "MEM0_API_KEY is missing from the .env file."
    )


memory_client = MemoryClient(
    api_key=MEM0_API_KEY
)

def add_memory(
    user_message: str,
    assistant_response: str,
    user_id: str,
    run_id: str | None = None,
    ttl_days: int | None = 30
):
    """
    Extract and store useful conversational information
    with optional memory expiration.
    """

    if not user_message.strip():
        raise ValueError("User message cannot be empty.")

    if not user_id.strip():
        raise ValueError("User ID cannot be empty.")

    if ttl_days is not None and ttl_days < 1:
        raise ValueError("ttl_days must be at least 1.")

    messages = [
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]

    expiration_date = None

    if ttl_days is not None:
        expiration_date = (
            datetime.now(timezone.utc)
            + timedelta(days=ttl_days)
        ).date().isoformat()

    response = memory_client.add(
        messages=messages,
        user_id=user_id,
        agent_id="advanced-rag-agent",
        run_id=run_id,
        expiration_date=expiration_date
    )

    print("\n========== MEM0 MEMORY STORAGE ==========")
    print(f"User ID: {user_id}")
    print(f"Expiration date: {expiration_date or 'Never'}")

    return response

def search_memory(
    query: str,
    user_id: str,
    limit: int = 5
):
    """
    Search for relevant conversational memories
    belonging to a specific user.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not user_id.strip():
        raise ValueError("User ID cannot be empty.")

    response = memory_client.search(
        query=query,
        filters={
            "user_id": user_id
        },
        top_k=limit
    )

    print("\n========== MEM0 MEMORY SEARCH ==========")

    print(f"User ID: {user_id}")
    print(f"Search query: {query}")

    return response

def get_all_memories(
    user_id: str,
    page: int = 1,
    page_size: int = 100
):
    """
    Retrieve all stored memories belonging to a user.
    """

    if not user_id.strip():
        raise ValueError("User ID cannot be empty.")

    response = memory_client.get_all(
        filters={
            "user_id": user_id
        },
        page=page,
        page_size=page_size
    )

    print("\n========== ALL USER MEMORIES ==========")

    print(f"User ID: {user_id}")
    print(f"Total memories: {response.get('count', 0)}")

    for memory in response.get("results", []):

        print(
            f"- {memory.get('memory')}"
        )

    return response

def delete_memory(memory_id: str):
    """
    Permanently delete a specific memory using its ID.
    """

    if not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")

    response = memory_client.delete(
        memory_id=memory_id
    )

    print("\n========== MEM0 MEMORY DELETION ==========")

    print(f"Deleted memory ID: {memory_id}")

    return response    


