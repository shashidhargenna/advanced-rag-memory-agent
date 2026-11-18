"""
Interactive Terminal Chat Client

This client allows users to interact with the
Advanced RAG Memory Agent through the terminal.

It sends requests to the FastAPI backend.
"""

import json
import urllib.request
import urllib.error
import uuid


# =====================================================
# CONFIGURATION
# =====================================================

API_URL = "http://127.0.0.1:8000/api/chat"


# =====================================================
# SEND MESSAGE TO FASTAPI
# =====================================================

def send_message(
    query: str,
    user_id: str,
    session_id: str
):
    """
    Send a user query to the FastAPI backend.
    """

    payload = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id
    }

    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=request_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_data = json.loads(
                response.read().decode("utf-8")
            )

            return response_data

    except urllib.error.HTTPError as error:

        error_response = error.read().decode("utf-8")

        print("\nAPI Error:")
        print(error_response)

        return None

    except urllib.error.URLError as error:

        print("\nUnable to connect to FastAPI.")

        print(
            "Make sure the API server is running "
            "in another terminal."
        )

        print(f"Error: {error}")

        return None

    except Exception as error:

        print("\nUnexpected error occurred.")

        print(f"Error: {error}")

        return None


# =====================================================
# DISPLAY TELEMETRY
# =====================================================

def display_telemetry(
    telemetry: dict
):
    """
    Display pipeline latency metrics.
    """

    print("\n" + "=" * 45)

    print("PIPELINE TELEMETRY")

    print("=" * 45)

    print(
        f"Retrieval latency: "
        f"{telemetry.get('retrieval_latency_ms', 0)} ms"
    )

    print(
        f"Reranking latency: "
        f"{telemetry.get('rerank_latency_ms', 0)} ms"
    )

    print(
        f"Generation latency: "
        f"{telemetry.get('generation_latency_ms', 0)} ms"
    )

    print(
        f"Total latency: "
        f"{telemetry.get('total_latency_ms', 0)} ms"
    )

    print("=" * 45)


# =====================================================
# INTERACTIVE CHAT
# =====================================================

def start_chat():

    print("\n" + "=" * 50)

    print("ADVANCED RAG MEMORY AGENT")

    print("=" * 50)

    # -----------------------------------------------
    # USER ID
    # -----------------------------------------------

    user_id = input(
        "\nEnter your User ID "
        "(press Enter to use usr_99): "
    ).strip()

    if not user_id:

        user_id = "usr_99"

    # -----------------------------------------------
    # SESSION ID
    # -----------------------------------------------

    session_id = (
        "session_"
        + uuid.uuid4().hex[:8]
    )

    print("\nChat session initialized.")

    print(f"User ID: {user_id}")

    print(f"Session ID: {session_id}")

    print("\nType 'exit' or 'quit' to end the chat.")

    print("=" * 50)

    # -----------------------------------------------
    # CHAT LOOP
    # -----------------------------------------------

    while True:

        query = input("\nYou: ").strip()

        # -------------------------------------------
        # EXIT CONDITION
        # -------------------------------------------

        if query.lower() in ["exit", "quit"]:

            print("\nEnding chat session.")

            print("Goodbye!")

            break

        # -------------------------------------------
        # EMPTY INPUT
        # -------------------------------------------

        if not query:

            print("Please enter a question.")

            continue

        # -------------------------------------------
        # SEND REQUEST
        # -------------------------------------------

        print("\nThinking...")

        response = send_message(
            query=query,
            user_id=user_id,
            session_id=session_id
        )

        # -------------------------------------------
        # HANDLE RESPONSE
        # -------------------------------------------

        if response is None:

            continue

        # -------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------

        print("\n" + "=" * 50)

        print("AI RESPONSE")

        print("=" * 50)

        print(response.get("answer", "No answer received."))

        # -------------------------------------------
        # DISPLAY TELEMETRY
        # -------------------------------------------

        telemetry = response.get(
            "telemetry",
            {}
        )

        display_telemetry(telemetry)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    start_chat()