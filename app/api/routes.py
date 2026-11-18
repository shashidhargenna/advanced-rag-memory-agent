from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    TelemetryMetrics,
    MemoryItem,
    UserMemoriesResponse
)

from app.agent.rag_agent import run_rag_pipeline

from app.memory.mem0_manager import get_all_memories


router = APIRouter()


# =====================================================
# POST /api/chat
# =====================================================

@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    request: Request,
    chat_request: ChatRequest
):
    """
    Execute the RAG pipeline and return
    the generated answer with telemetry.
    """

    try:

        # Retrieve initialized resources
        bm25 = request.app.state.bm25

        child_chunks = request.app.state.child_chunks

        index = request.app.state.pinecone_index

        # Initialize telemetry dictionary
        telemetry = {}

        # Execute RAG pipeline
        answer = run_rag_pipeline(
            query=chat_request.query,
            bm25=bm25,
            child_chunks=child_chunks,
            index=index,
            user_id=chat_request.user_id,
            run_id=chat_request.session_id,
            top_k=5,
            telemetry=telemetry
        )

        return ChatResponse(
            answer=answer,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            telemetry=TelemetryMetrics(
                retrieval_latency_ms=telemetry.get(
                    "retrieval_latency_ms", 0
                ),
                rerank_latency_ms=telemetry.get(
                    "rerank_latency_ms", 0
                ),
                generation_latency_ms=telemetry.get(
                    "generation_latency_ms", 0
                ),
                total_latency_ms=telemetry.get(
                    "total_latency_ms", 0
                )
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =====================================================
# GET /api/users/{user_id}/memories
# =====================================================

@router.get(
    "/users/{user_id}/memories",
    response_model=UserMemoriesResponse
)
def get_user_memories(user_id: str):

    """
    Retrieve all semantic memories belonging
    to a specific user.
    """

    try:

        response = get_all_memories(
            user_id=user_id
        )

        results = response.get("results", [])

        memories = []

        for item in results:

            memories.append(
                MemoryItem(
                    id=item.get("id"),
                    memory=item.get("memory", ""),
                    user_id=item.get("user_id"),
                    categories=item.get("categories", [])
                )
            )

        return UserMemoriesResponse(
            user_id=user_id,
            memories=memories
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )