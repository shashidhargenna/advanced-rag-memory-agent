from pydantic import BaseModel, Field


# =====================================================
# CHAT REQUEST
# =====================================================

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique user identifier"
    )

    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation session identifier"
    )


# =====================================================
# TELEMETRY RESPONSE
# =====================================================

class TelemetryMetrics(BaseModel):
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


# =====================================================
# CHAT RESPONSE
# =====================================================

class ChatResponse(BaseModel):
    answer: str

    user_id: str

    session_id: str

    telemetry: TelemetryMetrics


# =====================================================
# USER MEMORY RESPONSE
# =====================================================

class MemoryItem(BaseModel):
    id: str | None = None

    memory: str

    user_id: str | None = None

    categories: list[str] = Field(
        default_factory=list
    )


class UserMemoriesResponse(BaseModel):
    user_id: str

    memories: list[MemoryItem] = Field(
        default_factory=list
    )