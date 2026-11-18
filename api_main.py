from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router

from app.retrieval.ingestion import load_pdf_directory
from app.retrieval.chunking import (
    create_parent_chunks,
    create_child_chunks
)

from app.retrieval.hybrid_search import build_bm25_index

from app.retrieval.vector_store import connect_pinecone


# =====================================================
# APPLICATION LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\nInitializing Advanced RAG Memory Agent...")

    # Load documents
    documents = load_pdf_directory("data")

    # Create parent chunks
    parent_chunks = create_parent_chunks(documents)

    # Create child chunks
    child_chunks = create_child_chunks(parent_chunks)

    # Build BM25 index
    bm25, child_chunks = build_bm25_index(child_chunks)

    # Connect to Pinecone
    pinecone_index = connect_pinecone()

    # Store resources in application state
    app.state.bm25 = bm25
    app.state.child_chunks = child_chunks
    app.state.pinecone_index = pinecone_index

    print("\nAdvanced RAG Memory Agent initialized successfully!")

    yield

    print("\nShutting down Advanced RAG Memory Agent...")


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="Advanced RAG Memory Agent",
    description="Hybrid RAG system with reranking and semantic memory",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================
# REGISTER ROUTES
# =====================================================

app.include_router(
    router,
    prefix="/api"
)


# =====================================================
# ROOT ENDPOINT
# =====================================================

@app.get("/")
def root():

    return {
        "message": "Advanced RAG Memory Agent API is running"
    }