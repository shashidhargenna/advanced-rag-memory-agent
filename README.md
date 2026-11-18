## Advanced RAG Memory Agent

An AI-powered question-answering application that uses Hybrid Search, Reranking, Context Compression, and Long-Term Memory to provide relevant and personalized answers from a knowledge base.

## Features

* Hybrid Search: Combines BM25 keyword search and Pinecone semantic search.
* Reciprocal Rank Fusion (RRF): Combines results from both search methods.
* Cohere Reranking: Reranks retrieved documents based on relevance.
* Parent-Child Retrieval: Retrieves smaller child chunks and uses their parent documents for context.
* Context Compression: Filters retrieved documents to provide relevant context to the LLM.
* Long-Term Memory: Uses Mem0 to store and retrieve user preferences across sessions.
* FastAPI: Provides REST API endpoints for chat and memory retrieval.
* Telemetry: Tracks retrieval, reranking, generation, and total pipeline latency.

## Architecture

User Query → Hybrid Search (BM25 + Pinecone) → RRF → Cohere Reranking → Parent Retrieval → Context Compression → Mem0 Memory Retrieval → LLM Generation → Response

## Tech Stack

* Python
* FastAPI
* OpenAI
* Pinecone
* Cohere
* Mem0
* LangChain
* Pytest

## Project Structure

advanced-rag-memory-agent/
├── app/
│   ├── agent/
│   ├── api/
│   ├── memory/
│   ├── retrieval/
│   └── utils/
├── data/
├── tests/
├── api_main.py
├── chat.py
├── main.py
├── api_collection.json
├── requirements.txt
└── README.md


## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

pip install -r requirements.txt

4. Configure the API keys in the .env file.

Run the Application

Document Ingestion

## python main.py

Start the API Server
## python -m uvicorn api_main:app --reload


Interactive Terminal Chat
In another terminal:
## python chat.py

You can ask questions directly through the terminal.

API Endpoints

Chat

POST /api/chat

Accepts a user query and returns an AI-generated answer with pipeline telemetry.

Retrieve User Memories

GET /api/users/{user_id}/memories

Returns the long-term memories stored for a user.

## Testing

Run the test suite:

python -m pytest

## API Documentation

When the API server is running, interactive API documentation is available at:

http://127.0.0.1:8000/docs

The Postman collection is available in api_collection.json.