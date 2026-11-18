import os

from dotenv import load_dotenv


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "advanced-rag-memory"
)

MEM0_API_KEY = os.getenv("MEM0_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY is missing from .env")