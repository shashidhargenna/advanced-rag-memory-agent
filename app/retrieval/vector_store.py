from pinecone import Pinecone

from app.utils.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


def connect_pinecone():
    """
    Connect to the existing Pinecone index.
    """

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(PINECONE_INDEX_NAME)

    print("\n========== PINECONE CONNECTION ==========")

    print("Connected successfully!")

    print(f"Index name: {PINECONE_INDEX_NAME}")

    return index


def upsert_child_chunks(
    index,
    child_chunks,
    namespace="technova-knowledge-base"
):
    """
    Upload child chunks to Pinecone.

    Pinecone's integrated embedding model
    generates embeddings from the chunk text.
    """

    records = []

    for child in child_chunks:

        metadata = child.metadata

        record = {
            "_id": metadata["child_id"],

            "chunk_text": child.page_content,

            "parent_id": metadata["parent_id"],

            "child_id": metadata["child_id"],

            "source": str(
                metadata.get("source", "unknown")
            ),

            "page": int(
                metadata.get("page", 0)
            )
        }

        records.append(record)

    if not records:
        raise ValueError(
            "No child chunks available for indexing."
        )

    print("\n========== PINECONE INDEXING ==========")

    print(f"Total child chunks: {len(records)}")

    batch_size = 50

    total_uploaded = 0

    for start in range(0, len(records), batch_size):

        batch = records[start:start + batch_size]

        response = index.documents.upsert(
            namespace=namespace,
            documents=batch
        )

        total_uploaded += response.upserted_count

        print(
            f"Uploaded batch: "
            f"{total_uploaded}/{len(records)}"
        )

    print("\n========== INDEXING COMPLETE ==========")

    print(f"Total uploaded: {total_uploaded}")

    print(f"Namespace: {namespace}")

    return total_uploaded


def get_index_stats(index):
    """
    Retrieve Pinecone index statistics.
    """

    stats = index.describe_index_stats()

    print("\n========== INDEX STATISTICS ==========")

    print(stats)

    return stats