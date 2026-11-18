import json

from pathlib import Path

from langchain_core.documents import Document


DEFAULT_STORE_PATH = Path("data/parent_store.json")


def save_parent_chunks(
    parent_chunks: list[Document],
    storage_path: str | Path = DEFAULT_STORE_PATH
) -> None:
    """
    Save parent chunks into a persistent JSON document store.
    """

    storage_path = Path(storage_path)

    storage_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    parent_store = {}

    for parent in parent_chunks:

        parent_id = parent.metadata.get("parent_id")

        if not parent_id:
            raise ValueError(
                "Parent document is missing parent_id"
            )

        if parent_id in parent_store:
            raise ValueError(
                f"Duplicate parent ID found: {parent_id}"
            )

        parent_store[parent_id] = {
            "page_content": parent.page_content,
            "metadata": parent.metadata
        }

    with storage_path.open("w", encoding="utf-8") as file:

        json.dump(
            parent_store,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Saved {len(parent_store)} parent chunks "
        f"to {storage_path}"
    )


def load_parent_store(
    storage_path: str | Path = DEFAULT_STORE_PATH
) -> dict:
    """
    Load previously saved parent chunks from JSON.
    """

    storage_path = Path(storage_path)

    if not storage_path.exists():
        raise FileNotFoundError(
            f"Parent store not found: {storage_path}"
        )

    with storage_path.open("r", encoding="utf-8") as file:

        parent_store = json.load(file)

    return parent_store


def retrieve_parent(
    parent_id: str,
    storage_path: str | Path = DEFAULT_STORE_PATH
) -> Document:
    """
    Retrieve a parent document using its parent ID.
    """

    parent_store = load_parent_store(storage_path)

    if parent_id not in parent_store:
        raise KeyError(
            f"Parent document not found: {parent_id}"
        )

    parent_data = parent_store[parent_id]

    return Document(
        page_content=parent_data["page_content"],
        metadata=parent_data["metadata"]
    )
    

def retrieve_parents(
    parent_ids: list[str],
    storage_path: str | Path = DEFAULT_STORE_PATH
) -> list[Document]:
    """
    Retrieve multiple parent documents using their IDs.
    """

    parent_store = load_parent_store(storage_path)

    # Remove duplicate IDs while preserving order
    unique_parent_ids = list(dict.fromkeys(parent_ids))

    parents = []
    missing_ids = []

    for parent_id in unique_parent_ids:

        if parent_id not in parent_store:
            missing_ids.append(parent_id)
            continue

        parent_data = parent_store[parent_id]

        parents.append(
            Document(
                page_content=parent_data["page_content"],
                metadata=parent_data["metadata"]
            )
        )

    if missing_ids:
        raise KeyError(
            f"Parent documents not found: {missing_ids}"
        )

    print("\n========== PARENT DOCUMENT RETRIEVAL ==========")
    print(f"Requested parent IDs: {len(parent_ids)}")
    print(f"Unique parent documents: {len(parents)}")

    return parents