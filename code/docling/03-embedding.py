"""Embedding and database module.

This module handles the creation of vector embeddings and storage
in LanceDB for efficient retrieval.
"""

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from typing import List


def setup_embedding_function():
    """Set up the OpenAI embedding function.

    Returns:
        Configured embedding function
    """
    func = get_registry().get("openai").create(name="text-embedding-3-small")
    return func


def create_database_table(
    db_path: str = "data/lancedb", table_name: str = "docling", func=None
):
    """Create a LanceDB table for storing chunks.

    Args:
        db_path: Path to the LanceDB database
        table_name: Name of the table
        func: Embedding function

    Returns:
        LanceDB table object
    """
    db = lancedb.connect(db_path)

    # Define the schema inline to avoid circular dependency
    class ChunkMetadata(LanceModel):
        filename: str | None
        page_numbers: List[int] | None
        title: str | None

    class Chunks(LanceModel):
        text: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
        metadata: ChunkMetadata

    table = db.create_table(table_name, schema=Chunks, mode="overwrite")
    return table


def process_and_store_chunks(chunks, table):
    """Process chunks and store them in the database.

    Args:
        chunks: List of document chunks
        table: LanceDB table object

    Returns:
        None (updates table in place)
    """
    # Create table with processed chunks
    processed_chunks = [
        {
            "text": chunk.text,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "page_numbers": [
                    page_no
                    for page_no in sorted(
                        set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )
                    )
                ]
                or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        }
        for chunk in chunks
    ]

    table.add(processed_chunks)
