"""Query and retrieval module.

This module handles searching the database for relevant context
based on user queries.
"""

import lancedb
import streamlit as st
from typing import List


@st.cache_resource
def init_db(db_path: str = "data/lancedb"):
    """Initialize database connection.

    Args:
        db_path: Path to the LanceDB database

    Returns:
        LanceDB table object
    """
    db = lancedb.connect(db_path)
    return db.open_table("docling")


def get_context(query: str, table, num_results: int = 20) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()

    # Separate chunks into two groups: those with complete parameters (with values) and others
    complete_chunks = []
    other_chunks = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        chunk_text = f"{row['text']}{source}"

        # Check if this chunk has complete parameters (with values)
        has_complete_param = any(' = ' in line and ('tcp_rmem' in line or 'tcp_wmem' in line or 'kernel' in line.lower()) for line in row['text'].split('\n'))

        if has_complete_param:
            complete_chunks.append(chunk_text)
        else:
            other_chunks.append(chunk_text)

    # Return complete chunks first, then others
    contexts = complete_chunks + other_chunks

    return "\n\n".join(contexts)
