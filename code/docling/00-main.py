"""Main entry point for the RAG system.

This module orchestrates the entire document processing pipeline:
1. Document extraction and conversion
2. Chunking
3. Embedding
4. Chat interface
"""

import streamlit as st
from pathlib import Path
import glob
from extraction import convert_document
from chunking import create_chunker, chunk_document
from embedding import (
    setup_embedding_function,
    create_database_table,
    process_and_store_chunks,
)
from query import init_db
from chat import run_chat_interface


def process_documents(docs_dir: str = "docs"):
    """Process all documents in the docs directory.

    Args:
        docs_dir: Directory containing documents to process

    Returns:
        LanceDB table object ready for queries
    """
    # Find all PDF and MD files
    pdf_files = glob.glob(f"{docs_dir}/*.pdf")
    md_files = glob.glob(f"{docs_dir}/*.md")
    all_files = pdf_files + md_files

    if not all_files:
        print(f"No PDF or MD files found in {docs_dir}")
        return None

    # Setup chunker
    chunker = create_chunker(max_tokens=128)

    # Setup embedding function
    func = setup_embedding_function()

    # Create database table
    table = create_database_table(func=func)

    # Process each document
    for source in all_files:
        print(f"Processing: {source}")

        # Convert document
        result = convert_document(source)

        # Chunk document
        chunks = chunk_document(result.document, chunker)

        # Store chunks in database
        process_and_store_chunks(chunks, table)

    return table


def main():
    """Main function to run the RAG system."""
    # First time setup: process documents
    if "table" not in st.session_state:
        with st.status("Processing documents...", expanded=True) as status:
            st.write("Reading documents from docs/ folder...")
            table = process_documents("docs")
            st.session_state.table = table
            st.success("Documents processed successfully!")
            status.update(label="Documents processed!", state="complete")

    # Initialize database connection
    table = init_db()

    # Run chat interface
    run_chat_interface(table)


if __name__ == "__main__":
    main()
