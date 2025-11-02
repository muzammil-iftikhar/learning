# RAG System - Modular Code

This directory contains the modular components of the RAG (Retrieval-Augmented Generation) system.

## Structure

- **extraction.py** - Document conversion from PDF/MD to docling format
- **chunking.py** - Document segmentation into chunks using HybridChunker
- **embedding.py** - Vector embedding setup and LanceDB storage
- **query.py** - Database querying and context retrieval
- **chat.py** - Chat interface and OpenAI response generation
- **main.py** - Main entry point orchestrating the entire pipeline
- \***\*init**.py\*\* - Package initialization

## Usage

Run the complete system:

```bash
poetry run streamlit run code/main.py
```

## Features

- Processes all `.pdf` and `.md` files from the `docs/` folder
- Chunks documents with a maximum of 128 tokens
- Uses OpenAI's `text-embedding-3-small` for embeddings
- Stores data in LanceDB at `data/lancedb`
- Provides a chat interface, via streamlit, to query documents
- Displays relevant source citations with page numbers
