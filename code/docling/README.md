# RAG System - Modular Code

This directory contains the modular components of the RAG (Retrieval-Augmented Generation) system.

## Structure

- **extraction.py** - Document conversion from PDF/MD to docling format
- **chunking.py** - Document segmentation into chunks using HybridChunker
- **embedding.py** - Vector embedding setup and LanceDB storage
- **query.py** - Database querying and context retrieval
- **chat.py** - Chat interface and OpenAI response generation
- **main.py** - Main entry point orchestrating the entire pipeline
- **init.py** - Package initialization

## Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- OpenAI API key

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/muzammil-iftikhar/learning
   cd learning/code/docling
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

2. **Set up your OpenAI API key:**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Or create a `.env` file in the root directory:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Add your documents:**

   Create a `docs/` directory and add your PDF or Markdown files:

   ```bash
   mkdir -p docs
   # Copy your .pdf or .md files to docs/
   ```

## Usage

**Run the complete system:**

```bash
poetry run streamlit run 00-main.py
```

This will:

1. Process all PDF and MD files from the `docs/` directory
2. Extract and chunk documents (max 128 tokens per chunk)
3. Generate embeddings using OpenAI's `text-embedding-3-small`
4. Store vectors in LanceDB at `data/lancedb`
5. Launch a Streamlit chat interface to query your documents

**First run:** The system will automatically process all documents in the `docs/` folder and create the vector database.

**Subsequent runs:** The system will use the existing database in `data/lancedb` unless you delete it.

## Command Reference

- **Install dependencies:** `poetry install`
- **Run the app:** `poetry run streamlit run 00-main.py`
- **Add documents:** Place files in `docs/` directory (PDF or MD format)
- **Reset database:** Delete the `data/lancedb` directory to rebuild from scratch

## Features

- Processes all `.pdf` and `.md` files from the `docs/` folder
- Chunks documents with a maximum of 128 tokens
- Uses OpenAI's `text-embedding-3-small` for embeddings
- Stores data in LanceDB at `data/lancedb`
- Provides a chat interface, via streamlit, to query documents
- Displays relevant source citations with page numbers
