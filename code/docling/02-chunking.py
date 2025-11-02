"""Document chunking module.

This module handles the segmentation of documents into smaller chunks
for efficient embedding and retrieval.
"""

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


def create_chunker(max_tokens: int = 128, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create a HybridChunker for document segmentation.

    Args:
        max_tokens: Maximum tokens per chunk
        model_id: Model ID for the tokenizer

    Returns:
        Configured HybridChunker instance
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=max_tokens,
    )

    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=False)
    return chunker


def chunk_document(document, chunker):
    """Chunk a document into smaller pieces.

    Args:
        document: Docling document object
        chunker: HybridChunker instance

    Returns:
        List of document chunks
    """
    chunk_iter = chunker.chunk(dl_doc=document)
    chunks = list(chunk_iter)
    return chunks
