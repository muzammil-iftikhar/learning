"""Document extraction module.

This module handles the conversion of PDF and markdown files into a format
suitable for chunking and embedding.
"""

from pathlib import Path
from docling.document_converter import DocumentConverter


def convert_document(source_path: str):
    """Convert a document (PDF or MD) to a docling document.

    Args:
        source_path: Path to the source document

    Returns:
        Converted document object
    """
    converter = DocumentConverter()
    result = converter.convert(source_path)
    return result


def export_to_markdown(document, output_dir: Path, filename: str):
    """Export document to markdown format.

    Args:
        document: Docling document object
        output_dir: Directory to save the markdown file
        filename: Name for the output file (without extension)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export Markdown format:
    with (output_dir / f"{filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(document.export_to_markdown())
