"""readers.py - Read text content from supported file formats."""

import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".odt"}


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_docx(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def read_odt(file_path: str) -> str:
    from odf.opendocument import load
    from odf.text import P
    from odf import teletype

    doc = load(file_path)
    paragraphs = doc.getElementsByType(P)
    return "\n\n".join(
        teletype.extractText(p) for p in paragraphs if teletype.extractText(p).strip()
    )


_READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".docx": read_docx,
    ".odt": read_odt,
}


def read_file(file_path: str) -> str:
    """Read a file and return its text content.

    Raises ValueError for unsupported file types.
    """
    ext = Path(file_path).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return reader(file_path)
