"""ingest.py - Scan directories, chunk documents, and store embeddings in ChromaDB."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from readers import read_file, SUPPORTED_EXTENSIONS

# --- Configuration ---
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50  # characters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] if overlap else ""

            # If a single paragraph is too long, split on sentences
            if len(para) > chunk_size:
                sentences = para.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = current_chunk[-overlap:] if overlap else ""
                    current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


#TODO: use .gitignore-like file to specify skip patterns, and support globs (e.g. "data/**/*.pdf")
SKIP_DIRS = {
    ".venv", "venv", "env",          # Python virtual environments
    "node_modules",                   # Node.js dependencies
    "site-packages", "dist-info",     # installed package trees
    "__pycache__",                    # bytecode cache
    ".git", ".svn", ".hg",           # version control internals
}


def scan_directories(paths: list[str]) -> list[str]:
    """Recursively find all supported files in the given directories."""
    files = []
    for dir_path in paths:
        for root, dirs, filenames in os.walk(dir_path):
            # Prune subtrees we never want to descend into
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in filenames:
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, fname))
    return sorted(files)


def file_id(file_path: str) -> str:
    """Generate a stable ID prefix from a file path."""
    return hashlib.md5(os.path.abspath(file_path).encode()).hexdigest()[:12]


def ingest(paths: list[str]):
    """Main ingestion pipeline."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Opening ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = scan_directories(paths)
    print(f"Found {len(files)} supported files.\n")

    added, skipped, updated = 0, 0, 0

    for fpath in files:
        fid = file_id(fpath)
        mtime = os.path.getmtime(fpath)

        # Check if file is already indexed with same mtime
        existing = collection.get(where={"source_file_id": fid}, include=["metadatas"])

        if existing["ids"]:
            stored_mtime = existing["metadatas"][0].get("mtime", 0)
            if stored_mtime == mtime:
                skipped += 1
                print(f"  SKIP (unchanged): {fpath}")
                continue
            else:
                # File changed -- delete old chunks, re-ingest
                collection.delete(ids=existing["ids"])
                updated += 1
                print(f"  UPDATE: {fpath}")
        else:
            added += 1
            print(f"  ADD: {fpath}")

        # Read and chunk
        try:
            text = read_file(fpath)
        except Exception as e:
            print(f"  ERROR reading {fpath}: {e}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"  WARN: no content in {fpath}")
            continue

        # Generate embeddings
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Build IDs and metadata
        ids = [f"{fid}_chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": os.path.abspath(fpath),
                "source_file_id": fid,
                "chunk_index": i,
                "mtime": mtime,
            }
            for i in range(len(chunks))
        ]

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    total = collection.count()
    print(f"\nDone. Added: {added}, Updated: {updated}, Skipped: {skipped}")
    print(f"Total chunks in database: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector database."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories to scan for documents.",
    )
    args = parser.parse_args()

    for d in args.directories:
        if not os.path.isdir(d):
            print(f"Error: '{d}' is not a directory.")
            sys.exit(1)

    ingest(args.directories)
