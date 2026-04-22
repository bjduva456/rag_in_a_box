"""ingest.py - Scan directories, chunk documents, and store embeddings in ChromaDB."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import chromadb
import pathspec
from sentence_transformers import SentenceTransformer

from config import load_config, get_chroma_dir, get_ragignore_path, get_raginclude_path
from readers import read_file, SUPPORTED_EXTENSIONS

# --- Load configuration ---
_config = load_config()
CHROMA_DIR = get_chroma_dir()
COLLECTION_NAME = _config["chroma"]["collection_name"]
CHROMA_SPACE = _config["chroma"]["space"]
CHUNK_SIZE = _config["ingestion"]["chunk_size"]
CHUNK_OVERLAP = _config["ingestion"]["chunk_overlap"]
EMBEDDING_MODEL = _config["embedding"]["model"]
RAGIGNORE_FILE = get_ragignore_path()
RAGINCLUDE_FILE = get_raginclude_path()


def load_ragignore() -> pathspec.PathSpec:
    """Load patterns from .ragignore file (blacklist).
    
    Returns a PathSpec object for matching against paths.
    If .ragignore doesn't exist, returns an empty PathSpec.
    """
    if not os.path.exists(RAGIGNORE_FILE):
        print(f"Warning: {RAGIGNORE_FILE} not found. No patterns will be ignored.")
        return pathspec.PathSpec.from_lines('gitwildmatch', [])
    
    with open(RAGIGNORE_FILE, 'r', encoding='utf-8') as f:
        patterns = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    return pathspec.PathSpec.from_lines('gitwildmatch', patterns)


def load_raginclude() -> pathspec.PathSpec | None:
    """Load patterns from .raginclude file (whitelist).
    
    Returns a PathSpec object for matching against paths, or None if the file
    doesn't exist or is empty. If whitelist exists, only matching files are included.
    """
    if not os.path.exists(RAGINCLUDE_FILE):
        return None
    
    with open(RAGINCLUDE_FILE, 'r', encoding='utf-8') as f:
        patterns = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if not patterns:
        return None
    
    return pathspec.PathSpec.from_lines('gitwildmatch', patterns)


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


def scan_directories(paths: list[str], ignore_spec: pathspec.PathSpec, include_spec: pathspec.PathSpec | None) -> list[str]:
    """Recursively find all supported files in the given directories.
    
    Args:
        paths: List of directories to scan
        ignore_spec: PathSpec for blacklist patterns (files to exclude)
        include_spec: Optional PathSpec for whitelist patterns (files to include).
                      If provided, only files matching this will be included.
    
    Skips files and directories based on .ragignore (and .raginclude if present).
    """
    files = []
    for dir_path in paths:
        for root, dirs, filenames in os.walk(dir_path):
            # Prune subtrees matching ignore patterns
            dirs[:] = [d for d in dirs if not ignore_spec.match_file(os.path.join(root, d))]
            for fname in filenames:
                file_path = os.path.join(root, fname)
                # Check if file matches ignore patterns
                if ignore_spec.match_file(file_path):
                    continue
                # If whitelist exists, file must match it
                if include_spec and not include_spec.match_file(file_path):
                    continue
                # Check file extension
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(file_path)
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
        metadata={"hnsw:space": CHROMA_SPACE},
    )

    ignore_spec = load_ragignore()
    include_spec = load_raginclude()
    if include_spec:
        print("Whitelist (.raginclude) is active - only matching files will be included.")
    files = scan_directories(paths, ignore_spec, include_spec)
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
