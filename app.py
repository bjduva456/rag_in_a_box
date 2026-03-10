"""app.py - Gradio web UI for querying the RAG system via LM Studio."""

import os

import gradio as gr
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LM_STUDIO_URL = "http://localhost:1234/v1"
TOP_K = 5  # number of chunks to retrieve

# --- Initialize components (loaded once at startup) ---
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query."""
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "distance": results["distances"][0][i],
            }
        )
    return chunks


def build_system_prompt(chunks: list[dict]) -> str:
    """Build the system message containing retrieved context and instructions."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_name = chunk["source"].split("/")[-1]
        context_parts.append(f"[Source {i}: {source_name}]\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    return (
        "You are a helpful assistant. Answer the user's questions based on "
        "the provided context. If the context does not contain enough "
        "information, say so. Cite which source(s) you used.\n\n"
        f"## Context\n\n{context_block}"
    )


def format_sources(chunks: list[dict]) -> str:
    """Format source references for display below the answer."""
    lines = ["\n\n---\n**Sources used:**"]
    seen = set()
    for chunk in chunks:
        src = chunk["source"]
        if src not in seen:
            seen.add(src)
            lines.append(f"- `{src}`")
    return "\n".join(lines)


def ask(message: str, history: list) -> str:
    """Handle a user query: retrieve, prompt LLM, return response."""
    if collection.count() == 0:
        return (
            "The database is empty. Run `python ingest.py <directory>` "
            "first to index some documents."
        )

    chunks = retrieve(message)

    messages = [{"role": "system", "content": build_system_prompt(chunks)}]

    # Thread conversation history so the LLM can build on prior turns.
    # Gradio 6 passes history as list[dict] with "role" and "content" keys.
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": message})

    try:
        response = llm_client.chat.completions.create(
            model="local-model",  # LM Studio ignores this, uses loaded model
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return (
            f"Error connecting to LM Studio at {LM_STUDIO_URL}.\n\n"
            f"Make sure LM Studio is running with a model loaded "
            f"and the local server is started on port 1234.\n\n"
            f"Error: {e}"
        )

    answer += format_sources(chunks)
    return answer


# --- Build Gradio UI ---
with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=ask,
        title="RAG-in-a-Box",
        description=(
            "Ask questions about your documents. "
            f"Database has **{collection.count()}** chunks indexed."
        ),
        examples=[
            "Summarize the key points from the documents.",
            "What topics are covered in the indexed files?",
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
