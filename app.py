"""app.py - Gradio web UI for querying the RAG system via LM Studio."""

import os
import re

import gradio as gr
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import load_config

# --- Load configuration ---
_config = load_config()
CHROMA_DIR = _config["chroma"]["directory"]
if not os.path.isabs(CHROMA_DIR):
    CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHROMA_DIR)

COLLECTION_NAME = _config["chroma"]["collection_name"]
CHROMA_SPACE = _config["chroma"]["space"]
EMBEDDING_MODEL = _config["embedding"]["model"]
LM_STUDIO_URL = _config["llm"]["base_url"]
LLM_MODEL_NAME = _config["llm"]["model_name"]
LLM_TEMPERATURE = _config["llm"]["temperature"]
LLM_MAX_TOKENS = _config["llm"]["max_tokens"]
TOP_K = _config["retrieval"]["top_k"]
MAX_RESULTS = _config["retrieval"]["max_results"]
DISTANCE_THRESHOLD = _config["retrieval"]["distance_threshold"]

# TUNING GUIDE:
# Edit config.json to adjust:
# - Increase retrieval.top_k to retrieve more candidates for filtering (more thorough but slower)
# - Increase retrieval.max_results to return more sources for context (uses more tokens)
# - Lower retrieval.distance_threshold for stricter relevance filtering (0 = exact match only)
# - Raise retrieval.distance_threshold to include more loosely related results (up to 1.0)

# --- Initialize components (loaded once at startup) ---
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": CHROMA_SPACE},
)

llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")


def extract_keywords(text: str) -> list[str]:
    """Extract important keywords from text by removing common stopwords."""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'it', 'this', 'that',
        'what', 'which', 'who', 'when', 'where', 'why', 'how'
    }
    # Split on punctuation and whitespace, keep words longer than 2 chars
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]


def retrieve(query: str, top_k: int = TOP_K, max_results: int = MAX_RESULTS) -> list[dict]:
    """Retrieve the most relevant chunks for a query with improved ranking.
    
    Strategy:
    1. Retrieve more candidates than needed
    2. Filter by distance threshold
    3. Rank by keyword presence in results
    4. Return top results
    """
    # Extract keywords for post-ranking
    keywords = set(extract_keywords(query))
    
    # Embed query for semantic matching
    query_embedding = embedder.encode([query]).tolist()

    # Retrieve more candidates to filter and re-rank
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k * 2, 100),  # Fetch up to 2x requested for filtering
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        
        # Filter by distance threshold
        if distance > DISTANCE_THRESHOLD:
            continue
        
        text = results["documents"][0][i]
        chunk = {
            "text": text,
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": distance,
        }
        
        # Score by keyword presence (higher score = more keywords match)
        keyword_score = sum(1 for kw in keywords if kw.lower() in text.lower())
        chunk["keyword_score"] = keyword_score
        
        chunks.append(chunk)
    
    # Sort by keyword score (descending) then by distance (ascending)
    chunks.sort(key=lambda x: (-x["keyword_score"], x["distance"]))
    
    # Return top results
    return chunks[:max_results]


def build_system_prompt(chunks: list[dict]) -> str:
    """Build the system message containing retrieved context and instructions."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_name = chunk["source"].split("/")[-1]
        # Add relevance indicator (lower distance = more relevant)
        relevance = f"{(1 - chunk['distance']):.1%}" if chunk.get('distance') is not None else "N/A"
        context_parts.append(
            f"[Source {i}: {source_name} (relevance: {relevance})]\n{chunk['text']}"
        )

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
    """Handle a user query: retrieve with context, prompt LLM, return response."""
    if collection.count() == 0:
        return (
            "The database is empty. Run `python ingest.py <directory>` "
            "first to index some documents."
        )

    # Build context-aware query using recent conversation history
    query_for_retrieval = message
    if history and len(history) > 0:
        # Extract text from recent user messages (safely handle various content types)
        context_parts = []
        for turn in history[-6:]:
            if turn.get("role") == "user":
                content = turn.get("content", "")
                # Handle cases where content might be a list (e.g., multimodal)
                if isinstance(content, list):
                    # Extract strings from list items
                    for item in content:
                        if isinstance(item, str):
                            context_parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            context_parts.append(item["text"])
                elif isinstance(content, str):
                    context_parts.append(content)
        
        if context_parts:
            recent_context = " ".join(context_parts)
            # Combine current message with relevant context from history
            query_for_retrieval = f"{message} {recent_context}"

    chunks = retrieve(query_for_retrieval)

    if not chunks:
        return (
            "No relevant documents found. Try rephrasing your question or "
            "check that documents have been indexed with `python ingest.py <directory>`."
        )

    messages = [{"role": "system", "content": build_system_prompt(chunks)}]

    # Thread conversation history so the LLM can build on prior turns.
    # Gradio 6 passes history as list[dict] with "role" and "content" keys.
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": message})

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
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
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0")
