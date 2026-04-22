# RAG in a Box

A lightweight, self-contained Retrieval-Augmented Generation (RAG) system that lets you ask questions about your documents. Uses ChromaDB for vector storage, Sentence Transformers for embeddings, and LM Studio for local LLM inference.

## Features

- **Document Support**: Ingest `.txt`, `.md`, `.docx`, and `.odt` files
- **Smart Filtering**: Control ingestion with `.ragignore` (blacklist) and `.raginclude` (whitelist) files
- **Enhanced Semantic Search**: Query expansion, keyword extraction, and multi-stage retrieval for better relevance
- **Vector Search**: Retrieve relevant document chunks using semantic similarity with distance-based filtering
- **Local LLM**: Works with LM Studio for completely private inference
- **Web UI**: User-friendly Gradio interface with conversation history
- **Conversation Context**: Chat interface uses conversation history to improve relevance of retrieved documents
- **Centralized Configuration**: All settings in one `config.json` file

## Prerequisites

- Python 3.8+
- [LM Studio](https://lmstudio.ai/) installed and running locally
- A document directory to index

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd rag_in_a_box
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Index Your Documents

Run the ingestion script on a directory containing your documents:

```bash
python ingest.py /path/to/documents
```

This will:
- Scan the directory recursively for supported file types
- Extract text and split into overlapping chunks
- Generate embeddings using Sentence Transformers
- Store everything in ChromaDB

**Example:**
```bash
python ingest.py ~/my_docs
python ingest.py ./research_papers ./notes
```

### Step 2: Start LM Studio

1. Open LM Studio
2. Load your preferred model from the model library
3. Click the **Start Server** button (server will run on `http://localhost:1234`)

### Step 3: Launch the Web UI

```bash
python app.py
```

This will start the Gradio interface. Open your browser to the URL shown in the terminal (typically `http://127.0.0.1:7860`).

### Step 4: Ask Questions

Type your questions in the chat interface. The system will:
1. Find the most relevant document chunks
2. Send them to your local LLM with your question
3. Return an answer with source citations

## Configuration

All configuration is managed through `config.json` in the project root. Edit this file to customize:

### config.json Structure

```json
{
  "chroma": {
    "directory": "chroma_db",
    "collection_name": "documents",
    "space": "cosine"
  },
  "embedding": {
    "model": "all-MiniLM-L6-v2"
  },
  "ingestion": {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "supported_extensions": [".txt", ".md", ".docx", ".odt"]
  },
  "retrieval": {
    "top_k": 50,
    "max_results": 20,
    "distance_threshold": 0.5
  },
  "llm": {
    "base_url": "http://localhost:1234/v1",
    "model_name": "openai/gpt-oss-20b",
    "temperature": 0.5,
    "max_tokens": 4096
  }
}
```

### Configuration Guide

#### ChromaDB Settings
- `directory`: Path to ChromaDB storage (relative to project root)
- `collection_name`: Name of the document collection
- `space`: Similarity metric ("cosine", "l2", or "ip")

#### Embedding
- `model`: Sentence Transformer model for embeddings

#### Ingestion
- `chunk_size`: Document chunk size in characters
- `chunk_overlap`: Overlap between chunks to preserve context
- `supported_extensions`: File types to ingest

#### Retrieval (Semantic Search)
- `top_k`: Number of candidates to retrieve initially (higher = more thorough but slower)
- `max_results`: Maximum results to return after filtering and re-ranking
- `distance_threshold`: Maximum distance for semantic similarity (0 = exact, 1 = any)

#### LLM (Language Model)
- `base_url`: LM Studio API endpoint
- `model_name`: Model identifier for LM Studio
- `temperature`: Model creativity (0.0 = deterministic, 1.0 = creative)
- `max_tokens`: Maximum response length

### File Filtering

#### `.ragignore` (Blacklist)
Create a `.ragignore` file to exclude files and directories, similar to `.gitignore`. Patterns use gitignore syntax with support for wildcards and globs.

**Example:**
```
# Exclude directories
node_modules
.venv
venv

# Exclude file types
*.pdf
*.zip

# Exclude specific files
.DS_Store
```

#### `.raginclude` (Whitelist)
Create a `.raginclude` file to explicitly include only matching files. This takes precedence over supported extensions.

**Example:**
```
# Only include markdown and text files
*.md
*.txt

# Include specific directories
docs/
README*
```

**Note**: If `.raginclude` exists with patterns, only files matching those patterns will be included (still subject to `.ragignore` exclusions).

## Supported File Formats

- **Plain text** (`.txt`)
- **Markdown** (`.md`)
- **Word documents** (`.docx`)
- **LibreOffice documents** (`.odt`)

## How It Works

### Ingestion Pipeline
1. **File Scanning**: Recursively scans directories, filtering based on `.ragignore` (blacklist) and `.raginclude` (whitelist) patterns
2. **Text Extraction**: Reads supported file formats and extracts text content
3. **Chunking**: Splits documents into overlapping chunks to preserve context (configurable size and overlap)
4. **Embedding**: Converts each chunk to a dense vector using Sentence Transformers
5. **Storage**: Stores vectors and metadata in ChromaDB with configurable similarity metric

### Retrieval and Ranking
1. **Query Expansion**: Automatically adds synonyms to your query (e.g., "neural" → "deep learning", "AI", "ml")
2. **Semantic Search**: Embeds your query and retrieves candidate chunks from ChromaDB
3. **Distance Filtering**: Filters results by semantic similarity threshold
4. **Keyword Scoring**: Re-ranks results by matching query keywords in the text
5. **Ranking**: Sorts by keyword score (most relevant) then by distance (similarity)
6. **Context Awareness**: Uses recent conversation history to improve retrieval for follow-up questions

### Generation
1. **System Prompt**: Constructs a prompt with retrieved chunks and relevance scores
2. **LLM Call**: Sends context and your question to the local LLM
3. **Response**: Returns the answer with source citations and relevance indicators

## Troubleshooting

**"The database is empty"**
- Run `python ingest.py <directory>` first to index documents

**Connection refused to LM Studio**
- Make sure LM Studio is running and the server is started (port 1234)
- Check that `llm.base_url` in `config.json` matches your LM Studio configuration

**Slow ingestion**
- Reduce `ingestion.chunk_size` in `config.json` or skip large directories using `.ragignore`

**Specific topics not showing up in results**
- Try rephrasing your question with different keywords
- Increase `retrieval.top_k` in `config.json` to retrieve more candidates
- Lower `retrieval.distance_threshold` to include more loosely related results (default: 0.5)

## Project Structure

```
rag_in_a_box/
├── app.py              # Gradio web UI and retrieval logic
├── ingest.py           # Document ingestion pipeline
├── readers.py          # File format readers
├── config.py           # Configuration loader
├── config.json         # Central configuration file
├── .ragignore          # Blacklist patterns (gitignore style)
├── .raginclude         # Whitelist patterns (gitignore style)
├── requirements.txt    # Python dependencies
└── chroma_db/          # Vector database storage (created on first run)
```