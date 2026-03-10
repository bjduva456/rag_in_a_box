# RAG in a Box

A lightweight, self-contained Retrieval-Augmented Generation (RAG) system that lets you ask questions about your documents. Uses ChromaDB for vector storage, Sentence Transformers for embeddings, and LM Studio for local LLM inference.

## Features

- **Document Support**: Ingest `.txt`, `.md`, `.docx`, and `.odt` files
- **Vector Search**: Retrieve relevant document chunks using semantic similarity
- **Local LLM**: Works with LM Studio for completely private inference
- **Web UI**: User-friendly Gradio interface for querying
- **Conversation History**: Chat interface remembers previous messages for context

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

Edit the configuration variables in `app.py` and `ingest.py` to customize:

- `CHUNK_SIZE`: Size of document chunks (default: 500 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50 characters)
- `TOP_K`: Number of chunks to retrieve (default: 5)
- `EMBEDDING_MODEL`: Sentence Transformer model (default: `all-MiniLM-L6-v2`)
- `LM_STUDIO_URL`: LM Studio API endpoint (default: `http://localhost:1234/v1`)

## Supported File Formats

- **Plain text** (`.txt`)
- **Markdown** (`.md`)
- **Word documents** (`.docx`)
- **LibreOffice documents** (`.odt`)

## How It Works

1. **Ingestion**: Documents are split into chunks with configurable overlap to preserve context
2. **Embedding**: Each chunk is converted to a dense vector using a pre-trained model
3. **Storage**: Vectors and metadata are stored in ChromaDB with cosine similarity indexing
4. **Retrieval**: User queries are embedded and matched against stored chunks
5. **Generation**: Top-k chunks are sent to an LLM with system instructions for context-aware answers

## Troubleshooting

**"The database is empty"**
- Run `python ingest.py <directory>` first to index documents

**Connection refused to LM Studio**
- Make sure LM Studio is running and the server is started (port 1234)
- Check that `LM_STUDIO_URL` in `app.py` matches your LM Studio configuration

**Slow ingestion**
- Reduce `CHUNK_SIZE` or skip large directories by moving them out of the scan path

## Project Structure

```
rag_in_a_box/
├── app.py              # Gradio web UI
├── ingest.py           # Document ingestion pipeline
├── readers.py          # File format readers
├── requirements.txt    # Python dependencies
└── chroma_db/          # Vector database storage (created on first run)
```

## License

[Add your license here]
