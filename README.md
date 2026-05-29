# RAG-like Document Q&A App

A small Retrieval-Augmented Generation demo for asking questions about local
documents. It can ingest PDF, text, and Markdown files, chunk their text, embed
the chunks, store them in ChromaDB or FAISS, retrieve relevant context, and
optionally generate an answer with a public Hugging Face language model.

The default generation model is `Qwen/Qwen2.5-0.5B-Instruct`. It is small,
public, instruction-tuned, and realistic for a local demo. Larger models such
as `microsoft/Phi-3-mini-4k-instruct` usually produce better answers, but need
more RAM/VRAM and more patience on CPU.

## What It Includes

- Document ingestion for `.pdf`, `.txt`, `.md`, and `.markdown`
- PDF text extraction with `pdfplumber` and `pypdf` fallback
- Overlapping text chunking
- Local sentence-transformer embeddings
- ChromaDB persistent vector storage or in-memory FAISS search
- Question retrieval with configurable `top_k`
- Prompt construction from retrieved chunks
- Hugging Face answer generation
- Interactive CLI and MCP server entry points
- A non-interactive smoke demo in `demo_rag.py`

## Install

Use Python 3.8 or newer. Python 3.11 is known to compile this project locally.

```bash
cd /Users/ika/Documents/Codex/rag-like-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The first run downloads the embedding model. If `USE_LLM=true`, it also
downloads the configured Hugging Face generation model.

## Configure

Edit `.env` as needed:

```bash
VECTOR_STORE_TYPE=chromadb
CHROMA_DB_PATH=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5

USE_LLM=true
LLM_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
LLM_DEVICE=
LLM_MAX_NEW_TOKENS=256
LLM_TEMPERATURE=0.0
LLM_TOP_P=0.9
HUGGINGFACE_HUB_TOKEN=
```

`LLM_DEVICE` can be left blank for auto-detection. Supported values are usually
`cuda`, `mps`, or `cpu`, depending on your machine and PyTorch install.

Set `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` only if you choose a gated/private
model. Do not commit real tokens.

## Quick Smoke Demo

Run retrieval only:

```bash
python demo_rag.py --skip-llm
```

Run retrieval plus answer generation:

```bash
python demo_rag.py
```

The demo creates `data/sample_docs/acme_robotics.txt`, ingests it, asks a sample
question, and prints the retrieved chunks. With LLM enabled, it also prints a
generated answer.

## Interactive CLI

```bash
python client.py
```

Menu options:

1. Add a document (`.pdf`, `.txt`, `.md`)
2. Ask a question
3. Exit

When answer generation is enabled with `USE_LLM=true`, the CLI prints both the
generated answer and the retrieved source chunks.

## MCP Server

```bash
python -m src.mcp_server
```

Exposed tools:

- `add_document`: ingest a supported document path
- `query_documents`: retrieve context and optionally generate an answer
- `delete_document`: delete ChromaDB chunks for a document ID

## Programmatic Usage

```python
from src.rag_engine import RAGEngine

rag = RAGEngine(
    embedding_model="all-MiniLM-L6-v2",
    vector_store_type="chromadb",
    storage_path="./data/chroma_db",
    llm_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    use_llm=True,
)

rag.add_document("docs/example.md", document_id="example")
result = rag.query("What does the document say?", top_k=3)
print(result["answer"])
print(result["chunks"])
```

## Hardware Notes

- `all-MiniLM-L6-v2` embeddings are lightweight and CPU-friendly.
- Qwen2.5 0.5B can run on CPU, but generation may still be slow.
- Phi-3-mini and Mistral 7B are higher quality but much heavier.
- CUDA GPUs can use optional 4-bit loading if `bitsandbytes` is installed.
- On Apple Silicon, PyTorch may use MPS if your local install supports it.

## Known Limitations

- FAISS mode is in-memory and does not persist/reload indexes yet.
- Document deletion is only implemented for ChromaDB.
- PDF extraction quality depends on the PDF text layer.
- There is no web UI yet; the current interfaces are CLI, Python API, and MCP.
- This is a demo, not a production RAG service.
