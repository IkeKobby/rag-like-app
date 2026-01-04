# Quick Start Guide

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env if you want to use cloud embeddings or change settings
   ```

3. **Create data directories**:
   ```bash
   mkdir -p data/documents data/chroma_db
   ```

## Running the Interactive Client

The easiest way to get started is using the interactive client:

```bash
python client.py
```

This will:
1. Initialize the RAG engine (downloads the embedding model on first run)
2. Show a menu where you can:
   - Add PDF documents to the knowledge base
   - Query documents with questions
   - Exit

## Example Workflow

1. **Start the client**:
   ```bash
   python client.py
   ```

2. **Add a document**:
   - Choose option `1`
   - Enter the path to your PDF file
   - Optionally provide a document ID

3. **Query documents**:
   - Choose option `2`
   - Enter your question
   - Review the retrieved context

## Programmatic Usage

```python
from src.rag_engine import RAGEngine

# Initialize
rag = RAGEngine()

# Add a document
result = rag.add_document("path/to/document.pdf")
print(result)

# Query
result = rag.query("What is this document about?")
print(result['context'])
```

## Running as MCP Server

To run as an MCP server for integration with MCP clients:

```bash
python -m src.mcp_server
```

## Troubleshooting

### First Run

On first run, the embedding model will be downloaded (about 90MB). This only happens once.

### Memory Issues

If you encounter memory issues:
- Use a smaller embedding model (e.g., `all-MiniLM-L6-v2` - already the default)
- Reduce `chunk_size` in RAGEngine initialization
- Use FAISS instead of ChromaDB (set `VECTOR_STORE_TYPE=faiss` in `.env`)

### PDF Processing Issues

If PDFs fail to process:
- Ensure the PDF is not password-protected
- Try a different PDF file
- Check that the file path is correct

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for code examples
- Explore the source code in the `src/` directory
