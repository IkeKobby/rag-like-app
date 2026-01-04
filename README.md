# RAG-based MCP Server for Document Question Answering

A Retrieval-Augmented Generation (RAG) system built as an MCP (Model Context Protocol) server that enables document-based question answering. This application allows you to add PDF documents to a knowledge base and query them using semantic search.

## Features

- 📄 **PDF Document Processing**: Extract and chunk text from PDF files
- 🔍 **Semantic Search**: Find relevant document chunks using embeddings
- 💾 **Vector Storage**: Store and retrieve document embeddings (ChromaDB or FAISS)
- 🤖 **LLM Integration**: Generate answers using HuggingFace models (Mistral, Llama, etc.)
- 🔗 **Full RAG Pipeline**: Retrieval + Answer Generation
- 🌐 **MCP Server**: Expose functionality through Model Context Protocol
- 🚀 **Flexible**: Works on CPU (lightweight) or GPU (Colab Pro/A100)
- ☁️ **Colab Compatible**: Optimized for Google Colab with GPU support

## Architecture

### Full RAG Pipeline

```
┌─────────────────┐
│   PDF Files     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Document        │      │ Embedding Model  │
│ Processor       │─────▶│ (e.g., MiniLM)   │
│ (Chunking)      │      │ Creates Vectors  │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         │                        ▼
         │              ┌──────────────────┐
         │              │ Vector Store     │
         │              │ (ChromaDB/FAISS) │
         │              │ Stores Embeddings│
         │              └────────┬─────────┘
         │                       │
         │              User Question
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ Embedding Model  │
         │              │ (Vectorize Query)│
         │              └────────┬─────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ Similarity Search│
         │              │ Retrieve Chunks  │
         │              └────────┬─────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ LLM (HuggingFace)│
         │              │ Generate Answer  │
         │              └──────────────────┘
         │
         └─────────────── Final Answer
```

### Components Explained

1. **Embedding Model**: Converts text to vectors (semantic representations)
2. **Vector Store**: Database for fast similarity search of embeddings
3. **LLM**: Large Language Model that generates answers from context

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and configure your settings
   ```

4. **Create data directories**:
   ```bash
   mkdir -p data/documents
   mkdir -p data/chroma_db
   ```

## Configuration

Edit `.env` file to configure:

- **Embedding Model**: Choose between local models (e.g., `all-MiniLM-L6-v2`) or cloud models (requires API key)
- **Vector Store**: Select `chromadb` or `faiss`
- **Storage Paths**: Configure where to store documents and vector database
- **API Keys**: Optional - for cloud-based embeddings or LLM integration

## Usage

### Interactive Client

Run the interactive client to add documents and query them:

```bash
python client.py
```

The client provides a menu-driven interface:
1. Add PDF documents to the knowledge base
2. Query documents with questions
3. Exit

### MCP Server Mode

To run as an MCP server (for integration with MCP clients):

```bash
python -m src.mcp_server
```

### Programmatic Usage

```python
from src.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine(
    embedding_model="all-MiniLM-L6-v2",
    vector_store_type="chromadb",
    storage_path="./data/chroma_db"
)

# Add a document
result = rag.add_document("path/to/document.pdf", document_id="doc1")
print(result)

# Query documents
result = rag.query("What is the main topic of this document?", top_k=5)
print(result['context'])
```

## MCP Tools

The MCP server exposes the following tools:

### `add_document`
Add a PDF document to the knowledge base.

**Parameters:**
- `pdf_path` (required): Path to the PDF file
- `document_id` (optional): Identifier for the document

### `query_documents`
Query the document knowledge base to retrieve relevant context.

**Parameters:**
- `question` (required): The question or query text
- `top_k` (optional): Number of chunks to retrieve (default: 5)

### `delete_document`
Delete a document from the knowledge base.

**Parameters:**
- `document_id` (required): Identifier of the document to delete

## Running on Google Colab

Since your laptop cannot run heavy software, you can use Google Colab Pro:

### Quick Start

1. **Open the Colab notebook**: `colab_example.ipynb`
2. **Upload project files** to Colab (or clone from GitHub)
3. **Run the setup cells** to install dependencies
4. **Initialize RAG with LLM**:
   ```python
   from src.rag_engine import RAGEngine
   
   rag = RAGEngine(
       embedding_model="all-MiniLM-L6-v2",
       vector_store_type="chromadb",
       storage_path="./data/chroma_db",
       llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",  # HuggingFace model
       use_llm=True  # Enable answer generation
   )
   ```
5. **Add documents and query**:
   ```python
   # Add PDF
   rag.add_document("your_file.pdf")
   
   # Query with answer generation
   result = rag.query("Your question?", generate_answer=True)
   print(result['answer'])
   ```

### Recommended Models for Colab

- **Colab Pro/A100**: `mistralai/Mistral-7B-Instruct-v0.2` (high quality, ~14GB)
- **Colab Free**: Use retrieval-only mode (`use_llm=False`) or `distilgpt2` for testing

See `EXPLANATION.md` for detailed information about models and architecture.

## Project Structure

```
mcp_with_agentic_ai/
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF processing and chunking
│   ├── embedding_service.py     # Embedding generation
│   ├── vector_store.py          # Vector database interface
│   ├── rag_engine.py            # Main RAG orchestration
│   └── mcp_server.py            # MCP server implementation
├── data/                        # Data storage (created at runtime)
│   ├── documents/               # PDF files storage
│   └── chroma_db/               # Vector database
├── client.py                    # Interactive client application
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Technologies Used

- **MCP (Model Context Protocol)**: For server interface
- **Sentence Transformers**: For local embeddings
- **ChromaDB/FAISS**: Vector databases
- **PyPDF/pdfplumber**: PDF text extraction
- **OpenAI API** (optional): For cloud-based embeddings

## Portfolio Use Case

This project demonstrates:

- ✅ **Agentic AI Systems**: RAG pipeline with autonomous document processing
- ✅ **MCP Integration**: Modern protocol-based AI agent interaction
- ✅ **Vector Databases**: Efficient semantic search implementation
- ✅ **End-to-End System**: Complete workflow from PDF ingestion to Q&A
- ✅ **Production-Ready Code**: Well-structured, modular architecture

Perfect for showcasing SLM-first agentic AI systems expertise!

## Future Enhancements

- [ ] LLM integration for answer generation (using retrieved context)
- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Web UI for document management and querying
- [ ] Multi-modal support (images in PDFs)
- [ ] Advanced chunking strategies (semantic chunking)
- [ ] Query expansion and refinement
- [ ] Document metadata filtering

## License

This project is part of a research portfolio focused on agentic AI systems.

## Author

Isaac Kobby - PhD Researcher in SLM-first Agentic AI Systems
Portfolio: https://isaackobby.com
