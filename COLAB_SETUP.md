# Google Colab Setup Guide

## Quick Start Steps

### 1. Upload Files to Colab

**Option A: Upload via Colab File Browser**
1. Open Google Colab (colab.research.google.com)
2. Create a new notebook or upload `colab_example.ipynb`
3. Click the folder icon (📁) in the left sidebar
4. Create a `src/` folder
5. Upload all files from the `src/` directory:
   - `__init__.py`
   - `document_processor.py`
   - `embedding_service.py`
   - `vector_store.py`
   - `rag_engine.py`
   - `llm_service.py`

**Option B: Use GitHub (Recommended)**
1. Push your code to GitHub
2. In Colab, run:
   ```python
   !git clone https://github.com/yourusername/mcp_with_agentic_ai.git
   %cd mcp_with_agentic_ai
   ```

### 2. Install Dependencies

Run this in a Colab cell:
```python
!pip install -q pypdf pdfplumber sentence-transformers chromadb faiss-cpu transformers torch accelerate bitsandbytes python-dotenv
```

### 3. Verify GPU (Important for LLMs)

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Note**: For LLM models like Mistral-7B, you need a GPU (Colab Pro recommended).

### 4. Create Directories

```python
import os
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)
```

### 5. Initialize RAG Engine

**For Colab Pro/A100 (with LLM):**
```python
from src.rag_engine import RAGEngine

rag = RAGEngine(
    embedding_model="all-MiniLM-L6-v2",
    vector_store_type="chromadb",
    storage_path="./data/chroma_db",
    llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    use_llm=True,
    use_simple_llm=False
)
```

**For Colab Free (retrieval-only, no LLM):**
```python
from src.rag_engine import RAGEngine

rag = RAGEngine(
    embedding_model="all-MiniLM-L6-v2",
    vector_store_type="chromadb",
    storage_path="./data/chroma_db",
    use_llm=False  # No answer generation
)
```

### 6. Upload and Process PDFs

1. Upload PDF files using Colab's file browser to `data/documents/`
2. Add documents:
   ```python
   pdf_path = "data/documents/your_file.pdf"
   result = rag.add_document(pdf_path, document_id="doc1")
   print(result)
   ```

### 7. Query Documents

```python
question = "What is the main topic?"
result = rag.query(question, top_k=5, generate_answer=True)

if result.get('answer'):
    print("Answer:", result['answer'])
print("\nContext:", result['context'])
```

## Important Notes

### Memory Requirements

- **Mistral-7B-Instruct**: Needs ~16GB RAM (Colab Pro with A100)
- **Retrieval-only mode**: Works on Colab Free (~4GB RAM)
- **Embedding model**: Only needs ~500MB RAM

### First Run

- Embedding model downloads automatically (~90MB) - one time only
- LLM model downloads on first use (~14GB for Mistral-7B) - one time only
- Models are cached in Colab's environment

### Troubleshooting

**Out of Memory Error:**
- Use retrieval-only mode (`use_llm=False`)
- Or use a smaller model like `distilgpt2` (lower quality)

**Import Errors:**
- Make sure you uploaded all files in the `src/` directory
- Restart runtime after installing dependencies

**GPU Not Available:**
- LLM models need GPU
- Use retrieval-only mode if no GPU available
- Or upgrade to Colab Pro

## Complete Example

See `colab_example.ipynb` for a complete working example with all cells ready to run!
