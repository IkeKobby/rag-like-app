# Colab Testing Checklist ✅

## Pre-Flight Checks

Before testing on Colab, verify:

- [x] All source files are created (`src/` directory with all modules)
- [x] Dependencies are specified in requirements.txt
- [x] Colab notebook (`colab_example.ipynb`) is created
- [x] LLM integration code is complete
- [x] Documentation is ready

## Steps to Test on Colab

### 1. Upload Files to Colab
- [ ] Open Google Colab
- [ ] Upload `colab_example.ipynb` OR create new notebook
- [ ] Upload the entire `src/` folder (keep directory structure: `src/__init__.py`, `src/rag_engine.py`, etc.)

### 2. Run Setup Cell
- [ ] Run the dependency installation cell
- [ ] Verify GPU is available (for LLM models)
- [ ] Check that directories are created

### 3. Test Initialization
- [ ] Initialize RAG engine (start with `use_llm=False` for quick test)
- [ ] Verify no import errors

### 4. Test PDF Processing
- [ ] Upload a test PDF to `data/documents/`
- [ ] Run `add_document()` 
- [ ] Verify document is processed successfully

### 5. Test Retrieval
- [ ] Run a query with `use_llm=False`
- [ ] Verify context is retrieved

### 6. Test LLM (if GPU available)
- [ ] Initialize with `use_llm=True`
- [ ] Run query with `generate_answer=True`
- [ ] Verify answer is generated

## Expected Behavior

### First Run (Embedding Model)
- First query will download embedding model (~90MB)
- Takes 1-2 minutes first time
- Subsequent runs are instant

### LLM Model (if enabled)
- First query will download LLM model (~14GB for Mistral-7B)
- Takes 5-10 minutes first time
- Model is cached for future use

## Common Issues & Solutions

### ImportError: No module named 'src'
**Solution**: Make sure `src/` folder is uploaded with correct structure

### CUDA out of memory
**Solution**: Use retrieval-only mode (`use_llm=False`) or smaller model

### Model download timeout
**Solution**: Models download on first use - be patient, or use HuggingFace cache

### PDF processing fails
**Solution**: Ensure PDF is not password-protected, try a different PDF

## Quick Test (No LLM)

For a quick test without LLM:

```python
from src.rag_engine import RAGEngine

rag = RAGEngine(
    embedding_model="all-MiniLM-L6-v2",
    vector_store_type="chromadb",
    storage_path="./data/chroma_db",
    use_llm=False  # Skip LLM for quick test
)

# Add document
result = rag.add_document("your_file.pdf")
print(result)

# Query (retrieval only)
result = rag.query("your question?", top_k=3)
print(result['context'])
```

## Status

✅ **Code is ready for Colab testing!**

All components are in place. Follow the checklist above to test.
