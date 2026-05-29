"""Client application for interacting with the RAG MCP server"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Import the RAG engine directly for standalone use
from src.rag_engine import RAGEngine


def print_separator():
    """Print a separator line"""
    print("\n" + "="*80 + "\n")


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


async def interactive_client():
    """Interactive client for the RAG system"""
    load_dotenv()
    
    # Initialize RAG Engine
    print_section("Initializing RAG Engine")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    storage_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    use_simple_llm = os.getenv("USE_SIMPLE_LLM", "false").lower() == "true"
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    default_top_k = int(os.getenv("TOP_K", "5"))
    llm_device = os.getenv("LLM_DEVICE") or None
    llm_max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_top_p = float(os.getenv("LLM_TOP_P", "0.9"))
    
    print(f"Embedding Model: {embedding_model}")
    print(f"Vector Store: {vector_store_type}")
    print(f"Storage Path: {storage_path}")
    if use_llm and llm_model_name:
        print(f"LLM Model: {llm_model_name} (Answer generation enabled)")
    else:
        print("LLM: Not configured (Retrieval-only mode)")
    
    rag_engine = RAGEngine(
        embedding_model=embedding_model,
        vector_store_type=vector_store_type,
        storage_path=storage_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        llm_model_name=llm_model_name,
        use_llm=use_llm,
        use_simple_llm=use_simple_llm,
        llm_device=llm_device,
        llm_max_new_tokens=llm_max_new_tokens,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p
    )
    
    print("✓ RAG Engine initialized successfully!")
    
    # Main loop
    while True:
        print_separator()
        print("RAG Document Q&A System")
        print("\nOptions:")
        print("  1. Add a document (.pdf, .txt, .md)")
        print("  2. Query documents")
        print("  3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print_section("Add Document")
            document_path = input("Enter path to document file: ").strip()
            
            if not os.path.exists(document_path):
                print(f"❌ Error: File not found: {document_path}")
                continue
            
            document_id = input("Enter document ID (optional, press Enter to use filename): ").strip()
            if not document_id:
                document_id = None
            
            print("\nProcessing document...")
            result = rag_engine.add_document(document_path, document_id)
            
            if result['success']:
                print(f"✓ Successfully added document: {result['file_name']}")
                print(f"  Document ID: {result['document_id']}")
                print(f"  Number of chunks: {result['num_chunks']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        elif choice == "2":
            print_section("Query Documents")
            question = input("Enter your question: ").strip()
            
            if not question:
                print("❌ Error: Question cannot be empty")
                continue
            
            top_k = input(f"Number of chunks to retrieve (default: {default_top_k}): ").strip()
            top_k = int(top_k) if top_k.isdigit() else default_top_k
            
            print("\nSearching documents...")
            result = rag_engine.query(question, top_k=top_k, generate_answer=rag_engine.use_llm)
            
            print(f"\n{'─'*80}")
            print(f"Question: {result['question']}")
            print(f"{'─'*80}\n")
            
            # Show generated answer if available
            if result.get('answer'):
                print(f"{'='*80}")
                print("Generated Answer:")
                print(f"{'='*80}")
                print(result['answer'])
                print(f"\n{'─'*80}\n")
            
            print(f"Found {result['num_chunks']} relevant chunks:\n")
            
            for i, chunk in enumerate(result['chunks'], 1):
                print(f"Chunk {i} (Score: {chunk['score']:.3f})")
                print(f"Source: {chunk['metadata'].get('file_name', 'Unknown')}")
                print(f"Text: {chunk['text'][:200]}...")
                print()
            
            print(f"{'─'*80}")
            print("Retrieved Context:")
            print(f"{'─'*80}\n")
            print(result['context'])
            print()
        
        elif choice == "3":
            print("\nGoodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Console-script entry point."""
    asyncio.run(interactive_client())


if __name__ == "__main__":
    main()
