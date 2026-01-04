"""Example usage of the RAG system"""

import os
from src.rag_engine import RAGEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Example usage of the RAG engine"""
    
    # Initialize RAG Engine
    print("Initializing RAG Engine...")
    rag = RAGEngine(
        embedding_model="all-MiniLM-L6-v2",  # Lightweight local model
        vector_store_type="chromadb",
        storage_path="./data/chroma_db"
    )
    print("✓ RAG Engine initialized\n")
    
    # Example 1: Add a document (uncomment when you have a PDF)
    # pdf_path = "path/to/your/document.pdf"
    # if os.path.exists(pdf_path):
    #     print(f"Adding document: {pdf_path}")
    #     result = rag.add_document(pdf_path, document_id="example_doc")
    #     if result['success']:
    #         print(f"✓ Added document: {result['file_name']}")
    #         print(f"  Chunks: {result['num_chunks']}\n")
    #     else:
    #         print(f"✗ Error: {result.get('error')}\n")
    
    # Example 2: Query documents
    # question = "What is the main topic of the document?"
    # print(f"Querying: {question}")
    # result = rag.query(question, top_k=3)
    # 
    # print(f"\nFound {result['num_chunks']} relevant chunks:")
    # for i, chunk in enumerate(result['chunks'], 1):
    #     print(f"\nChunk {i} (Score: {chunk['score']:.3f}):")
    #     print(f"  Source: {chunk['metadata'].get('file_name', 'Unknown')}")
    #     print(f"  Text: {chunk['text'][:150]}...")
    # 
    # print(f"\n{'='*60}")
    # print("Retrieved Context:")
    # print(f"{'='*60}\n")
    # print(result['context'])
    
    print("Example script ready. Uncomment the examples above to use with your PDFs.")


if __name__ == "__main__":
    main()
