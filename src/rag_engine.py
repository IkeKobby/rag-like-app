"""RAG Engine for document retrieval and question answering"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore


class RAGEngine:
    """Retrieval-Augmented Generation engine"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "chromadb",
        storage_path: str = "./data/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        llm_model_name: Optional[str] = None,
        use_llm: bool = False,
        use_simple_llm: bool = False
    ):
        """
        Initialize the RAG engine.
        
        Args:
            embedding_model: Name of the embedding model
            vector_store_type: Type of vector store ('chromadb' or 'faiss')
            storage_path: Path to store the vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            llm_model_name: HuggingFace model name for answer generation (optional)
            use_llm: Whether to use LLM for answer generation
            use_simple_llm: Use a smaller, faster LLM (for CPU/quick testing)
        """
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            api_key=api_key
        )
        
        self.vector_store = VectorStore(
            store_type=vector_store_type,
            storage_path=storage_path
        )
        
        # Initialize LLM service if requested
        self.llm_service = None
        self.use_llm = use_llm
        
        if use_llm and llm_model_name:
            try:
                if use_simple_llm:
                    from .llm_service import SimpleLLMService
                    self.llm_service = SimpleLLMService(model_name=llm_model_name)
                else:
                    from .llm_service import LLMService
                    self.llm_service = LLMService(model_name=llm_model_name)
                print(f"✓ LLM service initialized: {llm_model_name}")
            except Exception as e:
                print(f"Warning: Failed to load LLM service: {e}")
                print("Continuing without LLM (retrieval-only mode)")
                self.use_llm = False
    
    def add_document(self, pdf_path: str, document_id: Optional[str] = None) -> Dict:
        """
        Add a PDF document to the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document identifier
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Process PDF
            chunks = self.document_processor.process_pdf(pdf_path, document_id)
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'No text extracted from PDF'
                }
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_service.embed_text(texts)
            
            # Prepare metadata
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Store in vector database
            self.vector_store.add_documents(texts, embeddings, metadatas)
            
            return {
                'success': True,
                'document_id': chunks[0]['metadata']['document_id'],
                'num_chunks': len(chunks),
                'file_name': chunks[0]['metadata']['file_name']
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k=top_k)
            
            return results
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        generate_answer: Optional[bool] = None
    ) -> Dict:
        """
        Query the knowledge base and optionally generate an answer.
        
        Args:
            question: Question to answer
            top_k: Number of document chunks to retrieve
            generate_answer: Whether to generate an answer (defaults to self.use_llm)
            
        Returns:
            Dictionary with question, retrieved context, answer (if LLM available), and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        
        # Combine context
        context = "\n\n".join([
            f"[Document: {chunk['metadata'].get('file_name', 'Unknown')}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        result = {
            'question': question,
            'context': context,
            'chunks': retrieved_chunks,
            'num_chunks': len(retrieved_chunks),
            'answer': None
        }
        
        # Generate answer if LLM is available and requested
        should_generate = generate_answer if generate_answer is not None else self.use_llm
        
        if should_generate and self.llm_service:
            try:
                answer = self.llm_service.generate_answer(question, context)
                result['answer'] = answer
            except Exception as e:
                result['answer'] = f"Error generating answer: {str(e)}"
                result['error'] = str(e)
        elif should_generate and not self.llm_service:
            result['answer'] = "LLM service not available. Only context retrieval is available."
        
        return result
