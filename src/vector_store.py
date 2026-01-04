"""Vector store for storing and retrieving document embeddings"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path


class VectorStore:
    """Vector store interface for document embeddings"""
    
    def __init__(self, store_type: str = "chromadb", storage_path: str = "./data/chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            store_type: Type of vector store ('chromadb' or 'faiss')
            storage_path: Path to store the vector database
        """
        self.store_type = store_type.lower()
        self.storage_path = storage_path
        self.store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store backend"""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.store_type == "chromadb":
            try:
                import chromadb
                from chromadb.config import Settings
                
                # Use persistent client for local storage
                self.client = chromadb.PersistentClient(
                    path=self.storage_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                self.store = "chromadb"
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ChromaDB: {e}")
        
        elif self.store_type == "faiss":
            try:
                import faiss
                self.faiss_index = None
                self.id_to_metadata = {}
                self.next_id = 0
                self.embedding_dim = None
                self.store = "faiss"
            except Exception as e:
                raise RuntimeError(f"Failed to initialize FAISS: {e}")
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: NumPy array of embeddings
            metadatas: List of metadata dictionaries
        """
        if self.store == "chromadb":
            ids = [f"doc_{i}" for i in range(len(texts))]
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        elif self.store == "faiss":
            import faiss
            
            if self.faiss_index is None:
                # Initialize FAISS index
                self.embedding_dim = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                # Normalize for cosine similarity
                self.normalize_index = True
            
            # Normalize embeddings for cosine similarity (creates a copy)
            embeddings_normalized = embeddings.astype('float32').copy()
            faiss.normalize_L2(embeddings_normalized)
            self.faiss_index.add(embeddings_normalized)
            
            # Store metadata
            start_id = self.next_id
            for i, metadata in enumerate(metadatas):
                self.id_to_metadata[start_id + i] = {
                    'text': texts[i],
                    'metadata': metadata
                }
            self.next_id += len(texts)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.store == "chromadb":
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    documents.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i] if 'distances' in results else 0.0
                    })
            return documents
        
        elif self.store == "faiss":
            import faiss
            
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            documents = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < self.next_id and idx in self.id_to_metadata:
                    doc_info = self.id_to_metadata[idx]
                    # Convert L2 distance to similarity score (cosine similarity)
                    score = 1 - (dist / 2.0)  # Approximate conversion
                    documents.append({
                        'text': doc_info['text'],
                        'metadata': doc_info['metadata'],
                        'score': max(0.0, score)
                    })
            return documents
    
    def delete_document(self, document_id: str):
        """Delete all chunks from a document"""
        if self.store == "chromadb":
            # Delete by metadata filter
            self.collection.delete(
                where={"document_id": document_id}
            )
        elif self.store == "faiss":
            # FAISS doesn't support deletion easily, would need to rebuild
            raise NotImplementedError("Document deletion not implemented for FAISS")
    
    def clear(self):
        """Clear all documents from the store"""
        if self.store == "chromadb":
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        elif self.store == "faiss":
            self.faiss_index = None
            self.id_to_metadata = {}
            self.next_id = 0
