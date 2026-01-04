"""Embedding service for generating vector embeddings"""

import os
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", api_key: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            api_key: Optional API key for cloud-based embeddings
        """
        self.model_name = model_name
        self.api_key = api_key
        self.model = None
        self.use_local = not model_name.startswith(('text-embedding', 'text-'))
        
        if self.use_local:
            try:
                print(f"Loading local embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
                print(f"Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            NumPy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        if self.use_local and self.model:
            embeddings = self.model.encode(text, convert_to_numpy=True)
            # Ensure 2D array
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings
        else:
            # Fallback to OpenAI API if available
            try:
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError("API key required for cloud embeddings")
                
                client = OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                embeddings = np.array([item.embedding for item in response.data])
                return embeddings
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if self.use_local and self.model:
            # Test with a dummy input to get dimension
            test_embedding = self.embed_text("test")
            return test_embedding.shape[-1]
        elif self.model_name.startswith('text-embedding-3'):
            return 1536  # OpenAI text-embedding-3 models
        else:
            return 1536  # Default for OpenAI models
