"""Document processing module for PDF extraction and chunking"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import pypdf
import pdfplumber


class DocumentProcessor:
    """Processes PDF documents and splits them into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"Processing PDF with pdfplumber ({len(pdf.pages)} pages)...")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    # Progress indicator for large PDFs
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(pdf.pages)} pages...")
        except Exception as e:
            print(f"pdfplumber failed, trying pypdf: {e}")
            # Fallback to pypdf
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    print(f"Processing PDF with pypdf ({len(pdf_reader.pages)} pages)...")
                    for i, page in enumerate(pdf_reader.pages):
                        text += page.extract_text() + "\n"
                        # Progress indicator for large PDFs
                        if (i + 1) % 10 == 0:
                            print(f"  Processed {i + 1}/{len(pdf_reader.pages)} pages...")
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF: {e}")
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        metadata = metadata or {}
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                for delimiter in ['. ', '.\n', '! ', '?\n', '?\n']:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter != -1:
                        end = last_delimiter + len(delimiter)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    'chunk_index': len(chunks),
                    'start_char': start,
                    'end_char': end
                }
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_length else end
        
        return chunks
    
    def process_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> List[Dict]:
        """
        Process a PDF file: extract text and chunk it.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document identifier
            
        Returns:
            List of chunks with text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        document_id = document_id or Path(pdf_path).stem
        
        print(f"Extracting text from PDF: {Path(pdf_path).name}")
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        print(f"Text extracted: {len(text)} characters")
        print(f"Chunking text...")
        
        # Create metadata
        metadata = {
            'document_id': document_id,
            'file_path': pdf_path,
            'file_name': Path(pdf_path).name,
            'total_chars': len(text)
        }
        
        # Chunk text
        chunks = self.chunk_text(text, metadata)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
