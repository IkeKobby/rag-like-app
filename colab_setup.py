"""
Colab Setup Script
Run this in Google Colab to set up and use the RAG system
"""

# Install dependencies
print("Installing dependencies...")
!pip install -q pypdf pdfplumber sentence-transformers chromadb faiss-cpu transformers torch accelerate bitsandbytes python-dotenv

# Clone or upload the project files to Colab
# Option 1: If you have the project on GitHub
# !git clone https://github.com/yourusername/mcp_with_agentic_ai.git
# %cd mcp_with_agentic_ai

# Option 2: Upload files manually using Colab's file browser
# Then run: %cd /content/your_uploaded_folder

print("\n✓ Dependencies installed")

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n⚠ No GPU available. Using CPU (slower)")

# Create data directories
import os
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)
print("\n✓ Data directories created")

print("\n" + "="*60)
print("Setup complete! You can now use the RAG system.")
print("="*60)
