"""MCP Server for RAG-based document question answering"""

import asyncio
import os
from pathlib import Path
from typing import Any, Sequence
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from .rag_engine import RAGEngine


# Initialize RAG Engine
rag_engine = None


def initialize_rag_engine():
    """Initialize the RAG engine with configuration from environment"""
    global rag_engine
    
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    storage_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    use_simple_llm = os.getenv("USE_SIMPLE_LLM", "false").lower() == "true"
    
    rag_engine = RAGEngine(
        embedding_model=embedding_model,
        vector_store_type=vector_store_type,
        storage_path=storage_path,
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        llm_model_name=llm_model_name,
        use_llm=use_llm,
        use_simple_llm=use_simple_llm,
        llm_device=os.getenv("LLM_DEVICE") or None,
        llm_max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "256")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "0.9"))
    )
    print(f"RAG Engine initialized with model: {embedding_model}")


# Create MCP server
server = Server("rag-mcp-server")


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources (documents in the knowledge base)"""
    # For now, return empty list
    # Could be extended to list all documents in the vector store
    return []


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="add_document",
            description="Add a supported document to the knowledge base. The document will be processed, chunked, and embedded for retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to the document file to add (.pdf, .txt, or .md)"
                    },
                    "pdf_path": {
                        "type": "string",
                        "description": "Deprecated alias for document_path"
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional identifier for the document"
                    }
                }
            }
        ),
        Tool(
            name="query_documents",
            description="Query the document knowledge base. Retrieves relevant document chunks and returns context that can be used to answer questions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or query to search for in the documents"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default: 5)",
                        "default": 5
                    },
                    "generate_answer": {
                        "type": "boolean",
                        "description": "Whether to generate an answer with the configured Hugging Face model",
                        "default": False
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="delete_document",
            description="Delete a document from the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Identifier of the document to delete"
                    }
                },
                "required": ["document_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls"""
    global rag_engine
    
    if rag_engine is None:
        initialize_rag_engine()
    
    if name == "add_document":
        document_path = arguments.get("document_path") or arguments.get("pdf_path")
        document_id = arguments.get("document_id")
        
        if not document_path or not os.path.exists(document_path):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"File not found: {document_path}"
                }, indent=2)
            )]
        
        result = rag_engine.add_document(document_path, document_id)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "query_documents":
        question = arguments.get("question")
        top_k = arguments.get("top_k", int(os.getenv("TOP_K", "5")))
        generate_answer = arguments.get("generate_answer", rag_engine.use_llm)
        
        if not question:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "Question parameter is required"
                }, indent=2)
            )]
        
        result = rag_engine.query(question, top_k=top_k, generate_answer=generate_answer)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "delete_document":
        document_id = arguments.get("document_id")
        
        try:
            rag_engine.vector_store.delete_document(document_id)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"Document {document_id} deleted"
                }, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e)
                }, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Unknown tool: {name}"
            }, indent=2)
        )]


async def main():
    """Main entry point for the MCP server"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize RAG engine
    initialize_rag_engine()
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def run():
    """Console-script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
