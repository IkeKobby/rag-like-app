"""Run a small end-to-end RAG demo with a local sample document."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag_engine import RAGEngine


SAMPLE_TEXT = """Acme Robotics builds lightweight warehouse robots for small businesses.
The Atlas Mini robot can carry 40 pounds, runs for 8 hours per charge, and uses
visual navigation to avoid people and shelving. The starter kit includes two
robots, a charging dock, and onboarding support. Acme recommends the starter
kit for warehouses under 20,000 square feet.
"""


def build_engine(use_llm: bool) -> RAGEngine:
    """Create a RAG engine from environment variables."""
    return RAGEngine(
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        vector_store_type=os.getenv("VECTOR_STORE_TYPE", "chromadb"),
        storage_path=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        use_llm=use_llm,
        use_simple_llm=os.getenv("USE_SIMPLE_LLM", "false").lower() == "true",
        llm_device=os.getenv("LLM_DEVICE") or None,
        llm_max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "256")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "0.9")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small local RAG demo.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only test ingestion and retrieval; do not load a language model.",
    )
    parser.add_argument(
        "--question",
        default="How long does the Atlas Mini robot run per charge?",
        help="Question to ask after ingesting the sample document.",
    )
    args = parser.parse_args()

    load_dotenv()

    sample_path = Path("data/sample_docs/acme_robotics.txt")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(SAMPLE_TEXT, encoding="utf-8")

    use_llm = (os.getenv("USE_LLM", "true").lower() == "true") and not args.skip_llm
    rag = build_engine(use_llm=use_llm)

    ingest_result = rag.add_document(str(sample_path), document_id="acme_robotics")
    if not ingest_result["success"]:
        raise RuntimeError(f"Document ingestion failed: {ingest_result.get('error')}")

    result = rag.query(
        args.question,
        top_k=int(os.getenv("TOP_K", "3")),
        generate_answer=use_llm,
    )

    print("Ingested:", ingest_result)
    print("Question:", result["question"])
    if result.get("answer"):
        print("Answer:", result["answer"])
    print("Retrieved chunks:")
    for index, chunk in enumerate(result["chunks"], 1):
        source = chunk["metadata"].get("file_name", "Unknown")
        print(f"{index}. {source} score={chunk['score']:.3f}")
        print(chunk["text"][:400])


if __name__ == "__main__":
    main()
