import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Typed, environment-backed application configuration."""

    documents_path: str
    chroma_path: str
    chroma_collection: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_provider: str
    gemini_model: str | None
    gemini_api_key: str | None


def load_settings() -> Settings:
    return Settings(
        documents_path=os.getenv("DOCUMENTS_PATH", "data/documents"),
        chroma_path=os.getenv("CHROMA_PATH", "data/chroma"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "documents"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "60")),
        top_k=int(os.getenv("TOP_K", "5")),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_model=os.getenv("GEMINI_MODEL"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )


settings = load_settings()
