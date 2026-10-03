import os


DOCUMENTS_PATH = os.getenv(
    "DOCUMENTS_PATH",
    "data/documents",
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "data/chroma",
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "documents",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "300")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "60")
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)