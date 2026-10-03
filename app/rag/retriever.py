from dataclasses import dataclass

from app.config import TOP_K
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import ChromaVectorStore


@dataclass
class RetrievalResult:
    id: str
    content: str
    source: str
    page: int
    distance: float


class Retriever:
    """
    Retrieves the most relevant document chunks for a query.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        top_k: int = TOP_K,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k

    def search(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        if not query or not query.strip():
            return []

        query = query.strip()

        query_embedding = self.embedding_service.embed(
            [query]
        )[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        return [
            RetrievalResult(
                id=result["id"],
                content=result["content"],
                source=result["source"],
                page=int(result["page"]),
                distance=result["distance"],
            )
            for result in results
        ]