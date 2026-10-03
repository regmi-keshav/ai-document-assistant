from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """
    Generates vector embeddings for text.

    The rest of the application does not need to know
    which embedding model is being used.
    """

    def __init__(self, model_name: str = settings.embedding_model):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of texts into embeddings.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """

        return self.model.get_embedding_dimension()