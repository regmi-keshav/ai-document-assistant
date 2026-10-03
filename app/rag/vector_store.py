import chromadb

from app.config import CHROMA_COLLECTION, CHROMA_PATH


class ChromaVectorStore:
    """
    ChromaDB-backed vector store.

    ChromaDB-specific operations are isolated inside this class.
    """

    def __init__(
        self,
        path: str = CHROMA_PATH,
        collection_name: str = CHROMA_COLLECTION,
    ):
        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output = []

        for document, metadata, distance, document_id in zip(
            documents,
            metadatas,
            distances,
            ids,
        ):
            output.append(
                {
                    "id": document_id,
                    "content": document,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "distance": float(distance),
                }
            )

        return output

    def clear(self) -> None:
        """Drop all stored chunks by recreating the collection."""

        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )

    def count(self) -> int:
        return self.collection.count()