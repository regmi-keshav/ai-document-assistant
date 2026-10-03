from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore


def main():

    embedding_service = EmbeddingService()

    vector_store = ChromaVectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=settings.top_k,
    )

    queries = [
        "What is Retrieval-Augmented Generation?",
        "What are the benefits of RAG?",
        "How does RAG differ from fine-tuning?",
        "What is the role of retrieval in RAG?",
    ]

    for query in queries:

        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = retriever.search(query)

        for index, result in enumerate(results, start=1):

            print(f"\n--- RESULT {index} ---")

            print(
                f"Distance: {result.distance:.4f}"
            )

            print(
                f"Source: {result.source}"
            )

            print(
                f"Page: {result.page}"
            )

            print(
                f"Text: {result.content[:500]}"
            )

        print()


if __name__ == "__main__":
    main()