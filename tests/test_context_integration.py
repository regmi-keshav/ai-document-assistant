from app.config import settings
from app.rag.context import ContextBuilder
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore


def main():
    query = "What is Retrieval-Augmented Generation?"

    # Use the real application components.
    embedding_service = EmbeddingService()
    vector_store = ChromaVectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=settings.top_k,
    )

    context_builder = ContextBuilder()

    # Retrieve from the actual ChromaDB collection.
    results = retriever.search(query)

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    if not results:
        print("No results retrieved.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n--- RESULT {index} ---")
        print(f"Source: {result.source}")
        print(f"Page: {result.page}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Content: {result.content[:300]}")

    # Convert the actual retrieval results into LLM context.
    context = context_builder.build(results)

    print("\n" + "=" * 80)
    print("GENERATED CONTEXT")
    print("=" * 80)
    print(context)

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    assert context, "Context should not be empty."
    assert results, "Retriever should return at least one result."

    for result in results:
        assert result.source in context
        assert str(result.page) in context
        assert result.content.strip() in context

    print("PASS: All retrieved chunks are present in the generated context.")
    print("PASS: Source metadata is preserved.")
    print("PASS: Page metadata is preserved.")
    print("PASS: Retrieved content is preserved.")


if __name__ == "__main__":
    main()