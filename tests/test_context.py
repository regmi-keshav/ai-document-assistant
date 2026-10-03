from app.rag.context import ContextBuilder
from app.rag.retriever import RetrievalResult


def main():
    results = [
        RetrievalResult(
            id="chunk-001",
            content="RAG combines retrieval with language model generation.",
            source="example.pdf",
            page=3,
            distance=0.25,
        ),
        RetrievalResult(
            id="chunk-002",
            content="The retrieval component searches an external knowledge base.",
            source="another.pdf",
            page=7,
            distance=0.31,
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(results)

    print("=" * 80)
    print("GENERATED CONTEXT")
    print("=" * 80)
    print(context)
    print("=" * 80)


if __name__ == "__main__":
    main()