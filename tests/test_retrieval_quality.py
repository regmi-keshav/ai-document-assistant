"""
Retrieval quality evaluation.

Measures retrieval over the existing pipeline:

    PDFs -> loader -> cleaner -> TextChunker -> embeddings
    -> ChromaDB -> Retriever

Evaluation design
-----------------
Each query carries *relevance judgments* over chunk content rather
than a single expected source filename. A retrieved chunk is:

    RELEVANT     -- satisfies at least one answer-bearing matcher
    RELATED      -- mentions the topic but is not answer-bearing
    NOT_RELEVANT -- everything else

A matcher may optionally constrain the source and/or require phrases
to appear in the chunk text (case-insensitive substring match).

All matchers below are evaluation data only. They are NOT referenced
by the retriever, vector store, embeddings, loader, chunker, or any
other production code.
"""

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_PATH
from app.rag.chunker import TextChunker
from app.rag.embeddings import EmbeddingService
from app.rag.loader import PDFDocumentLoader
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore

TOP_K_EVAL = 5

EVAL_QUERIES = [
    {
        "query": "What is Retrieval-Augmented Generation?",
        "matchers": [
            # Direct definitional passages.
            {"contains": ["retrieval-augmented generation (rag)"]},
            {"contains": ["rag is used because"]},
            {"contains": ["recipe for retrieval-augmented generation"],
             "source": "rag_foundational_paper.pdf"},
            {"contains_all": ["hybrid parametric", "retrieval-based"],
             "source": "rag_foundational_paper.pdf"},
        ],
        "related_contains": ["retrieval-augmented generation", "rag"],
    },
    {
        "query": "What are the benefits of RAG?",
        "matchers": [
            {"contains": ["benefits of"]},
            {"contains": ["excels in"]},
            {"contains": ["enhances the accuracy and credibility"]},
            {"contains": ["enriches the model responses with external data"]},
            {"contains": ["real-time knowledge updates"]},
        ],
        "related_contains": ["benefit", "advantage", "improve", "enhance"],
    },
    {
        "query": "How does RAG differ from fine-tuning?",
        "matchers": [
            {"contains_all": ["fine-tuning", "rag"]},
            {"contains_all": ["fine-tuning", "retrieval-augmented"]},
        ],
        "related_contains": ["fine-tuning", "retrieval-augmented"],
    },
    {
        "query": "What is the role of retrieval in RAG?",
        "matchers": [
            {"contains": ["retrieval mechanism"]},
            {"contains": ["retrieval phase"]},
            {"contains": ["retrieve text document"]},
            {"contains": ["retrieving relevant"]},
            {"contains": ["retrieval component"]},
        ],
        "related_contains": ["retrieval"],
    },
]


# ----------------------------------------------------------------------
# Matching helpers
# ----------------------------------------------------------------------

def _match_contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def chunk_is_relevant(chunk_source: str, chunk_text: str, query_case: dict) -> bool:
    """True when the chunk satisfies at least one relevance matcher."""

    for matcher in query_case["matchers"]:
        if "source" in matcher and matcher["source"] != chunk_source:
            continue

        if "contains" in matcher:
            if any(_match_contains(chunk_text, p) for p in matcher["contains"]):
                return True

        if "contains_all" in matcher:
            if all(_match_contains(chunk_text, p) for p in matcher["contains_all"]):
                return True

    return False


def chunk_is_related(chunk_text: str, query_case: dict) -> bool:
    return any(
        _match_contains(chunk_text, p)
        for p in query_case.get("related_contains", [])
    )


def judge(chunk_source: str, chunk_text: str, query_case: dict) -> str:
    if chunk_is_relevant(chunk_source, chunk_text, query_case):
        return "RELEVANT"

    if chunk_is_related(chunk_text, query_case):
        return "RELATED"

    return "NOT_RELEVANT"


# ----------------------------------------------------------------------
# Ingestion (reuses production components, rebuilds the collection)
# ----------------------------------------------------------------------

def build_vector_store() -> tuple[list[dict], ChromaVectorStore]:
    loader = PDFDocumentLoader(DOCUMENTS_PATH)
    documents = loader.load()
    print(f"Loaded documents: {len(documents)}")

    chunker = TextChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(documents)
    print(f"Generated chunks: {len(chunks)}")

    embedding_service = EmbeddingService()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed(texts)
    print(f"Generated embeddings: {len(embeddings)}")

    vector_store = ChromaVectorStore()
    vector_store.clear()  # never evaluate against stale chunks

    ids = [
        f"{chunk['source']}-{chunk['page']}-{index}"
        for index, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": ids[index],
        }
        for index, chunk in enumerate(chunks)
    ]

    vector_store.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Documents stored in ChromaDB: {vector_store.count()}")

    return chunks, vector_store


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def main() -> None:
    chunks, vector_store = build_vector_store()

    retriever = Retriever(
        embedding_service=EmbeddingService(),
        vector_store=vector_store,
        top_k=TOP_K_EVAL,
    )

    # Corpus-level relevance counts make Recall@k well-defined.
    relevant_in_corpus = {}

    for case in EVAL_QUERIES:
        relevant_in_corpus[case["query"]] = sum(
            1
            for chunk in chunks
            if chunk_is_relevant(chunk["source"], chunk["text"], case)
        )

    recall_sum = {1: 0.0, 3: 0.0, 5: 0.0}
    hit_sum = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    judgement_counts = {"RELEVANT": 0, "RELATED": 0, "NOT_RELEVANT": 0}

    for case in EVAL_QUERIES:
        query = case["query"]
        results = retriever.search(query)

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        print("\nRANK  SOURCE                              PAGE  DISTANCE  JUDGMENT")
        print("-" * 70)

        relevant_ranks = []

        for index, result in enumerate(results, start=1):
            judgment = judge(result.source, result.content, case)
            judgement_counts[judgment] += 1

            if judgment == "RELEVANT":
                relevant_ranks.append(index)

            print(
                f"{index:<5} {result.source:<40} {result.page:<5} "
                f"{result.distance:<9.4f} {judgment}"
            )

            snippet = " ".join(result.content.split())[:160]
            print(f"      {snippet}")

        total_relevant = relevant_in_corpus[query]
        denom = max(total_relevant, 1)

        for k in (1, 3, 5):
            found = sum(1 for r in relevant_ranks if r <= k)
            recall_sum[k] += found / denom
            hit_sum[k] += int(found > 0)

        rr_sum += 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

        print(f"\nRelevant chunks in corpus: {total_relevant}")
        print(f"Relevant chunks in top-{TOP_K_EVAL}: {len(relevant_ranks)}")
        if relevant_ranks:
            print(f"First relevant result at rank: {relevant_ranks[0]}")
        else:
            print("No relevant result in top results")

    total = len(EVAL_QUERIES)

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"\nQueries evaluated: {total}\n")
    print(f"Recall@1: {recall_sum[1] / total * 100:.1f}%")
    print(f"Recall@3: {recall_sum[3] / total * 100:.1f}%")
    print(f"Recall@5: {recall_sum[5] / total * 100:.1f}%")
    print(f"MRR: {rr_sum / total:.3f}")
    print(f"\nHit@1: {hit_sum[1] / total * 100:.1f}%")
    print(f"Hit@3: {hit_sum[3] / total * 100:.1f}%")
    print(f"Hit@5: {hit_sum[5] / total * 100:.1f}%")
    print(f"\nJudgment distribution across all ranked results: {judgement_counts}")


if __name__ == "__main__":
    main()
