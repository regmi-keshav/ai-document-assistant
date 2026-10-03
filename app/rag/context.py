from app.rag.retriever import RetrievalResult


class ContextBuilder:
    """
    Builds LLM-ready context from retrieved document chunks.

    This class only formats retrieval results.
    It does not perform retrieval or call an LLM.
    """

    def build(
        self,
        results: list[RetrievalResult],
    ) -> str:
        if not results:
            return ""

        sections = []

        for index, result in enumerate(results, start=1):
            section = (
                f"[Document {index}]\n"
                f"Source: {result.source}\n"
                f"Page: {result.page}\n"
                f"Content:\n"
                f"{result.content.strip()}"
            )

            sections.append(section)

        return "\n\n".join(sections)