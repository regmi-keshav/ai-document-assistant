import re


class TextChunker:
    """Split document text into overlapping, boundary-aware chunks."""

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 60,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> list[str]:
        """Split text using progressively smaller boundaries."""

        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            current = ""

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                if len(current) + len(sentence) + 1 <= self.chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)

                    current = sentence

            if current:
                chunks.append(current)

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""

        if not chunks:
            return []

        result = [chunks[0]]

        for chunk in chunks[1:]:
            previous = result[-1]

            overlap = previous[-self.chunk_overlap:]

            # Avoid starting the overlap in the middle of a word.
            space = overlap.find(" ")

            if space != -1:
                overlap = overlap[space + 1:]

            combined = f"{overlap} {chunk}".strip()

            if len(combined) <= self.chunk_size:
                result.append(combined)
            else:
                result.append(chunk)

        return result

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        chunks = []

        for document in documents:
            source = document["source"]

            for page in document["pages"]:
                page_chunks = self._split_text(page["text"])
                page_chunks = self._add_overlap(page_chunks)

                for chunk in page_chunks:
                    chunks.append(
                        {
                            "text": chunk,
                            "source": source,
                            "page": page["page"],
                        }
                    )

        return chunks