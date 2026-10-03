from pathlib import Path

from pypdf import PdfReader

from app.rag.cleaner import TextCleaner


class PDFDocumentLoader:
    """Load PDF documents while preserving page metadata."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.cleaner = TextCleaner()

    def load(self) -> list[dict]:
        documents = []

        for pdf_path in sorted(self.directory.glob("*.pdf")):
            reader = PdfReader(pdf_path)
            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""

                if self.cleaner.is_reference_heading(raw_text):
                    break

                text = self.cleaner.clean(raw_text)

                if text:
                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                        }
                    )

            if pages:
                documents.append(
                    {
                        "source": pdf_path.name,
                        "pages": _drop_repeated_lines(pages),
                    }
                )

        return documents


def _drop_repeated_lines(pages: list[dict]) -> list[dict]:
    """Remove running headers/footers repeated across many pages."""

    if len(pages) < 2:
        return pages

    counts: dict[str, int] = {}

    for page in pages:
        seen_on_page = set()

        for line in page["text"].splitlines():
            normalized = " ".join(line.split()).strip()

            if len(normalized) < 4:
                continue

            seen_on_page.add(normalized)

        for line in seen_on_page:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, len(pages) // 2)
    repeated = {line for line, count in counts.items() if count >= threshold}

    for page in pages:
        kept = [
            line
            for line in page["text"].splitlines()
            if " ".join(line.split()).strip() not in repeated
        ]
        page["text"] = "\n".join(kept).strip()

    return pages