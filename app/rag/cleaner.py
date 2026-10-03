import re


class TextCleaner:
    """Clean common artifacts produced by PDF text extraction."""

    REFERENCE_HEADINGS = {
        "REFERENCES",
        "BIBLIOGRAPHY",
        "REFERENCES AND NOTES",
        "LITERATURE CITED",
    }

    LIGATURES = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }

    def clean(self, text: str) -> str:
        # Rejoin words split across a line break.
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)

        # Restore PDF ligature/quote extraction artifacts.
        for bad, good in self.LIGATURES.items():
            text = text.replace(bad, good)

        # Normalize whitespace while preserving the text.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text)

        return text.strip()

    def is_reference_heading(self, text: str) -> bool:
        """
        Determine whether a page contains a standalone
        academic reference-section heading.
        """

        lines = [
            line.strip().upper()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:
            # Remove common numbering such as:
            # "VII. REFERENCES"
            # "8 REFERENCES"
            normalized = re.sub(
                r"^(?:[IVXLCDM]+|\d+)[\.\)]?\s+",
                "",
                line,
            )

            if normalized in self.REFERENCE_HEADINGS:
                return True

        return False