from google.genai import types

from app.tools.base import Tool


class DocumentSearchTool(Tool):

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search the user's documents for information relevant to a query. "
            "Use this tool when answering questions that require information "
            "from the user's document collection."
        )

    def execute(self, query: str) -> list[dict]:
        # Temporary implementation for testing tool calling.
        # This will later be replaced by the RAG retriever.
        return [
            {
                "title": "RAG Introduction",
                "content": (
                    "Retrieval-Augmented Generation (RAG) combines information "
                    "retrieval with language generation by providing relevant "
                    "external context to a language model."
                ),
            }
        ]

    def declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="The information to search for in the documents.",
                    )
                },
                required=["query"],
            ),
        )