from dotenv import load_dotenv

from app.llm.factory import create_llm_provider
from app.tools.document_search import DocumentSearchTool


load_dotenv()


SYSTEM_PROMPT = """
You are an AI Document Research Assistant.

Answer questions using the user's documents when relevant.

If the question requires information from the documents,
use the search_documents tool.

Do not invent information.
"""


def main():
    llm = create_llm_provider()

    document_search = DocumentSearchTool()

    response = llm.generate(
        prompt=(
            "What does the document say about "
            "Retrieval-Augmented Generation?"
        ),
        system_prompt=SYSTEM_PROMPT,
        temperature=0.2,
        top_p=0.9,
        tools=[document_search],
    )

    print(response)


if __name__ == "__main__":
    main()