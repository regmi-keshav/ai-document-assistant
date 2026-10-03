from dotenv import load_dotenv

from app.llm.factory import create_llm_provider
from app.prompts.system import DOCUMENT_ASSISTANT_SYSTEM_PROMPT
from app.tools.document_search import DocumentSearchTool

load_dotenv()


def main() -> None:
    llm = create_llm_provider()
    document_search = DocumentSearchTool()

    user_query = input("Ask a question: ").strip()

    if not user_query:
        print("No question provided.")
        return

    response = llm.generate(
        prompt=user_query,
        system_prompt=DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
        temperature=0.2,
        top_p=0.9,
        tools=[document_search],
    )

    print(response)


if __name__ == "__main__":
    main()
