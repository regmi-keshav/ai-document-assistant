from dotenv import load_dotenv

from app.llm.factory import create_llm_provider


load_dotenv()


SYSTEM_PROMPT = """
You are an AI Document Research Assistant.

Your job is to answer questions clearly, accurately, and concisely.

When answering questions about documents:
- Base your answer on the provided evidence.
- Do not invent facts.
- If the available information is insufficient, clearly say so.
- Prefer precise and factual explanations.
"""


def main():
    llm = create_llm_provider()

    response = llm.generate(
        prompt="What is Retrieval-Augmented Generation and why is it useful?",
        system_prompt=SYSTEM_PROMPT,
        temperature=0.2,
        top_p=0.9,
    )

    print(response)


if __name__ == "__main__":
    main()