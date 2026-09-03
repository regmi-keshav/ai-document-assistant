from dotenv import load_dotenv

from app.llm.factory import create_llm_provider


load_dotenv()


SYSTEM_PROMPT = """
You are an AI Document Research Assistant.

Answer questions accurately and concisely.
Do not invent facts.
"""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["answer", "confidence", "sources"],
}


def main():
    llm = create_llm_provider()

    response = llm.generate(
        prompt="What is Retrieval-Augmented Generation?",
        system_prompt=SYSTEM_PROMPT,
        temperature=0.2,
        top_p=0.9,
        response_schema=RESPONSE_SCHEMA,
    )

    print(response)


if __name__ == "__main__":
    main()