from dotenv import load_dotenv

from app.llm.factory import create_llm_provider


load_dotenv()


def main():
    llm = create_llm_provider()

    response = llm.generate(
        "Explain Retrieval-Augmented Generation in two sentences."
    )

    print(response)


if __name__ == "__main__":
    main()