import os

from dotenv import load_dotenv
from google import genai

from app.llm.base import LLMProvider


load_dotenv()


class GeminiProvider(LLMProvider):

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        if not self.model:
            raise ValueError("GEMINI_MODEL is not set")

        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:

        config = {
            "temperature": temperature,
            "top_p": top_p,
        }

        if system_prompt:
            config["system_instruction"] = system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        return response.text