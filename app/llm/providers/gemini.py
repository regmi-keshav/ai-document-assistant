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

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text