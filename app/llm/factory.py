import os

from app.llm.base import LLMProvider
from app.llm.providers.gemini import GeminiProvider


def create_llm_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        return GeminiProvider()

    raise ValueError(f"Unsupported LLM provider: {provider}")