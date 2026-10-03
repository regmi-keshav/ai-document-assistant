from app.config import settings
from app.llm.base import LLMProvider
from app.llm.providers.gemini import GeminiProvider


def create_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        return GeminiProvider()

    raise ValueError(f"Unsupported LLM provider: {provider}")