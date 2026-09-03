from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """Generate a text response from the LLM."""
        pass