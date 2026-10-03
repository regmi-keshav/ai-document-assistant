from abc import ABC, abstractmethod


class Tool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name used by the LLM to identify the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        pass

    @abstractmethod
    def execute(self, **kwargs):
        """Execute the tool with the provided arguments."""
        pass