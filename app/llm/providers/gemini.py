import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

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
        response_schema: dict | None = None,
        tools: list | None = None,
    ) -> str:

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
        )

        if system_prompt:
            config.system_instruction = system_prompt

        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        # Explicit (manual) function calling: the SDK must not execute
        # Python callables on its own. We handle tool execution ourselves.
        if tools:
            config.tools = [
                types.Tool(
                    function_declarations=[tool.declaration() for tool in tools]
                )
            ]
            config.automatic_function_calling = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        tools_by_name = {tool.name: tool for tool in tools or []}
        contents: list = [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Let the model request tool calls, execute them, and feed the
        # results back until it produces a final text answer.
        for _ in range(5):
            function_calls = response.function_calls
            if not function_calls:
                return response.text

            # Keep the model's tool-call message in the conversation.
            contents.append(response.candidates[0].content)

            response_parts = []
            for fc in function_calls:
                tool = tools_by_name.get(fc.name)
                if tool is None:
                    result = {"error": f"Unknown tool: {fc.name}"}
                else:
                    try:
                        result = tool.execute(**(fc.args or {}))
                    except Exception as exc:
                        result = {"error": str(exc)}

                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )

            contents.append(
                types.Content(role="user", parts=response_parts)
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        return response.text
