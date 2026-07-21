import json
import os

from openai import APIError, AzureOpenAI
from pydantic import BaseModel

from shared.llm.base import GenerateResult, ToolCall, ToolUseResult, Usage
from shared.utils.env import require_env

DEFAULT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
DEFAULT_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Translate the shared normalized message list (see `ToolUseResult`)
    into the Chat Completions wire format - mainly, re-serializing each
    `ToolCall`'s `arguments` dict back into the JSON-string shape
    `tool_calls[].function.arguments` requires on the way to the API.
    """
    api_messages = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            api_messages.append(message)
            continue

        api_messages.append(
            {
                "role": "assistant",
                "content": message["content"],
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
    return api_messages


class AzureOpenAIProvider:
    """GPT-4o via Azure OpenAI Service. Same interface as BedrockProvider -
    lab content should be able to swap providers without any code changes.
    """

    def __init__(
        self,
        deployment: str = DEFAULT_DEPLOYMENT,
        embedding_deployment: str = DEFAULT_EMBEDDING_DEPLOYMENT,
    ):
        self.deployment = deployment
        self.embedding_deployment = embedding_deployment
        self.client = AzureOpenAI(
            api_key=require_env("AZURE_OPENAI_API_KEY"),
            azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    def generate(
        self,
        user_message: str,
        system_prompt: str = "",
    ) -> GenerateResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
            )
        except APIError as e:
            raise RuntimeError(
                f"Azure OpenAI request failed ({e.__class__.__name__}): {e}. "
                f"Check your AZURE_OPENAI_* values in .env and that the "
                f"'{self.deployment}' deployment exists in your Azure resource."
            ) from e

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                "Azure OpenAI returned no text content (the response may have "
                "been blocked by a content filter, or the model tried to call "
                "a tool). Inspect `response` directly to see what came back."
            )

        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return GenerateResult(text=content, usage=usage, model=self.deployment)

    def generate_structured(
        self,
        user_message: str,
        response_model: type[BaseModel],
        system_prompt: str = "",
    ) -> BaseModel:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=messages,
                response_format=response_model,
            )
        except APIError as e:
            raise RuntimeError(
                f"Azure OpenAI request failed ({e.__class__.__name__}): {e}. "
                f"Check your AZURE_OPENAI_* values in .env and that the "
                f"'{self.deployment}' deployment exists in your Azure resource."
            ) from e

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                "Azure OpenAI failed to parse the response as the requested schema. "
                "The model may have returned incomplete or invalid structured data."
            )
        return parsed

    def generate_with_tools(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str = "",
        messages: list[dict] | None = None,
    ) -> ToolUseResult:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

        api_messages = _to_openai_messages(messages)
        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=api_messages,
                tools=api_tools,
                tool_choice="auto",
            )
        except APIError as e:
            raise RuntimeError(
                f"Azure OpenAI request failed ({e.__class__.__name__}): {e}. "
                f"Check your AZURE_OPENAI_* values in .env and that the "
                f"'{self.deployment}' deployment exists in your Azure resource."
            ) from e

        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]

        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,
        }
        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return ToolUseResult(
            tool_calls=tool_calls,
            text=message.content or "",
            messages=messages + [assistant_message],
            usage=usage,
            model=self.deployment,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment, input=texts
            )
        except APIError as e:
            raise RuntimeError(
                f"Azure OpenAI embedding request failed ({e.__class__.__name__}): {e}. "
                f"Check your AZURE_OPENAI_* values in .env and that the "
                f"'{self.embedding_deployment}' embedding deployment exists in your "
                f"Azure resource."
            ) from e

        return [item.embedding for item in response.data]
