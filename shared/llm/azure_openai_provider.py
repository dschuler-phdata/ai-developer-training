import os

from openai import APIError, AzureOpenAI
from pydantic import BaseModel

from shared.llm.base import GenerateResult, Usage
from shared.utils.env import require_env

DEFAULT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


class AzureOpenAIProvider:
    """GPT-4o via Azure OpenAI Service. Same interface as BedrockProvider -
    lab content should be able to swap providers without any code changes.
    """

    def __init__(self, deployment: str = DEFAULT_DEPLOYMENT):
        self.deployment = deployment
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
        
        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return GenerateResult(text=parsed, usage=usage, model=self.deployment)
