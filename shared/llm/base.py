from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class GenerateResult:
    text: str
    usage: Usage
    model: str


class LLMProvider(Protocol):
    """Provider-agnostic interface every lab notebook should call through.

    Lab content should only ever call `generate()`/`embed()` on whatever
    provider the factory hands back - it should never import boto3 or the
    openai SDK directly. That's what keeps the teaching content unchanged
    when the underlying provider switches from Bedrock to Azure OpenAI.
    """

    def generate(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> GenerateResult:
        ...

    def generate_structured(
        self,
        user_message: str,
        response_model: type[BaseModel],
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> BaseModel:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
