from dataclasses import dataclass, field
from typing import Any, Protocol

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


@dataclass
class ToolCall:
    """One tool invocation the model requested, normalized across providers."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolUseResult:
    """Result of one `generate_with_tools()` turn.

    `tool_calls` is empty when the model chose to respond in plain text
    instead of calling a tool - `text` holds that response in that case.

    `messages` is the running conversation so far, in one normalized shape
    used by both providers regardless of which one is active:
      - `{"role": "user", "content": "..."}`
      - `{"role": "assistant", "content": "..." | None, "tool_calls": [ToolCall, ...]}`
      - `{"role": "tool", "tool_call_id": "...", "content": "..."}`

    To continue the loop: execute each requested `ToolCall`, append one
    `{"role": "tool", "tool_call_id": call.id, "content": <result as a string>}`
    dict per call onto `result.messages`, then call `generate_with_tools`
    again passing that extended list as `messages`. Each provider translates
    this shape to/from its own native API format internally, so notebook
    code building the next turn never needs to branch on provider.
    """

    tool_calls: list[ToolCall]
    text: str
    messages: list[dict] = field(default_factory=list)
    usage: Usage = None
    model: str = ""


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

    def generate_with_tools(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str = "",
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> ToolUseResult:
        """One turn of a tool-calling conversation.

        `tools` is a provider-agnostic list of tool specs:
        `[{"name": ..., "description": ..., "input_schema": <JSON schema dict>}]`
        (e.g. `MyModel.model_json_schema()` for the last field).

        On the first call, leave `messages=None` - it's built from
        `system_prompt` + `user_message`. To continue the loop after
        executing the requested tools, pass the extended `messages` list
        from the previous `ToolUseResult` back in (see `ToolUseResult` for
        the exact shape) and `user_message` is ignored.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
