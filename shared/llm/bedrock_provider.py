import json
import os

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

from shared.llm.base import GenerateResult, ToolCall, ToolUseResult, Usage

# Newer Claude models on Bedrock require the cross-region inference profile ID
# (the "us." prefix), not the bare model ID - verified working 2026-07-13.
DEFAULT_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"
)
DEFAULT_EMBEDDING_MODEL_ID = os.environ.get(
    "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)


def _to_converse_messages(messages: list[dict]) -> list[dict]:
    """Translate the shared normalized message list (see `ToolUseResult`)
    into Converse's message format. Converse has no "tool" role - each
    `{"role": "tool", ...}` entry becomes a `toolResult` content block on a
    "user" turn, and consecutive tool messages (one per call from the same
    agent turn) are merged into a single user turn, since Converse requires
    strict user/assistant alternation.
    """
    converse_messages = []
    i = 0
    while i < len(messages):
        message = messages[i]

        if message["role"] == "tool":
            tool_results = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": messages[i]["tool_call_id"],
                            "content": [{"text": messages[i]["content"]}],
                        }
                    }
                )
                i += 1
            converse_messages.append({"role": "user", "content": tool_results})
            continue

        if message["role"] == "assistant":
            content = []
            if message.get("content"):
                content.append({"text": message["content"]})
            for call in message.get("tool_calls") or []:
                content.append(
                    {
                        "toolUse": {
                            "toolUseId": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    }
                )
            converse_messages.append({"role": "assistant", "content": content})
        else:
            converse_messages.append(
                {"role": message["role"], "content": [{"text": message["content"]}]}
            )
        i += 1

    return converse_messages


class BedrockProvider:
    """Claude via Amazon Bedrock, using the provider-agnostic Converse API
    so the request/response shape stays consistent across Bedrock model
    families if we ever swap Claude for another Bedrock-hosted model.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        region: str | None = None,
    ):
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.client = boto3.client(
            "bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1")
        )

    def generate(
        self,
        user_message: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> GenerateResult:
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": user_message}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(
                f"Bedrock request failed ({code}): {e}. Check AWS_REGION, your "
                f"credentials, and that model access for '{self.model_id}' is "
                f"enabled in the Bedrock console for this account/region."
            ) from e

        content_block = response["output"]["message"]["content"][0]
        if "text" not in content_block:
            raise RuntimeError(
                f"Bedrock returned no text content (got: {list(content_block.keys())}). "
                f"Inspect `response` directly to see what came back."
            )

        usage = Usage(
            input_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
        )
        return GenerateResult(text=content_block["text"], usage=usage, model=self.model_id)

    def generate_structured(
        self,
        user_message: str,
        response_model: type[BaseModel],
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        tool_name = response_model.__name__

        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": user_message}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool_name,
                            "description": f"Extracts structured data matching the {tool_name} schema",
                            "inputSchema": {"json": schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": tool_name}},
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(
                f"Bedrock request failed ({code}): {e}. Check AWS_REGION, your "
                f"credentials, and that model access for '{self.model_id}' is "
                f"enabled in the Bedrock console for this account/region."
            ) from e

        content_blocks = response["output"]["message"]["content"]
        tool_use_block = None
        for block in content_blocks:
            if "toolUse" in block:
                tool_use_block = block["toolUse"]
                break

        if not tool_use_block:
            raise RuntimeError(
                f"Bedrock did not return a tool use block for {tool_name}. "
                f"Response: {response['output']['message']['content']}"
            )

        try:
            tool_input = tool_use_block["input"]
            return response_model.model_validate(tool_input)
        except ValidationError as e:
            raise RuntimeError(
                f"Bedrock returned tool input that failed schema validation: {e}"
            ) from e

    def generate_with_tools(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str = "",
        messages: list[dict] | None = None,
    ) -> ToolUseResult:
        if messages is None:
            messages = [{"role": "user", "content": user_message}]

        kwargs = {
            "modelId": self.model_id,
            "messages": _to_converse_messages(messages),
            # Bedrock Converse falls back to a small default when this is omitted -
            # too small to finish emitting multiple parallel tool calls' JSON, which
            # truncates mid-argument and corrupts the parsed tool input. Kept generous
            # and not exposed as a param since callers shouldn't need to think about it.
            "inferenceConfig": {"maxTokens": 4096},
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "inputSchema": {"json": tool["input_schema"]},
                        }
                    }
                    for tool in tools
                ],
                "toolChoice": {"auto": {}},
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(
                f"Bedrock request failed ({code}): {e}. Check AWS_REGION, your "
                f"credentials, and that model access for '{self.model_id}' is "
                f"enabled in the Bedrock console for this account/region."
            ) from e

        content_blocks = response["output"]["message"]["content"]
        text = " ".join(block["text"] for block in content_blocks if "text" in block)
        tool_calls = [
            ToolCall(
                id=block["toolUse"]["toolUseId"],
                name=block["toolUse"]["name"],
                arguments=block["toolUse"]["input"],
            )
            for block in content_blocks
            if "toolUse" in block
        ]

        assistant_message = {
            "role": "assistant",
            "content": text or None,
            "tool_calls": tool_calls,
        }
        usage = Usage(
            input_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
        )
        return ToolUseResult(
            tool_calls=tool_calls,
            text=text,
            messages=messages + [assistant_message],
            usage=usage,
            model=self.model_id,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan Embed takes one text per invoke_model call - no native batch endpoint.
        vectors = []
        for text in texts:
            try:
                response = self.client.invoke_model(
                    modelId=self.embedding_model_id,
                    body=json.dumps({"inputText": text}),
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "Unknown")
                raise RuntimeError(
                    f"Bedrock embedding request failed ({code}): {e}. Check AWS_REGION, "
                    f"your credentials, and that model access for "
                    f"'{self.embedding_model_id}' is enabled in the Bedrock console."
                ) from e

            body = json.loads(response["body"].read())
            vectors.append(body["embedding"])
        return vectors
