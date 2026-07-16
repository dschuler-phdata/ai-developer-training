import json
import os

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

from shared.llm.base import GenerateResult, Usage

# Newer Claude models on Bedrock require the cross-region inference profile ID
# (the "us." prefix), not the bare model ID - verified working 2026-07-13.
DEFAULT_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"
)


class BedrockProvider:
    """Claude via Amazon Bedrock, using the provider-agnostic Converse API
    so the request/response shape stays consistent across Bedrock model
    families if we ever swap Claude for another Bedrock-hosted model.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str | None = None):
        self.model_id = model_id
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
        max_tokens: int = 1024,
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
