import os

from shared.llm.base import LLMProvider

_PROVIDERS = {"bedrock", "azure_openai"}
_cached_client: LLMProvider | None = None
_cached_provider: str | None = None


def get_client(provider: str | None = None) -> LLMProvider:
    """Returns the active LLM provider, chosen by the PROVIDER env var
    (bedrock | azure_openai). This is the ONLY place that should know which
    provider is active - lab notebooks call get_client().generate(...) and
    never import boto3 or the openai SDK directly.

    The underlying client is built once per provider and reused - notebooks
    can call this repeatedly across cells without opening a new connection
    each time.
    """
    global _cached_client, _cached_provider

    provider = provider or os.environ.get("PROVIDER", "bedrock")
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown PROVIDER '{provider}'. Expected one of: {_PROVIDERS}")

    if _cached_client is not None and _cached_provider == provider:
        return _cached_client

    if provider == "bedrock":
        from shared.llm.bedrock_provider import BedrockProvider

        _cached_client = BedrockProvider()
    else:
        from shared.llm.azure_openai_provider import AzureOpenAIProvider

        _cached_client = AzureOpenAIProvider()

    _cached_provider = provider
    return _cached_client
