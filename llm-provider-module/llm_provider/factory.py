from __future__ import annotations

import os

from .base import LLMProvider
from .bedrock import BedrockProvider
from .ollama import OllamaProvider
from .openapi import OpenAPIProvider

import os
from llm_provider.bedrock import BedrockProvider
from llm_provider.openapi import OpenAPIProvider
from llm_provider.ollama import OllamaProvider


def build_llm_provider():
    provider = os.getenv("LLM_PROVIDER")

    if not provider:
        raise ValueError("LLM_PROVIDER must be set")

    provider = provider.lower()

    if provider == "bedrock":
        return BedrockProvider(
            model_name=os.getenv("BEDROCK_MODEL_ID")
        )

    elif provider == "openapi":
        return OpenAPIProvider(
            base_url=os.getenv("OPENAPI_BASE_URL"),
            api_key=os.getenv("OPENAPI_API_KEY"),
            model_name=os.getenv("OPENAPI_MODEL")
        )

    elif provider == "ollama":
        return OllamaProvider(
            model_name=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL")
        )

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

# def build_llm_provider() -> LLMProvider:
#     provider_name = os.getenv("LLM_PROVIDER", "bedrock").lower()

#     if provider_name == "bedrock":
#         return BedrockProvider(
#             model_name=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
#         )

#     if provider_name == "ollama":
#         return OllamaProvider(
#             model_name=os.getenv("OLLAMA_MODEL", "llama3.1")
#         )

#     if provider_name in {"openapi", "openai", "openai-compatible"}:
#         return OpenAPIProvider(
#             model_name=os.getenv("OPENAPI_MODEL", "gpt-4o-mini")
#         )

#     raise ValueError(f"Unsupported LLM_PROVIDER: {provider_name}")
