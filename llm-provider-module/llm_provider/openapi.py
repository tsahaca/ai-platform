from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from .base import LLMProvider
from .schemas import LLMRequest, LLMResponse


class OpenAPIProvider(LLMProvider):
    """
    OpenAI-compatible chat completions provider.

    Works with endpoints that expose an OpenAI-style API, for example:
    - OpenAI
    - Azure OpenAI (with a compatible base URL/proxy)
    - self-hosted gateways like LiteLLM or vLLM with OpenAI compatibility
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(model_name=model_name)
        self.base_url = (base_url or os.getenv("OPENAPI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAPI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAPI_API_KEY or OPENAI_API_KEY must be set for OpenAPIProvider")

    def generate(self, request: LLMRequest) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": request.system_prompt})

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]["message"]
            text = choice.get("content", "") or ""

            return LLMResponse(
                text=text,
                model=data.get("model", self.model_name),
                provider="openapi",
                usage=data.get("usage", {}),
                raw=data,
            )
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            raise RuntimeError(f"OpenAPI generate failed: {exc}") from exc
