from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from .base import LLMProvider
from .schemas import LLMRequest, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: Optional[str] = None, timeout: int = 120) -> None:
        super().__init__(model_name=model_name)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            }

            if request.system_prompt:
                payload["messages"].insert(0, {"role": "system", "content": request.system_prompt})

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            usage = {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "total_duration": data.get("total_duration"),
            }

            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=self.model_name,
                provider="ollama",
                usage=usage,
                raw=data,
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc
