from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .base import LLMProvider
from .schemas import LLMMessage, LLMRequest, LLMResponse


class BedrockProvider(LLMProvider):
    def __init__(self, model_name: str, region_name: Optional[str] = None) -> None:
        super().__init__(model_name=model_name)
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.tools:
            raise NotImplementedError(
                "This sample Bedrock provider implements text generation only. "
                "Add tool schema translation if you want Bedrock tool use."
            )

        try:
            system_blocks: List[Dict[str, str]] = []
            if request.system_prompt:
                system_blocks.append({"text": request.system_prompt})

            messages = [self._to_bedrock_message(msg) for msg in request.messages]

            response = self.client.converse(
                modelId=self.model_name,
                system=system_blocks,
                messages=messages,
                inferenceConfig={
                    "temperature": request.temperature,
                    "maxTokens": request.max_tokens,
                },
            )

            output_message = response["output"]["message"]
            text_parts = [
                block.get("text", "")
                for block in output_message.get("content", [])
                if "text" in block
            ]
            text = "\n".join(part for part in text_parts if part)

            return LLMResponse(
                text=text,
                model=self.model_name,
                provider="bedrock",
                usage=response.get("usage", {}),
                raw=response,
            )
        except (ClientError, BotoCoreError, KeyError) as exc:
            raise RuntimeError(f"Bedrock generate failed: {exc}") from exc

    @staticmethod
    def _to_bedrock_message(msg: LLMMessage) -> Dict[str, Any]:
        return {
            "role": msg.role,
            "content": [{"text": msg.content}],
        }
