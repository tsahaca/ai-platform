from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_provider import LLMMessage, LLMRequest, build_llm_provider

app = FastAPI(title="LLM Provider Example", version="0.1.0")


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str


class GenerateRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = "You are a helpful assistant."
    temperature: float = 0.2
    max_tokens: int = 256


class GenerateResponse(BaseModel):
    provider: str
    model: str
    text: str
    usage: dict


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        provider = build_llm_provider()
        llm_request = LLMRequest(
            messages=[LLMMessage(role=m.role, content=m.content) for m in req.messages],
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        result = provider.generate(llm_request)
        return GenerateResponse(
            provider=result.provider,
            model=result.model,
            text=result.text,
            usage=result.usage,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
