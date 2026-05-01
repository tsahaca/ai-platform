from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import LLMRequest, LLMResponse


class LLMProvider(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
