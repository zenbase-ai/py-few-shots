from typing import Any, Awaitable, Callable

from few_shots.types import Vector
from .base import AsyncEmbed, Embed


class LiteLLMEmbed(Embed):
    def __init__(self, embedder: Callable[[str], Any]):
        self.embedder = embedder

    def __call__(self, inputs: list[str]) -> list[Vector]:
        response = self.embedder(inputs)
        return [r["vector"] for r in response["data"]]


class AsyncLiteLLMEmbed(AsyncEmbed):
    def __init__(self, embedder: Callable[[str], Awaitable[Any]]):
        self.embedder = embedder

    async def __call__(self, inputs: list[str]) -> list[Vector]:
        response = await self.embedder(inputs)
        return [r["vector"] for r in response["data"]]
