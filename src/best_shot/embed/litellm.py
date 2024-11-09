from typing import Any, Awaitable, Callable

from best_shot.types import Embedding
from .base import AsyncEmbedder, Embedder


class LiteLLMEmbedder(Embedder):
    def __init__(self, embed: Callable[[str], Any]):
        self.embed = embed

    def __call__(self, inputs: list[str]) -> list[Embedding]:
        response = self.embed(inputs)
        return [r["embedding"] for r in response["data"]]


class AsyncLiteLLMEmbedder(AsyncEmbedder):
    def __init__(self, embed: Callable[[str], Awaitable[Any]]):
        self.embed = embed

    async def __call__(self, inputs: list[str]) -> list[Embedding]:
        response = await self.embed(inputs)
        return [r["embedding"] for r in response["data"]]
