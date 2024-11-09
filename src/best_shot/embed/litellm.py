from typing import Awaitable, Callable

from .base import AsyncEmbedder, Embedder


class LiteLLMEmbedder(Embedder):
    def __init__(self, partial_embedding_fn: Callable[[str], list[float]]):
        self.embed = partial_embedding_fn

    def __call__(self, inputs: list[str]) -> list[list[float]]:
        response = self.embed(inputs)
        return [r["embedding"] for r in response["data"]]


class AsyncLiteLLMEmbedder(AsyncEmbedder):
    def __init__(self, partial_embedding_fn: Callable[[str], Awaitable[list[float]]]):
        self.embed = partial_embedding_fn

    async def __call__(self, inputs: list[str]) -> list[list[float]]:
        response = await self.embed(inputs)
        return [r["embedding"] for r in response["data"]]
