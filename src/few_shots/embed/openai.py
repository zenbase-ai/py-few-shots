from functools import partial
from typing import Any, Awaitable, Callable

from few_shots.types import Vector
from .base import AsyncEmbed, Embed


class OpenAIEmbed(Embed):
    def __init__(self, embedder: Callable[[str], Any], model: str, **kwargs):
        self.embedder = partial(embedder, model=model, **kwargs)

    def __call__(self, inputs: list[str]) -> list[Vector]:
        response = self.embedder(inputs)
        return [r["embedding"] for r in response["data"]]


class AsyncOpenAIEmbed(AsyncEmbed):
    def __init__(self, embedder: Callable[[str], Awaitable[Any]], model: str, **kwargs):
        self.embedder = partial(embedder, model=model, **kwargs)

    async def __call__(self, inputs: list[str]) -> list[Vector]:
        response = await self.embedder(inputs)
        return [r["embedding"] for r in response["data"]]
