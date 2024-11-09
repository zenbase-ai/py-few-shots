from typing import Callable

from .base import Embedder


class LiteLLMEmbedder(Embedder):
    def __init__(self, partial_embedding_fn: Callable[[str], list[float]]):
        self.embed = partial_embedding_fn

    def embed(self, inputs: list[str]) -> list[list[float]]:
        response = self.embed(inputs)
        return [r["embedding"] for r in response["data"]]
