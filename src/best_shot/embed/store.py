"""
Use this embedder when the store handles embeddings.
"""

from .base import AsyncEmbedder, Embedder


class StoreEmbedder(Embedder):
    def __call__(self, inputs: list[str]) -> list[list[float]]:
        return [[]] * len(inputs)


class AsyncStoreEmbedder(AsyncEmbedder):
    async def __call__(self, inputs: list[str]) -> list[list[float]]:
        return [[]] * len(inputs)
