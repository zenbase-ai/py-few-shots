from abc import abstractmethod

from few_shots.types import Embedding


class Embedder:
    @abstractmethod
    def __call__(self, inputs: list[str]) -> list[Embedding]: ...


class AsyncEmbedder(Embedder):
    @abstractmethod
    async def __call__(self, inputs: list[str]) -> list[Embedding]: ...
