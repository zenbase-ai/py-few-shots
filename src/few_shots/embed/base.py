from abc import abstractmethod

from few_shots.types import Vector


class Embedder:
    @abstractmethod
    def __call__(self, inputs: list[str]) -> list[Vector]: ...


class AsyncEmbedder(Embedder):
    @abstractmethod
    async def __call__(self, inputs: list[str]) -> list[Vector]: ...
