from abc import abstractmethod


class Embedder:
    @abstractmethod
    def __call__(self, inputs: list[str]) -> list[list[float]]: ...


class AsyncEmbedder(Embedder):
    @abstractmethod
    async def __call__(self, inputs: list[str]) -> list[list[float]]: ...
