from abc import abstractmethod

from best_shot.types import Embedding, Shot, ShotWithSimilarity


class Store:
    @abstractmethod
    def add(
        self,
        shots: list[Shot],
        embeddings: list[Embedding],
        namespace: str,
    ): ...

    @abstractmethod
    def remove(
        self,
        ids: list[str],
        namespace: str,
    ): ...

    @abstractmethod
    def clear(self, namespace: str): ...

    @abstractmethod
    def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]: ...


class AsyncStore(Store):
    @abstractmethod
    async def add(
        self,
        shots: list[Shot],
        embeddings: list[Embedding],
        namespace: str,
    ): ...

    @abstractmethod
    async def remove(self, ids: list[str], namespace: str): ...

    @abstractmethod
    async def clear(self, namespace: str): ...

    @abstractmethod
    async def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]: ...
