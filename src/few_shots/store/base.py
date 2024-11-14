from abc import abstractmethod

from few_shots.types import Vector, Shot, ScoredShot


class Store:
    @abstractmethod
    def add(
        self,
        shots: list[Shot],
        vectors: list[Vector],
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
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]: ...


class AsyncStore(Store):
    @abstractmethod
    async def add(
        self,
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ): ...

    @abstractmethod
    async def remove(self, ids: list[str], namespace: str): ...

    @abstractmethod
    async def clear(self, namespace: str): ...

    @abstractmethod
    async def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]: ...
