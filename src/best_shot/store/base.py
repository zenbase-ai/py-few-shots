from abc import abstractmethod

from best_shot.types import Shot, ShotWithSimilarity


class Store:
    @abstractmethod
    def add(
        self,
        shots: list[Shot],
        embeddings: list[list[float]],
        namespace: str,
    ): ...

    @abstractmethod
    def remove(
        self,
        ids: list[str],
        namespace: str,
    ): ...

    @abstractmethod
    def list(
        self,
        embedding: list[float],
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]: ...
