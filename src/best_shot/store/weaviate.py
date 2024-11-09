from dataclasses import dataclass

import weaviate

from best_shot.types import Shot, ShotWithSimilarity

from .base import Store


@dataclass
class WeaviateStore(Store):
    client: weaviate.WeaviateClient

    def add(self, shots: list[Shot], _embeddings, namespace: str):
        """
        Since Weaviate handles embedding server-side, we can ignore the embeddings.
        """

    def remove(self, ids: list[str], namespace: str): ...

    def clear(self, namespace: str): ...

    def list(
        self,
        embedding: list[float],
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]: ...
