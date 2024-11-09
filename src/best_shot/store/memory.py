from collections import defaultdict
from operator import itemgetter

import numpy as np

from best_shot.types import Shot
from best_shot.utils.asyncify import asyncify_class

from .base import ShotWithSimilarity, Store


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class MemoryStore(Store):
    # namespace => key => (shot, embedding)
    _storage: dict[str, dict[str, tuple[Shot, list[float]]]]

    def __init__(self):
        self._storage = defaultdict(dict)

    def add(self, shots: list[Shot], embeddings: list[list[float]], namespace: str):
        for shot, embedding in zip(shots, embeddings):
            self._storage[namespace][shot.id] = (shot, embedding)

    def remove(self, ids: list[str], namespace: str):
        for id in ids:
            del self._storage[namespace][id]

    def clear(self, namespace: str):
        self._storage[namespace].clear()

    def list(
        self,
        embedding: list[float],
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        shots_with_similarities = [
            (shot, 1 - cosine_similarity(embedding, emb))
            for (shot, emb) in self._storage[namespace].values()
        ]
        return sorted(shots_with_similarities, key=itemgetter(1), reverse=True)[:limit]


@asyncify_class
class AsyncMemoryStore(MemoryStore): ...
