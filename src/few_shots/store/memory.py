from collections import defaultdict
from operator import itemgetter
from typing import Callable

import numpy as np

from .base import ScoredShot, Shot, Store, Vector


__all__ = ["MemoryStore"]


def cosine_distance(a: Vector, b: Vector) -> float:
    if not (norm_a := np.linalg.norm(a)) or not (norm_b := np.linalg.norm(b)):
        return 1.0
    return 1 - np.dot(a, b) / (norm_a * norm_b)


class MemoryStore(Store):
    # namespace => id => (shot, vector)
    _storage: dict[str, dict[str, tuple[Shot, Vector]]]
    distance: Callable[[Vector, Vector], float]

    def __init__(self, distance: Callable[[Vector, Vector], float] = cosine_distance):
        self._storage = defaultdict(dict)
        self.distance = distance

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        for shot, vector in zip(shots, vectors):
            self._storage[namespace][shot.id] = (shot, vector)

    def get(self, ids: list[str], namespace: str) -> list[Shot]:
        results: list[Shot] = []
        for id in ids:
            result = self._storage[namespace].get(id)
            if result:
                results.append(result[0])
        return results

    def remove(self, ids: list[str], namespace: str):
        for id in ids:
            del self._storage[namespace][id]

    def clear(self, namespace: str):
        self._storage[namespace].clear()

    def list(self, vector: Vector, namespace: str, limit: int) -> list[ScoredShot]:
        scored_shots = [
            ScoredShot(self.distance(vector, emb), shot)
            for (shot, emb) in self._storage[namespace].values()
        ]
        return sorted(scored_shots, key=itemgetter(1), reverse=True)[:limit]
