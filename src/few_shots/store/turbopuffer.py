from typing import Literal, TypeVar

import turbopuffer as tpuf

from few_shots.types import Shot, Vector, ScoredShot
from few_shots.utils.datetime import utcnow

from .base import Store

DistanceMetric = TypeVar(
    "DistanceMetric", bound=Literal["cosine_distance", "euclidean_squared"]
)


class TurboPufferStore(Store):
    distance_metric: DistanceMetric

    def __init__(self, distance_metric: DistanceMetric = "cosine_distance"):
        self.distance_metric = distance_metric

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        updated_at = utcnow()
        tpuf.Namespace(namespace).upsert(
            ids=[shot.id for shot in shots],
            vectors=vectors,
            attributes={
                "inputs": [shot.input for shot in shots],
                "outputs": [shot.output for shot in shots],
                "updated_at": [updated_at for _ in shots],
            },
        )

    def remove(self, ids: list[str], namespace: str):
        tpuf.Namespace(namespace).delete(ids)

    def clear(self, namespace: str):
        tpuf.Namespace(namespace).delete_all()

    def list(self, query_vector: Vector, namespace: str, limit: int):
        vector_results = tpuf.Namespace(namespace).query(
            vector=query_vector,
            top_k=limit,
            distance_metric=self.distance_metric,
            include_attributes=["inputs", "outputs"],
        )
        return [
            ScoredShot(
                row.dist,
                Shot(
                    id=row.id,
                    input=row.attributes["inputs"],
                    output=row.attributes["outputs"],
                ),
            )
            for row in vector_results
        ]
