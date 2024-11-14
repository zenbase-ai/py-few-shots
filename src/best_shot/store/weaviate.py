from weaviate.classes.query import Filter
from weaviate.collections import Collection
from weaviate.classes.data import DataObject

from best_shot.types import (
    dump_io_value,
    Embedding,
    parse_io_value,
    Shot,
    ShotWithSimilarity,
)

from .base import Store


class WeaviateStore(Store):
    collection: Collection

    def __init__(self, collection: Collection):
        self.collection = collection

    def add(
        self,
        shots: list[Shot],
        embeddings: list[Embedding],
        namespace: str,
    ):
        objects = [
            DataObject(
                uuid=shot.id,
                vector=embedding,
                properties={
                    "namespace": namespace,
                    "inputs": dump_io_value(shot.inputs),
                    "outputs": dump_io_value(shot.outputs),
                },
            )
            for shot, embedding in zip(shots, embeddings)
        ]
        self.collection.data.insert_many(objects)

    def remove(self, ids: list[str], namespace: str):
        self.collection.data.delete_many(
            Filter.by_property("namespace").equal(namespace)
            & Filter.by_id().contains_any(ids)
        )

    def clear(self, namespace: str):
        self.collection.data.delete_many(
            Filter.by_property("namespace").equal(namespace)
        )

    def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        response = self.collection.query.near_vector(
            embedding,
            filters=Filter.by_property("namespace").equal(namespace),
            limit=limit,
        )

        return [
            (
                Shot(
                    parse_io_value(o.properties["inputs"]),
                    parse_io_value(o.properties["outputs"]),
                    str(o.uuid),
                ),
                o.metadata.distance,
            )
            for o in response.objects
        ]
