from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections import Collection, CollectionAsync
from weaviate.collections.classes.internal import QuerySearchReturnType

from best_shot.types import (
    dump_io_value,
    Embedding,
    parse_io_value,
    Shot,
    ShotWithSimilarity,
)

from .base import Store


class WeaviateBase(Store):
    @staticmethod
    def _shots_to_data_objects(
        shots: list[Shot],
        embeddings: list[Embedding],
        namespace: str,
    ) -> list[DataObject]:
        return [
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

    @staticmethod
    def _response_to_shots(response: QuerySearchReturnType) -> list[ShotWithSimilarity]:
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

    @staticmethod
    def _namespace_filter(namespace: str) -> Filter:
        return Filter.by_property("namespace").equal(namespace)


class WeaviateStore(WeaviateBase):
    collection: Collection

    def __init__(self, collection: Collection):
        self.collection = collection

    def add(
        self,
        shots: list[Shot],
        embeddings: list[Embedding],
        namespace: str,
    ):
        self.collection.data.insert_many(
            self._shots_to_data_objects(shots, embeddings, namespace)
        )

    def remove(self, ids: list[str], namespace: str):
        self.collection.data.delete_many(
            self._namespace_filter(namespace) & Filter.by_id().contains_any(ids)
        )

    def clear(self, namespace: str):
        self.collection.data.delete_many(self._namespace_filter(namespace))

    def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        response = self.collection.query.near_vector(
            embedding,
            filters=self._namespace_filter(namespace),
            limit=limit,
        )

        return self._response_to_shots(response)


class AsyncWeaviateStore(WeaviateBase):
    collection: CollectionAsync

    def __init__(self, collection: CollectionAsync):
        self.collection = collection

    async def add(
        self,
        shots: list[Shot],
        embeddings: list[Embedding],
        namespace: str,
    ):
        await self.collection.data.insert_many(
            self._shots_to_data_objects(shots, embeddings, namespace)
        )

    async def remove(self, ids: list[str], namespace: str):
        await self.collection.data.delete_many(
            self._namespace_filter(namespace) & Filter.by_id().contains_any(ids)
        )

    async def clear(self, namespace: str):
        await self.collection.data.delete_many(self._namespace_filter(namespace))

    async def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        response = await self.collection.query.near_vector(
            embedding,
            filters=self._namespace_filter(namespace),
            limit=limit,
        )

        return self._response_to_shots(response)
