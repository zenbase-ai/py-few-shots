from typing import Mapping
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections import Collection, CollectionAsync
from weaviate.collections.classes.internal import QueryReturnType

from few_shots.types import (
    dump_io_value,
    Vector,
    parse_io_value,
    Shot,
    ShotWithSimilarity,
)

from .base import Store


class WeaviateBase(Store):
    @staticmethod
    def _shots_to_data_objects(
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ) -> list[DataObject]:
        return [
            DataObject(
                uuid=shot.id,
                vector=vector,
                properties={
                    "namespace": namespace,
                    "inputs": dump_io_value(shot.inputs),
                    "outputs": dump_io_value(shot.outputs),
                },
            )
            for shot, vector in zip(shots, vectors)
        ]

    @staticmethod
    def _response_to_shots_list(response: QueryReturnType) -> list[ShotWithSimilarity]:
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
    def _filter(namespace: str, ids: list[str] | None = None) -> Filter:
        filters = Filter.by_property("namespace").equal(namespace)
        if ids:
            filters &= Filter.by_id().contains_any(ids)
        return filters


class WeaviateStore(WeaviateBase):
    collection: Collection

    def __init__(self, collection: Collection):
        self.collection = collection

    def add(
        self,
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ):
        self.collection.data.insert_many(
            self._shots_to_data_objects(shots, vectors, namespace)
        )

    def remove(self, ids: list[str], namespace: str):
        self.collection.data.delete_many(self._filter(namespace, ids))

    def clear(self, namespace: str):
        self.collection.data.delete_many(self._filter(namespace))

    def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        response = self.collection.query.near_vector(
            vector,
            filters=self._filter(namespace),
            limit=limit,
        )

        return self._response_to_shots_list(response)


class AsyncWeaviateStore(WeaviateBase):
    collection: CollectionAsync

    def __init__(self, collection: CollectionAsync):
        self.collection = collection

    async def add(
        self,
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ):
        await self.collection.data.insert_many(
            self._shots_to_data_objects(shots, vectors, namespace)
        )

    async def remove(self, ids: list[str], namespace: str):
        await self.collection.data.delete_many(self._filter(namespace, ids))

    async def clear(self, namespace: str):
        await self.collection.data.delete_many(self._filter(namespace))

    async def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        response = await self.collection.query.near_vector(
            vector,
            filters=self._filter(namespace),
            limit=limit,
        )

        return self._response_to_shots_list(response)
