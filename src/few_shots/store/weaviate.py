from sorcery import dict_of
from weaviate import WeaviateAsyncClient, WeaviateClient
from weaviate.classes.config import Property, DataType, VectorDistances, Configure
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections.classes.internal import QueryReturnType
from weaviate.exceptions import WeaviateBaseError

from few_shots.types import (
    dump_io_value,
    parse_io_value,
    ScoredShot,
    Shot,
    Vector,
)
from few_shots.utils.datetime import utcnow

from .base import AsyncStore, Store

__all__ = ["WeaviateStore", "AsyncWeaviateStore", "VectorDistances"]


class WeaviateStore(Store):
    client: WeaviateClient
    collection_name: str
    distance_metric: VectorDistances

    def __init__(
        self,
        client: WeaviateClient,
        collection_name: str,
        distance_metric: VectorDistances = VectorDistances.COSINE,
    ):
        self.client = client
        self.collection_name = collection_name
        self.distance_metric = distance_metric

    def setup(self):
        try:
            self.collection = self.client.collections.create(
                **WeaviateHelper.collection_config(self.collection_name, self.distance_metric)
            )
        except WeaviateBaseError:
            self.collection = self.client.collections.get(self.collection_name)

    def teardown(self):
        self.client.collections.delete(self.collection_name)

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        self.collection.data.insert_many(WeaviateHelper.upsert_shots(shots, vectors, namespace))

    def get(self, ids: list[str], _namespace: str):
        return WeaviateHelper.fetch_shots(ids, self.collection.query.fetch_objects_by_ids(ids))

    def remove(self, ids: list[str], _namespace: str):
        self.collection.data.delete_many(Filter.by_id().contains_any(ids))

    def clear(self, namespace: str):
        self.collection.data.delete_many(Filter.by_property("namespace").equal(namespace))

    def list(self, vector: Vector, namespace: str, limit: int) -> list[ScoredShot]:
        return WeaviateHelper.query_scored_shots(
            self.collection.query.near_vector(
                vector,
                filters=Filter.by_property("namespace").equal(namespace),
                limit=limit,
            )
        )


class AsyncWeaviateStore(AsyncStore):
    client: WeaviateAsyncClient
    collection_name: str
    distance_metric: VectorDistances

    def __init__(
        self,
        client: WeaviateAsyncClient,
        collection_name: str,
        distance_metric: VectorDistances = VectorDistances.COSINE,
    ):
        self.client = client
        self.collection_name = collection_name
        self.distance_metric = distance_metric

    async def setup(self):
        try:
            self.collection = await self.client.collections.create(
                **WeaviateHelper.collection_config(self.collection_name, self.distance_metric)
            )
        except WeaviateBaseError:
            self.collection = await self.client.collections.get(self.collection_name)

    async def teardown(self):
        await self.client.collections.delete(self.collection_name)

    async def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        await self.collection.data.insert_many(
            WeaviateHelper.upsert_shots(shots, vectors, namespace)
        )

    async def get(self, ids: list[str], _namespace: str):
        return WeaviateHelper.fetch_shots(
            ids, await self.collection.query.fetch_objects_by_ids(ids)
        )

    async def remove(self, ids: list[str], _namespace: str):
        await self.collection.data.delete_many(Filter.by_id().contains_any(ids))

    async def clear(self, namespace: str):
        await self.collection.data.delete_many(Filter.by_property("namespace").equal(namespace))

    async def list(self, vector: Vector, namespace: str, limit: int) -> list[ScoredShot]:
        return WeaviateHelper.query_scored_shots(
            await self.collection.query.near_vector(
                vector,
                filters=Filter.by_property("namespace").equal(namespace),
                limit=limit,
            )
        )


class WeaviateHelper:
    @staticmethod
    def collection_config(collection_name: str, distance_metric: VectorDistances) -> dict:
        return dict(
            name=collection_name,
            properties=[
                Property(
                    name="namespace",
                    data_type=DataType.TEXT,
                    index_filterable=True,
                ),
                Property(name="inputs", data_type=DataType.TEXT),
                Property(name="outputs", data_type=DataType.TEXT),
                Property(name="updated_at", data_type=DataType.NUMBER),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=distance_metric),
        )

    @staticmethod
    def upsert_shots(
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ) -> list[DataObject]:
        updated_at = utcnow()
        return [
            DataObject(
                uuid=shot.id,
                vector=vector,
                properties=dict_of(
                    namespace,
                    updated_at,
                    inputs=dump_io_value(shot.inputs),
                    outputs=dump_io_value(shot.outputs),
                ),
            )
            for shot, vector in zip(shots, vectors)
        ]

    @staticmethod
    def fetch_shots(ids: list[str], response: QueryReturnType) -> list[Shot]:
        """
        Weaviate returns objects in a random order, so we need to order them by the ids we requested.
        """
        shots = {
            str(o.uuid): Shot(
                parse_io_value(o.properties["inputs"]),
                parse_io_value(o.properties["outputs"]),
                str(o.uuid),
            )
            for o in response.objects
        }
        return [shots[id] for id in ids if id in shots]

    @staticmethod
    def query_scored_shots(response: QueryReturnType) -> list[ScoredShot]:
        return [
            ScoredShot(
                o.metadata.distance,
                Shot(
                    parse_io_value(o.properties["inputs"]),
                    parse_io_value(o.properties["outputs"]),
                    str(o.uuid),
                ),
            )
            for o in response.objects
        ]
