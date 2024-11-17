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

from .base import Store


class WeaviateBase(Store):
    collection_name: str
    distance_metric: VectorDistances

    def __init__(
        self,
        collection_name: str,
        distance_metric: VectorDistances = VectorDistances.COSINE,
    ):
        self.collection_name = collection_name
        self.distance_metric = distance_metric

    def _create_collection_kwargs(self) -> dict:
        return dict(
            name=self.collection_name,
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
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=self.distance_metric
            ),
        )

    @staticmethod
    def _shots_to_data_objects(
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
    def _response_to_shots_list(response: QueryReturnType) -> list[ScoredShot]:
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


class WeaviateStore(WeaviateBase):
    client: WeaviateClient

    def __init__(
        self,
        client: WeaviateClient,
        collection_name: str,
        distance_metric: VectorDistances = VectorDistances.COSINE,
    ):
        super().__init__(collection_name, distance_metric)
        self.client = client

    def setup(self):
        try:
            self.collection = self.client.collections.create(
                **self._create_collection_kwargs()
            )
        except WeaviateBaseError:
            self.collection = self.client.collections.get(self.collection_name)

    def teardown(self):
        self.client.collections.delete(self.collection_name)

    def add(
        self,
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ):
        self.collection.data.insert_many(
            self._shots_to_data_objects(shots, vectors, namespace)
        )

    def remove(self, ids: list[str], _namespace: str):
        self.collection.data.delete_many(Filter.by_id().contains_any(ids))

    def clear(self, namespace: str):
        self.collection.data.delete_many(
            Filter.by_property("namespace").equal(namespace)
        )

    def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]:
        response = self.collection.query.near_vector(
            vector,
            filters=Filter.by_property("namespace").equal(namespace),
            limit=limit,
        )

        return self._response_to_shots_list(response)


class AsyncWeaviateStore(WeaviateBase):
    client: WeaviateAsyncClient

    def __init__(
        self,
        client: WeaviateAsyncClient,
        collection_name: str,
        distance_metric: VectorDistances = VectorDistances.COSINE,
    ):
        super().__init__(collection_name, distance_metric)
        self.client = client

    async def setup(self):
        try:
            self.collection = await self.client.collections.create(
                **self._create_collection_kwargs()
            )
        except WeaviateBaseError:
            self.collection = await self.client.collections.get(self.collection_name)

    async def teardown(self):
        await self.client.collections.delete(self.collection_name)

    async def add(
        self,
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ):
        await self.collection.data.insert_many(
            self._shots_to_data_objects(shots, vectors, namespace)
        )

    async def remove(self, ids: list[str], _namespace: str):
        await self.collection.data.delete_many(Filter.by_id().contains_any(ids))

    async def clear(self, namespace: str):
        await self.collection.data.delete_many(
            Filter.by_property("namespace").equal(namespace)
        )

    async def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]:
        response = await self.collection.query.near_vector(
            vector,
            filters=Filter.by_property("namespace").equal(namespace),
            limit=limit,
        )

        return self._response_to_shots_list(response)
