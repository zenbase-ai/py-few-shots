from typing import List

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HnswConfigDiff,
    KeywordIndexParams,
    MatchValue,
    PointStruct,
    Record,
    ScoredPoint,
    VectorParams,
)
from sorcery import dict_of

from few_shots.types import (
    dump_io_value,
    parse_io_value,
    ScoredShot,
    Shot,
    Vector,
)
from few_shots.utils.datetime import utcnow

from .base import AsyncStore, Store

__all__ = ["QdrantStore", "AsyncQdrantStore", "Distance"]


class QdrantStore(Store):
    client: QdrantClient
    collection_name: str

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def setup(self, size: int, distance: Distance, payload_m: int = 16):
        self.client.create_collection(
            **QdrantHelper.create_collection(self.collection_name, size, distance, payload_m)
        )
        self.client.create_payload_index(
            **QdrantHelper.create_payload_index(self.collection_name),
        )

    def teardown(self):
        self.client.delete_collection(self.collection_name)

    def add(self, shots: List[Shot], vectors: List[Vector], namespace: str):
        self.client.upsert(
            collection_name=self.collection_name,
            points=QdrantHelper.upsert_points(shots, vectors, namespace),
        )

    def get(self, ids: List[str], _namespace: str) -> List[Shot]:
        return QdrantHelper.retrieve_shots(
            self.client.retrieve(collection_name=self.collection_name, ids=ids)
        )

    def remove(self, ids: List[str], namespace: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=QdrantHelper.selector(namespace, ids),
        )

    def clear(self, namespace: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=QdrantHelper.selector(namespace),
        )

    def list(self, vector: Vector, namespace: str, limit: int) -> List[ScoredShot]:
        results = self.client.search(
            collection_name=namespace,
            query_vector=vector,
            query_filter=QdrantHelper.selector(namespace),
            limit=limit,
        )
        return QdrantHelper.search_scored_shots(results)


class AsyncQdrantStore(AsyncStore):
    client: AsyncQdrantClient
    collection_name: str

    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    async def setup(self, size: int, distance: Distance, payload_m: int = 16):
        await self.client.create_collection(
            **QdrantHelper.create_collection(self.collection_name, size, distance, payload_m)
        )
        await self.client.create_payload_index(
            **QdrantHelper.create_payload_index(self.collection_name),
        )

    async def teardown(self):
        await self.client.delete_collection(self.collection_name)

    async def add(self, shots: List[Shot], vectors: List[Vector], namespace: str):
        await self.client.upsert(
            collection_name=namespace,
            points=QdrantHelper.upsert_points(shots, vectors, namespace),
        )

    async def get(self, ids: List[str], _namespace: str) -> List[Shot]:
        return QdrantHelper.retrieve_shots(
            await self.client.retrieve(collection_name=self.collection_name, ids=ids)
        )

    async def remove(self, ids: List[str], namespace: str):
        await self.client.delete(
            collection_name=namespace,
            points_selector=QdrantHelper.selector(namespace, ids),
        )

    async def clear(self, namespace: str):
        await self.client.delete(
            collection_name=namespace,
            points_selector=QdrantHelper.selector(namespace),
        )

    async def list(self, vector: Vector, namespace: str, limit: int) -> List[ScoredShot]:
        results = await self.client.search(
            collection_name=namespace,
            query_vector=vector,
            query_filter=QdrantHelper.selector(namespace),
            limit=limit,
        )
        return QdrantHelper.search_scored_shots(results)


class QdrantHelper:
    @staticmethod
    def create_collection(collection_name: str, size: int, distance: Distance, payload_m: int = 16):
        """
        Use these as kwargs for `.create_collection` for optimal performance.
        """
        return dict(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
            hnsw_config=HnswConfigDiff(payload_m=payload_m, m=0),
        )

    @staticmethod
    def create_payload_index(collection_name: str):
        """
        Use these as kwargs for `.create_payload_index` for optimal performance.
        """
        return dict(
            collection_name=collection_name,
            field_name="namespace",
            field_schema=KeywordIndexParams(type="keyword", is_tenant=True),
        )

    @staticmethod
    def selector(namespace: str, ids: list[str] | None = None) -> Filter:
        cond = (
            FieldCondition(key="namespace", match=MatchValue(value=namespace))
            if not ids
            else HasIdCondition(has_id=ids)
        )
        return Filter(must=[cond])

    @staticmethod
    def upsert_points(
        shots: List[Shot],
        vectors: List[Vector],
        namespace: str,
    ) -> List[PointStruct]:
        updated_at = utcnow()
        return [
            PointStruct(
                id=shot.id,
                vector=vector,
                payload=dict_of(
                    namespace,
                    updated_at,
                    inputs=dump_io_value(shot.inputs),
                    outputs=dump_io_value(shot.outputs),
                ),
            )
            for shot, vector in zip(shots, vectors)
        ]

    @staticmethod
    def retrieve_shots(results: List[Record]) -> list[Shot]:
        return [
            Shot(
                inputs=parse_io_value(result.payload["inputs"]),
                outputs=parse_io_value(result.payload["outputs"]),
                id=str(result.id),
            )
            for result in results
        ]

    @staticmethod
    def search_scored_shots(results: List[ScoredPoint]) -> List[ScoredShot]:
        return [
            ScoredShot(
                result.score,
                Shot(
                    id=result.id,
                    inputs=parse_io_value(result.payload["inputs"]),
                    outputs=parse_io_value(result.payload["outputs"]),
                ),
            )
            for result in results
        ]
