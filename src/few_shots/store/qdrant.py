from typing import List

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    KeywordIndexParams,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
    HasIdCondition,
)

from few_shots.types import (
    dump_io_value,
    parse_io_value,
    ScoredShot,
    Shot,
    Vector,
)

from .base import Store


class QdrantBase(Store):
    collection_name: str

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def collection_config(self, size: int, distance: Distance, payload_m: int = 16):
        """
        Use these as kwargs for `.create_collection` for optimal performance.
        """
        return dict(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
            hnsw_config=HnswConfigDiff(payload_m=payload_m, m=0),
        )

    def payload_config(self):
        """
        Use these as kwargs for `.create_payload_index` for optimal performance.
        """
        return dict(
            collection_name=self.collection_name,
            field_name="namespace",
            field_schema=KeywordIndexParams(type="keyword", is_tenant=True),
        )

    def _filter(self, namespace: str, ids: list[str] | None = None) -> Filter:
        must = [FieldCondition(key="namespace", match=MatchValue(value=namespace))]
        if ids:
            must.append(HasIdCondition(has_id=ids))
        return Filter(must=must)

    def _shots_to_point_structs(
        self,
        shots: List[Shot],
        vectors: List[Vector],
        namespace: str,
    ) -> List[PointStruct]:
        return [
            PointStruct(
                id=shot.id,
                vector=vector,
                payload={
                    "namespace": namespace,
                    "inputs": dump_io_value(shot.inputs),
                    "outputs": dump_io_value(shot.outputs),
                },
            )
            for shot, vector in zip(shots, vectors)
        ]

    def _search_to_shots_list(
        self,
        results: List[ScoredPoint],
    ) -> List[ScoredShot]:
        return [
            (
                Shot(
                    id=result.id,
                    inputs=parse_io_value(result.payload["inputs"]),
                    outputs=parse_io_value(result.payload["outputs"]),
                ),
                result.score,
            )
            for result in results
        ]


class QdrantStore(QdrantBase):
    client: QdrantClient

    def __init__(self, client: QdrantClient, collection_name: str):
        super().__init__(collection_name)
        self.client = client

    def add(self, shots: List[Shot], vectors: List[Vector], namespace: str):
        self.client.upsert(
            collection_name=self.collection_name,
            points=self._shots_to_point_structs(shots, vectors, namespace),
        )

    def remove(self, ids: List[str], namespace: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._filter(namespace, ids),
        )

    def clear(self, namespace: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._filter(namespace),
        )

    def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> List[ScoredShot]:
        results = self.client.search(
            collection_name=namespace,
            query_vector=vector,
            query_filter=self._filter(namespace),
            limit=limit,
        )
        return self._search_to_shots_list(results)


class AsyncQdrantStore(QdrantBase):
    client: AsyncQdrantClient

    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        super().__init__(collection_name)
        self.client = client

    async def add(self, shots: List[Shot], vectors: List[Vector], namespace: str):
        await self.client.upsert(
            collection_name=namespace,
            points=self._shots_to_point_structs(shots, vectors, namespace),
        )

    async def remove(self, ids: List[str], namespace: str):
        await self.client.delete(
            collection_name=namespace,
            points_selector=self._filter(namespace, ids),
        )

    async def clear(self, namespace: str):
        await self.client.delete(
            collection_name=namespace,
            points_selector=self._filter(namespace),
        )

    async def list(
        self, vector: Vector, namespace: str, limit: int
    ) -> List[ScoredShot]:
        results = await self.client.search(
            collection_name=namespace,
            query_vector=vector,
            query_filter=self._filter(namespace),
            limit=limit,
        )
        return self._search_to_shots_list(results)
