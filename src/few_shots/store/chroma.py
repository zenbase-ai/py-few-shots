from chromadb import Collection
from chromadb.api.async_client import AsyncCollection
from sorcery import dict_of

from few_shots.types import (
    dump_io_value,
    parse_io_value,
    ScoredShot,
    Shot,
    Vector,
)
from few_shots.utils.datetime import utcnow

from .base import Store


class ChromaBase(Store):
    @staticmethod
    def _upsert_kwargs(
        shots: list[Shot], vectors: list[Vector], namespace: str
    ) -> dict:
        updated_at = utcnow()
        return dict(
            ids=[s.id for s in shots],
            embeddings=vectors,
            documents=[s.key for s in shots],
            metadatas=[
                dict_of(namespace, updated_at, outputs=dump_io_value(s.outputs))
                for s in shots
            ],
        )

    @staticmethod
    def _query_to_shots_list(results: dict) -> list[ScoredShot]:
        return [
            (
                Shot(parse_io_value(inputs), parse_io_value(metadata["outputs"]), id),
                distance,
            )
            for (id, inputs, distance, metadata) in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]


class ChromaStore(ChromaBase):
    collection: Collection

    def __init__(self, collection: Collection):
        self.collection = collection

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        self.collection.upsert(**self._upsert_kwargs(shots, vectors, namespace))

    def remove(self, ids: list[str], _namespace: str):
        self.collection.delete(ids=ids)

    def clear(self, namespace: str):
        self.collection.delete(where=dict_of(namespace))

    def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]:
        return self._query_to_shots_list(
            self.collection.query(
                query_embeddings=[vector],
                where=dict_of(namespace),
                n_results=limit,
            )
        )


class AsyncChromaStore(ChromaBase):
    collection: AsyncCollection

    def __init__(self, collection: AsyncCollection):
        self.collection = collection

    async def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        await self.collection.upsert(**self._upsert_kwargs(shots, vectors, namespace))

    async def remove(self, ids: list[str], _namespace: str):
        await self.collection.delete(ids=ids)

    async def clear(self, namespace: str):
        await self.collection.delete(where=dict_of(namespace))

    async def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]:
        return self._query_to_shots_list(
            await self.collection.query(
                query_embeddings=[vector],
                where=dict_of(namespace),
                n_results=limit,
            )
        )
