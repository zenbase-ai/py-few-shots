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

from .base import AsyncStore, Store

__all__ = ["ChromaStore", "AsyncChromaStore"]


class ChromaStore(Store):
    collection: Collection

    def __init__(self, collection: Collection):
        self.collection = collection

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        self.collection.upsert(**ChromaHelper.upsert_shots(shots, vectors, namespace))

    def get(self, ids: list[str], _namespace: str) -> list[Shot]:
        return ChromaHelper.get_shots(self.collection.get(ids))

    def remove(self, ids: list[str], _namespace: str):
        self.collection.delete(ids=ids)

    def clear(self, namespace: str):
        self.collection.delete(where=dict_of(namespace))

    def list(self, vector: Vector, namespace: str, limit: int) -> list[ScoredShot]:
        return ChromaHelper.query_scored_shots(
            self.collection.query(
                query_embeddings=[vector],
                where=dict_of(namespace),
                n_results=limit,
            )
        )


class AsyncChromaStore(AsyncStore):
    collection: AsyncCollection

    def __init__(self, collection: AsyncCollection):
        self.collection = collection

    async def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        await self.collection.upsert(**ChromaHelper.upsert_shots(shots, vectors, namespace))

    async def get(self, ids: list[str], _namespace: str) -> list[Shot]:
        return ChromaHelper.get_shots(await self.collection.get(ids))

    async def remove(self, ids: list[str], _namespace: str):
        await self.collection.delete(ids=ids)

    async def clear(self, namespace: str):
        await self.collection.delete(where=dict_of(namespace))

    async def list(self, vector: Vector, namespace: str, limit: int) -> list[ScoredShot]:
        return ChromaHelper.query_scored_shots(
            await self.collection.query(
                query_embeddings=[vector],
                where=dict_of(namespace),
                n_results=limit,
            )
        )


class ChromaHelper:
    @staticmethod
    def upsert_shots(shots: list[Shot], vectors: list[Vector], namespace: str) -> dict:
        updated_at = utcnow()
        return dict(
            ids=[s.id for s in shots],
            embeddings=vectors,
            documents=[s.key for s in shots],
            metadatas=[
                dict_of(namespace, updated_at, outputs=dump_io_value(s.outputs)) for s in shots
            ],
        )

    @staticmethod
    def get_shots(results: dict) -> list[Shot]:
        return [
            Shot(parse_io_value(inputs), parse_io_value(metadata["outputs"]), id)
            for (inputs, metadata, id) in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
            )
        ]

    @staticmethod
    def query_scored_shots(results: dict) -> list[ScoredShot]:
        return [
            ScoredShot(
                distance,
                Shot(parse_io_value(inputs), parse_io_value(metadata["outputs"]), id),
            )
            for (distance, id, inputs, metadata) in zip(
                results["distances"][0],
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
        ]
