from typing import List

from chromadb import Collection
from chromadb.api.async_client import AsyncCollection

from few_shots.types import (
    dump_io_value,
    Embedding,
    parse_io_value,
    Shot,
    ShotWithSimilarity,
)

from .base import Store


class ChromaBase(Store):
    @staticmethod
    def _query_to_list(results: dict) -> list[ShotWithSimilarity]:
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

    def add(self, shots: list[Shot], embeddings: list[Embedding], namespace: str):
        self.collection.upsert(
            ids=[shot.id for shot in shots],
            documents=[shot.key for shot in shots],
            embeddings=embeddings,
            metadatas=[
                {"namespace": namespace, "outputs": dump_io_value(s.outputs)}
                for s in shots
            ],
        )

    def remove(self, ids: list[str], namespace: str):
        self.collection.delete(ids=ids, where={"namespace": namespace})

    def clear(self, namespace: str):
        self.collection.delete(where={"namespace": namespace})

    def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={"namespace": namespace},
        )
        return self._query_to_list(results)


class AsyncChromaStore(ChromaBase):
    collection: AsyncCollection

    async def add(self, shots: list[Shot], embeddings: list[Embedding], namespace: str):
        await self.collection.add(
            ids=[shot.id for shot in shots],
            embeddings=embeddings,
            documents=[shot.key for shot in shots],
            metadatas=[
                {"namespace": namespace, "outputs": dump_io_value(s.outputs)}
                for s in shots
            ],
        )

    async def remove(self, ids: list[str], namespace: str):
        await self.collection.delete(ids=ids, where={"namespace": namespace})

    async def clear(self, namespace: str):
        await self.collection.delete(where={"namespace": namespace})

    async def list(
        self,
        embedding: Embedding,
        namespace: str,
        limit: int,
    ) -> list[ShotWithSimilarity]:
        results = await self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={"namespace": namespace},
        )
        return self._query_to_list(results)
