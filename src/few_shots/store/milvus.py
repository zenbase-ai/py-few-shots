from typing import Literal, TypeVar
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from few_shots.types import (
    dump_io_value,
    parse_io_value,
    ScoredShot,
    Shot,
    Vector,
)
from few_shots.utils.asyncio import asyncify_class

from .base import Store


MetricType = TypeVar("MetricType", bound=Literal["L2", "IP", "COSINE", "HAMMING", "JACCARD"])


class MilvusStore(Store):
    client: MilvusClient
    collection_name: str

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def setup(self, size: int, metric_type: MetricType = "COSINE"):
        if self.client.has_collection(self.collection_name):
            return

        fields = [
            FieldSchema("id", DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema("namespace", DataType.VARCHAR, max_length=512),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=size),
            FieldSchema("payload", DataType.JSON),
            FieldSchema("updated_at", DataType.FLOAT),
        ]

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=CollectionSchema(fields),
            index=self.client.prepare_index_params(
                field_name="vector",
                metric_type=metric_type,
                index_type="AUTOINDEX",
                index_name="vectors_index",
            ),
        )

    def teardown(self):
        self.client.drop_collection(self.collection_name)

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "id": shot.id,
                    "namespace": namespace,
                    "vectors": vector,
                    "payload": {
                        "inputs": dump_io_value(shot.inputs),
                        "outputs": dump_io_value(shot.outputs),
                    },
                }
                for shot, vector in zip(shots, vectors)
            ],
        )

    def get(self, ids: list[str], namespace: str) -> list[Shot]:
        response = self.client.query(
            collection_name=self.collection_name,
            filter=f"namespace == '{namespace}'",
            output_fields=["id", "payload"],
            ids=ids,
        )[0]

        results: list[Shot] = []
        for datum in response:
            metadata = datum.get("entity", {}).get("payload")
            inputs = parse_io_value(metadata["inputs"])
            outputs = parse_io_value(metadata["outputs"])
            results.append(Shot(inputs, outputs, datum["id"]))

        return results

    def remove(self, ids: list[str], namespace: str):
        self.client.delete(
            collection_name=self.collection_name,
            ids=ids,
            filter=f"namespace == '{namespace}'",
        )

    def clear(self, namespace: str):
        self.client.delete(
            collection_name=self.collection_name,
            filter=f"namespace == '{namespace}'",
        )

    def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]:
        response = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            filter=f"namespace == '{namespace}'",
            limit=limit,
        )[0]

        results: list[ScoredShot] = []
        for datum in response:
            metadata = datum.get("entity", {}).get("payload")
            inputs = parse_io_value(metadata["inputs"])
            outputs = parse_io_value(metadata["outputs"])
            results.append(ScoredShot(datum["distance"], Shot(inputs, outputs, datum["id"])))

        return results


@asyncify_class
class AsyncMilvusStore(MilvusStore): ...
