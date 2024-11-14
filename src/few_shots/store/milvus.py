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

from .base import Store


MetricType = TypeVar(
    "MetricType", bound=Literal["L2", "IP", "COSINE", "HAMMING", "JACCARD"]
)


class MilvusStore(Store):
    client: MilvusClient
    collection_name: str

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def collection_config(self, size: int, metric_type: MetricType = "COSINE"):
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
            ),
            FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=size),
            FieldSchema(name="payload", dtype=DataType.JSON),
        ]

        return dict(
            collection_name=self.collection_name,
            schema=CollectionSchema(fields),
            index=self.client.prepare_index_params(
                field_name="vectors",
                metric_type=metric_type,
                index_type="AUTOINDEX",
                index_name="vectors_index",
            ),
        )

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
            results.append((Shot(inputs, outputs, datum["id"]), datum["distance"]))

        return results
