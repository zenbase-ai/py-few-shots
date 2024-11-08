import pytest
from sentence_transformers import SentenceTransformer

from best_shot.client import BestShots
from best_shot.embed.transformers import TransformersEmbedder
from best_shot.store.memory import MemoryStore
from best_shot.types import Shot


@pytest.fixture(scope="module")
def client():
    return BestShots(
        embedder=TransformersEmbedder(model=SentenceTransformer("all-MiniLM-L6-v2")),
        store=MemoryStore(),
    )


def test_flow(client: BestShots):
    inputs = {"a": 1}
    outputs = {"b": 2}

    id = client.add(inputs, outputs, namespace="test")
    assert id == Shot(inputs, outputs).id

    results = client.list(inputs, namespace="test", limit=1)
    results = [shot for shot, _ in results]
    assert results == [Shot(inputs, outputs)]

    results = client.list(inputs, namespace="test", limit=5)
    results = [shot for shot, _ in results]
    assert results == [Shot(inputs, outputs)]

    client.remove(inputs, outputs, namespace="test")
    assert [] == client.list(inputs, namespace="test", limit=1)

    client.add(inputs, outputs, namespace="test")
    client.clear(namespace="test")
    assert [] == client.list(inputs, namespace="test")
