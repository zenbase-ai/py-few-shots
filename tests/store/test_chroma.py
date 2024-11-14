import pytest

from chromadb import Client
from chromadb.api.client import ClientAPI

from few_shots.store.chroma import ChromaStore, Collection, Shot


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def collection(client: ClientAPI):
    return client.create_collection("test", get_or_create=True)


@pytest.fixture
def store(collection: Collection):
    return ChromaStore(collection)


def test_crud(store: ChromaStore, collection: Collection):
    shots = [Shot("input1", "output1", "id1"), Shot("input2", "output2", "id2")]
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    namespace = "test"

    store.add(shots, embeddings, namespace)
    assert collection.count() == 2

    store.add(shots, embeddings, namespace)
    assert collection.count() == 2

    results = store.list([1.0, 2.0], namespace, 2)
    assert len(results) == 2
    assert results[0][0].id == "id1"
    assert results[1][0].id == "id2"

    store.remove(["id1"], namespace)
    assert collection.count() == 1

    store.clear(namespace)
    assert collection.count() == 0


def test_structured_io(store: ChromaStore):
    shots = [
        Shot({"key": "input1"}, {"key": "output1"}, "id1"),
        Shot({"key": "input2"}, {"key": "output2"}, "id2"),
    ]
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    namespace = "test"

    store.add(shots, embeddings, namespace)

    (shot_0, distance_0), (shot_1, distance_1) = store.list([1.0, 2.0], namespace, 2)

    assert shot_0 == shots[0]
    assert shot_1 == shots[1]
    assert distance_0 <= distance_1
