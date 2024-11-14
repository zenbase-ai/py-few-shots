import pytest
import weaviate

from few_shots.store.weaviate import WeaviateStore
from few_shots.types import Shot, id_io_value


@pytest.fixture(scope="module")
def client():
    with weaviate.connect_to_embedded() as c:
        yield c


@pytest.fixture
def collection(client: weaviate.WeaviateClient):
    try:
        return client.collections.get("test")
    except weaviate.exceptions.WeaviateBaseError:
        return client.collections.create("test")


@pytest.fixture
def store(collection: weaviate.collections.Collection):
    return WeaviateStore(collection)


def test_crud(store: WeaviateStore):
    shots = [
        Shot("input1", "output1", id_io_value("id1")),
        Shot("input2", "output2", id_io_value("id2")),
    ]
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    namespace = "test"

    store.add(shots, embeddings, namespace)
    store.add(shots, embeddings, namespace)

    results = store.list([1.0, 2.0], namespace, 2)
    assert len(results) == 2
    assert results[0][0].id == id_io_value("id1")
    assert results[1][0].id == id_io_value("id2")

    store.remove([id_io_value("id1")], namespace)
    results = store.list([1.0, 2.0], namespace, 2)
    assert len(results) == 1

    store.clear(namespace)
    results = store.list([1.0, 2.0], namespace, 2)
    assert len(results) == 0


def test_structured_io(store: WeaviateStore):
    shots = [
        Shot({"key": "input1"}, {"key": "output1"}, id_io_value("id1")),
        Shot({"key": "input2"}, {"key": "output2"}, id_io_value("id2")),
    ]
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    namespace = "test"

    store.add(shots, embeddings, namespace)

    (shot_0, _distance_0), (shot_1, _distance_1) = store.list([1.0, 2.0], namespace, 2)

    assert shot_0 == shots[0]
    assert shot_1 == shots[1]
