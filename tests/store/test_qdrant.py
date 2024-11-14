from qdrant_client.local.qdrant_local import QdrantLocal
import pytest

from few_shots.store.qdrant import QdrantStore, Shot, Distance
from few_shots.types import id_io_value


@pytest.fixture
def client():
    return QdrantLocal(":memory:")


@pytest.fixture
def store(client: QdrantLocal):
    s = QdrantStore(client=client, collection_name="test")
    client.create_collection(**s.collection_config(2, Distance.COSINE))
    client.create_payload_index(**s.payload_config())
    yield s
    client.delete_payload_index("test", "namespace")
    client.delete_collection("test")


def test_crud(store: QdrantStore):
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


def test_structured_io(store: QdrantStore):
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
