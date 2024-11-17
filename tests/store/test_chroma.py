import pytest

from chromadb import Client
from chromadb.api.client import ClientAPI

from few_shots.store.chroma import ChromaStore, Collection
from few_shots.types import Vector, Shot


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def collection(client: ClientAPI):
    return client.create_collection("test", get_or_create=True)


@pytest.fixture
def store(collection: Collection):
    return ChromaStore(collection)


def test_crud(
    store: ChromaStore,
    collection: Collection,
    str_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.add(str_shots, mock_vectors, namespace)
    store.add(str_shots, mock_vectors, namespace)
    assert collection.count() == 2

    store.add(str_shots, mock_vectors, namespace)
    assert collection.count() == 2

    query_vector = mock_vectors[0]
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    assert results[0][0].id == str_shots[0].id
    assert results[1][0].id == str_shots[1].id

    store.remove([str_shots[0].id], namespace)
    assert collection.count() == 1

    store.clear(namespace)
    assert collection.count() == 0


def test_structured_io(
    store: ChromaStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (shot_0, d0), (shot_1, d1) = store.list(
        query_vector,
        namespace,
        limit=2,
    )

    assert shot_0 == struct_shots[0]
    assert shot_1 == struct_shots[1]
    assert d0 <= d1
