from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
import pytest

from few_shots.store.qdrant import AsyncQdrantStore, QdrantStore, Shot, Distance
from few_shots.types import Vector, id_io_value


@pytest.fixture
def client():
    return QdrantLocal(":memory:")


@pytest.fixture
def store(client: QdrantLocal):
    s = QdrantStore(client=client, collection_name="test")
    s.setup(size=2, distance=Distance.COSINE)
    yield s
    s.teardown()


def test_crud(
    store: QdrantStore,
    str_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.add(str_shots, mock_vectors, namespace)
    store.add(str_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    assert results[0][0].id == id_io_value("id1")
    assert results[1][0].id == id_io_value("id2")

    store.remove([id_io_value("id1")], namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 1

    store.clear(namespace)
    results = store.list(query_vector, namespace, 2)
    assert len(results) == 0


def test_structured_io(
    store: QdrantStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.add(struct_shots, mock_vectors, namespace)
    query_vector = mock_vectors[0]
    (shot_0, _d0), (shot_1, _d1) = store.list(query_vector, namespace, limit=2)

    assert shot_0 == struct_shots[0]
    assert shot_1 == struct_shots[1]


@pytest.fixture
def async_client():
    return AsyncQdrantLocal(":memory:")


@pytest.fixture
async def async_store(async_client: AsyncQdrantLocal):
    s = AsyncQdrantStore(client=async_client, collection_name="test")
    await s.setup(size=2, distance=Distance.COSINE)
    yield s
    await s.teardown()


@pytest.mark.anyio
async def test_crud_async(
    async_store: AsyncQdrantStore,
    str_shots: list[Shot],
    mock_vectors: list[Shot],
    namespace: str,
):
    await async_store.add(str_shots, mock_vectors, namespace)
    await async_store.add(str_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    results = await async_store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    assert results[0][0].id == id_io_value("id1")
    assert results[1][0].id == id_io_value("id2")

    await async_store.remove([id_io_value("id1")], namespace)
    results = await async_store.list(query_vector, namespace, limit=2)
    assert len(results) == 1

    await async_store.clear(namespace)
    results = await async_store.list(query_vector, namespace, limit=2)
    assert len(results) == 0


@pytest.mark.anyio
async def test_structured_io_async(
    async_store: AsyncQdrantStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    await async_store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (shot_0, _d0), (shot_1, _d1) = await async_store.list(
        query_vector,
        namespace,
        limit=2,
    )

    assert shot_0 == struct_shots[0]
    assert shot_1 == struct_shots[1]
