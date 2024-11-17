import pytest
import weaviate

from few_shots.store.weaviate import AsyncWeaviateStore, WeaviateStore
from few_shots.types import Shot, Vector


@pytest.fixture
def client():
    with weaviate.connect_to_local() as c:
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


def test_crud(
    store: WeaviateStore,
    str_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.add(str_shots, mock_vectors, namespace)
    store.add(str_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    assert results[0][0].id == str_shots[0].id
    assert results[1][0].id == str_shots[1].id

    store.remove([str_shots[0].id], namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 1

    store.clear(namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 0


def test_structured_io(
    store: WeaviateStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (shot_0, _d0), (shot_1, _d1) = store.list(
        query_vector,
        namespace,
        limit=2,
    )

    assert shot_0 == struct_shots[0]
    assert shot_1 == struct_shots[1]


@pytest.fixture
async def async_client():
    async with weaviate.use_async_with_local() as c:
        yield c


@pytest.fixture
async def async_collection(async_client: weaviate.WeaviateAsyncClient):
    try:
        return async_client.collections.get("test")
    except weaviate.exceptions.WeaviateBaseError:
        return await async_client.collections.create("test")


@pytest.fixture
async def async_store(async_collection: weaviate.collections.CollectionAsync):
    return AsyncWeaviateStore(async_collection)


@pytest.mark.anyio
async def test_async_crud(
    async_store: AsyncWeaviateStore,
    str_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    await async_store.add(str_shots, mock_vectors, namespace)
    await async_store.add(str_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    results = await async_store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    assert results[0][0].id == str_shots[0].id
    assert results[1][0].id == str_shots[1].id

    await async_store.remove([str_shots[0].id], namespace)
    results = await async_store.list(query_vector, namespace, limit=2)
    assert len(results) == 1

    await async_store.clear(namespace)
    results = await async_store.list(query_vector, namespace, limit=2)
    assert len(results) == 0


@pytest.mark.anyio
async def test_async_structured_io(
    async_store: AsyncWeaviateStore,
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
