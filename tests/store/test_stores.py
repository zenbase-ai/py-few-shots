"""
Since some stores are stateful, make sure to clear them at the beginning of each test.
"""

import pytest
from pytest_lazy_fixtures import lf

from few_shots.store.base import Store, AsyncStore
from few_shots.types import Shot, Vector


providers = ["chroma", "pg", "qdrant", "weaviate"]  # TODO: Add Milvus & TurboPuffer


lazy_sync_stores = [lf(f"{p}_store") for p in providers]
lazy_async_stores = [lf(f"async_{p}_store") for p in providers]


@pytest.mark.parametrize("store", lazy_sync_stores)
def test_crud(
    store: Store,
    str_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.clear(namespace)

    store.add(str_shots, mock_vectors, namespace)
    store.add(str_shots, mock_vectors, namespace)  # Test idempotency

    query_vector = mock_vectors[0]
    (_d0, s0), (_d1, s1) = store.list(query_vector, namespace, limit=2)
    assert [s0, s1] == str_shots

    assert [s0, s1] == store.get([s0.id, s1.id], namespace)

    store.remove([s0.id], namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 1
    assert results[0].shot == s1
    assert [] == store.get([s0.id], namespace)

    store.clear(namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 0
    assert [] == store.get([s0.id], namespace)


@pytest.mark.parametrize("store", lazy_sync_stores)
def test_structured_io(
    store: Store,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    store.clear(namespace)

    store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (_d0, s0), (_d1, s1) = store.list(query_vector, namespace, limit=2)

    assert [s0, s1] == struct_shots
    assert [s0, s1] == store.get([s0.id, s1.id], namespace)


@pytest.mark.asyncio
@pytest.mark.parametrize("store", lazy_async_stores)
async def test_async_crud(
    store: AsyncStore,
    str_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    await store.clear(namespace)
    await store.add(str_shots, mock_vectors, namespace)
    await store.add(str_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (_d0, s0), (_d1, s1) = await store.list(query_vector, namespace, limit=2)
    assert [s0, s1] == str_shots
    assert [s0, s1] == await store.get([s0.id, s1.id], namespace)

    await store.remove([s0.id], namespace)
    results = await store.list(query_vector, namespace, limit=2)
    assert len(results) == 1
    assert results[0].shot == s1
    assert [] == (await store.get([s0.id], namespace))

    await store.clear(namespace)
    results = await store.list(query_vector, namespace, limit=2)
    assert len(results) == 0
    assert [] == await store.get([s0.id], namespace)


@pytest.mark.asyncio
@pytest.mark.parametrize("store", lazy_async_stores)
async def test_async_structured_io(
    store: AsyncStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    await store.clear(namespace)
    await store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (_d0, s0), (_d1, s1) = await store.list(query_vector, namespace, limit=2)

    assert [s0, s1] == struct_shots
    assert [s0, s1] == await store.get([s0.id, s1.id], namespace)
