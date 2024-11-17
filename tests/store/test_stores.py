"""
Since some stores are stateful, make sure to clear them at the beginning of each test.
"""

import pytest
from pytest_lazy_fixtures import lf

from few_shots.store.base import Store, AsyncStore
from few_shots.types import Shot, Vector


providers = ["chroma", "pg", "qdrant", "weaviate"]
sync_providers = [] + providers  # TODO: Add Milvus
async_providers = [] + providers


lazy_sync_stores = [lf(f"{p}_store") for p in sync_providers]
lazy_async_stores = [lf(f"async_{p}_store") for p in async_providers]


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
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    (_d0, s0), (_d1, s1) = results
    assert [s0, s1] == str_shots

    store.remove([s0.id], namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 1

    store.clear(namespace)
    results = store.list(query_vector, namespace, limit=2)
    assert len(results) == 0


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
    (_d0, s0), (_d1, s1) = store.list(
        query_vector,
        namespace,
        limit=2,
    )

    assert [s0, s1] == struct_shots


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
    results = await store.list(query_vector, namespace, limit=2)
    assert len(results) == 2
    (_d0, s0), (_d1, s1) = results
    assert [s0, s1] == str_shots

    await store.remove([str_shots[0].id], namespace)
    results = await store.list(query_vector, namespace, limit=2)
    assert len(results) == 1

    await store.clear(namespace)
    results = await store.list(query_vector, namespace, limit=2)
    assert len(results) == 0


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
    (_d0, s0), (_d1, s1) = await store.list(
        query_vector,
        namespace,
        limit=2,
    )

    assert [s0, s1] == struct_shots
