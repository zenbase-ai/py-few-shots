from psycopg import AsyncConnection, Connection
import pytest


from few_shots.store.pg import AsyncPGStore, PGStore
from few_shots.types import Vector, Shot


@pytest.fixture
def connection():
    with Connection.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres"
    ) as c:
        yield c


@pytest.fixture
def store(connection: Connection):
    s = PGStore(connection=connection, tablename="few_shots")
    s.setup(dimensions=2)
    yield s
    s.teardown()


def test_crud(store: PGStore, str_shots: list[Shot], mock_vectors: list[Vector]):
    namespace = "test"

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
    store: PGStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
):
    namespace = "test"

    store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (shot_0, _d1), (shot_1, _d1) = store.list(query_vector, namespace, limit=2)

    assert shot_0 == struct_shots[0]
    assert shot_1 == struct_shots[1]


@pytest.fixture
async def async_connection():
    async with await AsyncConnection.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres"
    ) as c:
        yield c


@pytest.fixture
async def async_store(async_connection: AsyncConnection):
    s = AsyncPGStore(connection=async_connection, tablename="few_shots")
    await s.setup(dimensions=2)
    yield s
    await s.teardown()


@pytest.mark.anyio
async def test_async_crud(
    async_store: AsyncPGStore,
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
    async_store: AsyncPGStore,
    struct_shots: list[Shot],
    mock_vectors: list[Vector],
    namespace: str,
):
    await async_store.add(struct_shots, mock_vectors, namespace)

    query_vector = mock_vectors[0]
    (shot_0, _d1), (shot_1, _d1) = await async_store.list(
        query_vector, namespace, limit=2
    )

    assert shot_0 == struct_shots[0]
    assert shot_1 == struct_shots[1]
