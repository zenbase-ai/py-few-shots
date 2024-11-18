import pytest

from chromadb import HttpClient, AsyncHttpClient
from psycopg import AsyncConnection, Connection
from pymilvus import MilvusClient
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
from qdrant_client.local.qdrant_local import QdrantLocal
import weaviate

from few_shots.store.chroma import ChromaStore, AsyncChromaStore
from few_shots.store.milvus import AsyncMilvusStore, MilvusStore
from few_shots.store.pg import AsyncPGStore, PGStore
from few_shots.store.qdrant import AsyncQdrantStore, QdrantStore, Distance
from few_shots.store.weaviate import AsyncWeaviateStore, WeaviateStore
from few_shots.types import Shot, Vector


# Shared fixtures
@pytest.fixture
def str_shots():
    return [Shot("input1", "output1"), Shot("input2", "output2")]


@pytest.fixture
def struct_shots():
    return [
        Shot({"key": "input1"}, {"key": "output1"}),
        Shot({"key": "input2"}, {"key": "output2"}),
    ]


@pytest.fixture
def mock_vectors() -> list[Vector]:
    return [[1.0, 2.0], [3.0, 4.0]]


@pytest.fixture
def namespace() -> str:
    return "test"


# Chroma fixtures
@pytest.fixture
def chroma_store():
    client = HttpClient()
    collection = client.create_collection("test", get_or_create=True)
    return ChromaStore(collection)


@pytest.fixture
async def async_chroma_store():
    client = await AsyncHttpClient()
    collection = await client.create_collection("test", get_or_create=True)
    return AsyncChromaStore(collection)


# PostgreSQL fixtures
@pytest.fixture
def pg_conn():
    with Connection.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres"
    ) as c:
        yield c


@pytest.fixture
def pg_store(pg_conn: Connection):
    s = PGStore(connection=pg_conn, tablename="few_shots")
    s.setup(dimensions=2)
    yield s
    s.teardown()


# PostgreSQL async fixtures
@pytest.fixture
async def async_pg_conn():
    async with await AsyncConnection.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres"
    ) as c:
        yield c


@pytest.fixture
async def async_pg_store(async_pg_conn: AsyncConnection):
    s = AsyncPGStore(connection=async_pg_conn, tablename="few_shots")
    await s.setup(dimensions=2)
    yield s
    await s.teardown()


# Qdrant fixtures
@pytest.fixture
def qdrant_store():
    s = QdrantStore(client=QdrantLocal(":memory:"), collection_name="test")
    s.setup(size=2, distance=Distance.COSINE)
    yield s
    s.teardown()


# Qdrant async fixtures
@pytest.fixture
async def async_qdrant_store():
    s = AsyncQdrantStore(client=AsyncQdrantLocal(":memory:"), collection_name="test")
    await s.setup(size=2, distance=Distance.COSINE)
    yield s
    await s.teardown()


# Weaviate fixtures
@pytest.fixture
def weaviate_client():
    with weaviate.connect_to_local() as c:
        yield c


@pytest.fixture
def weaviate_store(weaviate_client):
    s = WeaviateStore(weaviate_client, "test")
    s.setup()
    yield s
    s.teardown()


# Weaviate async fixtures
@pytest.fixture
async def async_weaviate_client():
    async with weaviate.use_async_with_local() as c:
        yield c


@pytest.fixture
async def async_weaviate_store(async_weaviate_client):
    s = AsyncWeaviateStore(async_weaviate_client, "test")
    await s.setup()
    yield s
    await s.teardown()


@pytest.fixture
def milvus_store():
    s = MilvusStore(MilvusClient(), "test")
    s.setup()
    yield s
    s.teardown()


@pytest.fixture
async def async_milvus_store():
    s = AsyncMilvusStore(MilvusClient(), "test")
    await s.setup()
    yield s
    await s.teardown()
