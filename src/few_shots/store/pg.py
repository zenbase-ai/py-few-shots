from typing import Literal, TypeVar
from uuid import UUID
from psycopg import AsyncConnection, Connection
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector, register_vector_async

from few_shots.types import ScoredShot, Shot, Vector

from .base import Store


DistanceType = TypeVar("DistanceType", bound=Literal["cosine", "l2"])


class PGStore(Store):
    _sql: "SQLHelper"
    connection: Connection

    def __init__(
        self,
        connection: Connection[TupleRow],
        tablename: str,
        schema: str = "public",
        distance: DistanceType = "cosine",
    ):
        self._sql = SQLHelper(tablename, schema, distance)
        self.connection = connection

    def setup(self, dimensions: int, m: int = 64, ef_construction: int = 128):
        """
        Sets up the database table and index for vector similarity search.
        Idempotent, will not re-create the table if it already exists.
        Will use DiskANN as an index if available, HNSW otherwise.

        Args:
            dimensions: The number of dimensions in the vectors to be stored
            m: Number of connections per element in HNSW index (pgvector only)
            ef_construction: Size of the dynamic candidate list for constructing HNSW index (pgvector only)

        Raises:
            ValueError: If pgvector extension is not installed in the database
        """
        with self.connection.cursor() as cursor:
            cursor.execute(self._sql.pgvector_create())
            cursor.execute(self._sql.table_create(dimensions))

            cursor.execute(self._sql.diskann_check())
            if cursor.fetchone():
                cursor.execute(self._sql.diskann_index())
            else:
                cursor.execute(self._sql.pgvector_index(m, ef_construction))

        register_vector(self.connection)

    def teardown(self):
        with self.connection.cursor() as cursor:
            cursor.execute(self._sql.table_drop())

    def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        with self.connection.cursor() as cursor:
            cursor.executemany(
                self._sql.upsert(),
                self._sql.shots_to_upsert_tuples(shots, vectors, namespace),
            )

    def remove(self, ids: list[str], _namespace: str):
        with self.connection.cursor() as cursor:
            cursor.execute(self._sql.remove(), (ids,))

    def clear(self, namespace: str):
        with self.connection.cursor() as cursor:
            cursor.execute(self._sql.clear(), (namespace,))

    def list(self, vector: Vector, namespace: str, limit: int) -> list[ScoredShot]:
        with self.connection.cursor() as cursor:
            cursor.execute(self._sql.select(), (vector, namespace, limit))
            return self._sql.tuples_to_scored_shots(cursor.fetchall())


class AsyncPGStore(Store):
    _sql: "SQLHelper"
    connection: AsyncConnection

    def __init__(
        self,
        connection: AsyncConnection[TupleRow],
        tablename: str,
        schema: str = "public",
        distance: DistanceType = "cosine",
    ):
        self._sql = SQLHelper(tablename, schema, distance)
        self.connection = connection

    async def setup(self, dimensions: int, m: int = 64, ef_construction: int = 128):
        """
        Sets up the database table and index for vector similarity search.
        Idempotent, will not re-create the table if it already exists.
        Will use DiskANN as an index if available, HNSW otherwise.
        """
        async with self.connection.cursor() as cursor:
            await cursor.execute(self._sql.pgvector_create())
            await cursor.execute(self._sql.table_create(dimensions))

            await cursor.execute(self._sql.diskann_check())
            if await cursor.fetchone():
                await cursor.execute(self._sql.diskann_index())
            else:
                await cursor.execute(self._sql.pgvector_index(m, ef_construction))

        await register_vector_async(self.connection)

    async def teardown(self):
        async with self.connection.cursor() as cursor:
            await cursor.execute(self._sql.table_drop())

    async def add(self, shots: list[Shot], vectors: list[Vector], namespace: str):
        async with self.connection.cursor() as cursor:
            await cursor.executemany(
                self._sql.upsert(),
                self._sql.shots_to_upsert_tuples(shots, vectors, namespace),
            )

    async def remove(self, ids: list[str], _namespace: str):
        async with self.connection.cursor() as cursor:
            await cursor.execute(self._sql.remove(), (ids,))

    async def clear(self, namespace: str):
        async with self.connection.cursor() as cursor:
            await cursor.execute(self._sql.clear(), (namespace,))

    async def list(
        self,
        vector: Vector,
        namespace: str,
        limit: int,
    ) -> list[ScoredShot]:
        async with self.connection.cursor() as cursor:
            await cursor.execute(self._sql.select(), (vector, namespace, limit))
            return self._sql.tuples_to_scored_shots(await cursor.fetchall())


class SQLHelper:
    tablename: str
    schema: str
    distance: DistanceType

    def __init__(
        self,
        tablename: str,
        schema: str = "public",
        distance: DistanceType = "cosine",
    ):
        self.tablename = tablename
        self.schema = schema
        self.distance = distance

    def table_create(self, vector_dimensions: int):
        return f"""\
        CREATE TABLE IF NOT EXISTS {self.schema}.{self.tablename} (
            id UUID PRIMARY KEY,
            namespace VARCHAR(256) NOT NULL,
            payload JSONB NOT NULL,
            vector VECTOR({vector_dimensions}) NOT NULL
        );
        """

    def table_drop(self):
        return f"DROP TABLE IF EXISTS {self.schema}.{self.tablename};"

    @staticmethod
    def pgvector_check():
        return "SELECT * FROM pg_extension WHERE extname = 'vector'"

    @staticmethod
    def pgvector_create():
        return "CREATE EXTENSION IF NOT EXISTS vector"

    def pgvector_index(self, m: int, ef_construction: int):
        return f"""\
        CREATE INDEX IF NOT EXISTS index_{self.tablename}_vector
        ON {self.schema}.{self.tablename}
        USING hnsw (vector vector_{self.distance}_ops)
        WITH (m = {m}, ef_construction = {ef_construction});
        """

    @staticmethod
    def diskann_check():
        return "SELECT * FROM pg_extension WHERE extname = 'vectorscale';"

    @staticmethod
    def diskann_create():
        return "CREATE EXTENSION IF NOT EXISTS vectorscale;"

    def diskann_index(self):
        return f"""\
        CREATE INDEX IF NOT EXISTS index_{self.tablename}_vector
        ON {self.schema}.{self.tablename}
        USING DISKANN (vector);
        """

    def upsert(self):
        return f"""\
        INSERT INTO {self.schema}.{self.tablename} (id, namespace, payload, vector)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            namespace = EXCLUDED.namespace,
            payload = EXCLUDED.payload,
            vector = EXCLUDED.vector;
        """

    def shots_to_upsert_tuples(
        self,
        shots: list[Shot],
        vectors: list[Vector],
        namespace: str,
    ) -> list[tuple[str, str, dict, Vector]]:
        return [
            (
                shot.id,
                namespace,
                Jsonb({"inputs": shot.inputs, "outputs": shot.outputs}),
                vector,
            )
            for shot, vector in zip(shots, vectors)
        ]

    def select(self):
        return f"""\
        SELECT {self.tablename}.id,
               {self.tablename}.payload,
               {self.tablename}.vector <-> %s::vector AS distance
        FROM {self.schema}.{self.tablename}
        WHERE {self.tablename}.namespace = %s
        ORDER BY distance ASC
        LIMIT %s;
        """

    def tuples_to_scored_shots(
        self,
        tuples: list[tuple[UUID, dict, float]],
    ) -> list[ScoredShot]:
        return [
            (
                Shot(payload["inputs"], payload["outputs"], str(id)),
                distance,
            )
            for (id, payload, distance) in tuples
        ]

    def remove(self):
        return f"DELETE FROM {self.schema}.{self.tablename} WHERE id = ANY(%s)"

    def clear(self):
        return f"DELETE FROM {self.schema}.{self.tablename} WHERE namespace = %s"
