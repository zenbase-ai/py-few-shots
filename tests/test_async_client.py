import pytest

from few_shots.async_client import AsyncFewShots
from few_shots.store.memory import MemoryStore
from few_shots.types import Shot
from few_shots.utils.asyncio import asyncify_class


@asyncify_class
class AsyncMemoryStore(MemoryStore):
    """
    Async version of MemoryStore for testing only.
    """


@pytest.fixture(scope="function")
def client():
    async def embed(inputs: list[str]):
        return [[1] * 384] * len(inputs)

    return AsyncFewShots(embed=embed, store=AsyncMemoryStore())


@pytest.mark.asyncio
async def test_functional_flow(client: AsyncFewShots):
    inputs = {"a": 1}
    outputs = {"b": 2}
    shot = Shot(inputs, outputs)

    id = await client.add(inputs, outputs)
    assert id == shot.id

    results = [r.shot for r in await client.list(inputs)]
    assert results == [shot]
    assert shot == await client.get(inputs)

    await client.remove(inputs, outputs)
    assert [] == await client.list(inputs)

    await client.add(inputs, outputs)
    await client.clear()
    assert [] == await client.list(inputs)
    assert (await client.get(inputs)) is None


@pytest.mark.asyncio
async def test_dispatch(client: AsyncFewShots):
    inputs = {"a": 1}
    outputs = {"b": 2}
    shot = Shot(inputs, outputs)

    id = await client.add(inputs, outputs)
    assert id == shot.id
    assert shot == await client.get(inputs)
    assert [shot] == await client.get([inputs])

    await client.remove([id])
    assert [] == await client.list(inputs)

    [id] = await client.add([(inputs, outputs)])
    await client.remove([id])
    assert [] == await client.list(inputs)

    await client.add([(inputs, outputs)])
    await client.remove([(inputs, outputs)])
    assert [] == await client.list(inputs)

    [id] = await client.add([(inputs, outputs, "id")])
    assert id == "id"

    await client.remove([(inputs, outputs, "id")])
    assert [] == await client.list(inputs)

    await client.add([(inputs, outputs, "id")])
    await client.remove(["id"])
    assert [] == await client.list(inputs)


@pytest.mark.asyncio
async def test_string_flow(client: AsyncFewShots):
    inputs = "User question..."
    outputs = "AI answer..."
    shot = Shot(inputs, outputs)

    id = await client.add(inputs, outputs)
    assert id == shot.id
    assert shot == await client.get(inputs)

    results = [s.shot for s in await client.list(inputs, limit=5)]
    assert [shot] == results

    await client.remove(inputs, outputs)
    assert [] == await client.list(inputs, limit=1)

    await client.add(inputs, outputs)
    await client.clear()
    assert [] == await client.list(inputs)
    assert (await client.get(inputs)) is None
