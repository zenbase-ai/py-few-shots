import pytest
from sentence_transformers import SentenceTransformer

from few_shots.async_client import AsyncFewShots
from few_shots.embed.transformers import AsyncTransformersEmbedder
from few_shots.store.memory import AsyncMemoryStore
from few_shots.types import Shot


@pytest.fixture(scope="function")
def client():
    return AsyncFewShots(
        embed=AsyncTransformersEmbedder(model=SentenceTransformer("all-MiniLM-L6-v2")),
        store=AsyncMemoryStore(),
    )


@pytest.mark.anyio
async def test_functional_flow(client: AsyncFewShots):
    inputs = {"a": 1}
    outputs = {"b": 2}

    id = await client.add(inputs, outputs)
    assert id == Shot(inputs, outputs).id

    results = await client.list(inputs, limit=1)
    results = [shot for shot, _ in results]
    assert results == [Shot(inputs, outputs)]

    results = await client.list(inputs, limit=5)
    results = [shot for shot, _ in results]
    assert results == [Shot(inputs, outputs)]

    await client.remove(inputs, outputs)
    assert [] == await client.list(inputs, limit=1)

    await client.add(inputs, outputs)
    await client.clear()
    assert [] == await client.list(inputs)


@pytest.mark.anyio
async def test_dispatch(client: AsyncFewShots):
    inputs = {"a": 1}
    outputs = {"b": 2}

    id = await client.add(inputs, outputs)
    assert id == Shot(inputs, outputs).id

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


@pytest.mark.anyio
async def test_string_flow(client: AsyncFewShots):
    inputs = "User question..."
    outputs = "AI answer..."

    id = await client.add(inputs, outputs)
    assert id == Shot(inputs, outputs).id

    results = await client.list(inputs, limit=1)
    results = [shot for shot, _ in results]
    assert results == [Shot(inputs, outputs)]

    results = await client.list(inputs, limit=5)
    results = [shot for shot, _ in results]
    assert results == [Shot(inputs, outputs)]

    await client.remove(inputs, outputs)
    assert [] == await client.list(inputs, limit=1)

    await client.add(inputs, outputs)
    await client.clear()
    assert [] == await client.list(inputs)
