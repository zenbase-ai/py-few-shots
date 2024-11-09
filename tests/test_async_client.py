import pytest
from sentence_transformers import SentenceTransformer

from best_shot.async_client import AsyncBestShots
from best_shot.embed.transformers import AsyncTransformersEmbedder
from best_shot.store.memory import AsyncMemoryStore
from best_shot.types import Shot


@pytest.fixture(scope="function")
def client():
    return AsyncBestShots(
        embed=AsyncTransformersEmbedder(model=SentenceTransformer("all-MiniLM-L6-v2")),
        store=AsyncMemoryStore(),
    )


@pytest.mark.anyio
async def test_functional_flow(client: AsyncBestShots):
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
async def test_dispatch(client: AsyncBestShots):
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
async def test_string_flow(client: AsyncBestShots):
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
