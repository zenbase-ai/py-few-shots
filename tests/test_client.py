import pytest

from few_shots.client import FewShots
from few_shots.store.memory import MemoryStore
from few_shots.types import Shot


@pytest.fixture
def client():
    def embed(inputs: list[str]):
        return [[1] * 384] * len(inputs)

    return FewShots(embed=embed, store=MemoryStore())


def test_functional_flow(client: FewShots):
    inputs = {"a": 1}
    outputs = {"b": 2}
    shot = Shot(inputs, outputs)

    id = client.add(inputs, outputs)
    assert id == shot.id

    results = [r.shot for r in client.list(inputs)]
    assert results == [shot]
    assert shot == client.get(inputs)

    client.remove(inputs, outputs)
    assert [] == client.list(inputs)
    assert client.get(inputs) is None

    client.add(inputs, outputs)
    client.remove(inputs, outputs)
    assert [] == client.list(inputs)
    assert client.get(inputs) is None


def test_dispatch(client: FewShots):
    inputs = {"a": 1}
    outputs = {"b": 2}

    id = client.add(inputs, outputs)
    assert id == Shot(inputs, outputs).id

    client.remove([id])
    assert [] == client.list(inputs)

    [id] = client.add([(inputs, outputs)])
    client.remove([id])
    assert [] == client.list(inputs)

    client.add([(inputs, outputs)])
    client.remove([(inputs, outputs)])
    assert [] == client.list(inputs)

    [id] = client.add([(inputs, outputs, "id")])
    assert id == "id"

    client.remove([(inputs, outputs, "id")])
    assert [] == client.list(inputs)

    client.add([(inputs, outputs, "id")])
    client.remove(["id"])
    assert [] == client.list(inputs)


def test_string_flow(client: FewShots):
    inputs = "User question..."
    outputs = "AI answer..."
    shot = Shot(inputs, outputs)

    id = client.add(inputs, outputs)
    assert id == shot.id

    results = [r.shot for r in client.list(inputs, limit=1)]
    assert results == [shot]
    assert client.get(inputs) == shot

    results = [r.shot for r in client.list(inputs, limit=5)]
    assert results == [shot]

    client.remove(inputs, outputs)
    assert [] == [r.shot for r in client.list(inputs, limit=1)]

    client.add(inputs, outputs)
    client.clear()
    assert [] == [r.shot for r in client.list(inputs)]
    assert client.get(inputs) is None
