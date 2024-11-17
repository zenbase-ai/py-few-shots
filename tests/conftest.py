import pytest

from few_shots.types import Shot, id_io_value


@pytest.fixture(scope="package")
def anyio_backend():
    return "asyncio"


@pytest.fixture
def str_shots():
    return [
        Shot("input1", "output1", id_io_value("id1")),
        Shot("input2", "output2", id_io_value("id2")),
    ]


@pytest.fixture
def struct_shots():
    return [
        Shot({"key": "input1"}, {"key": "output1"}, id_io_value("id1")),
        Shot({"key": "input2"}, {"key": "output2"}, id_io_value("id2")),
    ]


@pytest.fixture
def mock_vectors():
    return [[1.0, 2.0], [3.0, 4.0]]


@pytest.fixture
def namespace():
    return "test"
