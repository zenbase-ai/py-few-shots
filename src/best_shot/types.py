from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from typing import TypeVar

import ujson


IO = TypeVar("IO", bound=dict | str)
Datum = TypeVar("Datum", bound=tuple[IO, IO] | tuple[IO, IO, str])
Embedding = TypeVar("Embedding", bound=list[float])


def is_io_value(value) -> bool:
    return isinstance(value, (dict, str))


def data_key(data: dict) -> str:
    return ujson.dumps(data, sort_keys=True)


def data_hash(data: dict) -> str:
    return sha256(data_key(data).encode()).hexdigest()


@dataclass
class Shot:
    inputs: IO
    outputs: IO
    id: str

    def __init__(self, inputs: IO, outputs: IO, id: str = ""):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.id = id or data_hash(inputs)

    @cached_property
    def key(self) -> str:
        return data_key(self.inputs)


ShotWithSimilarity = TypeVar("ShotWithSimilarity", bound=tuple[Shot, float])
