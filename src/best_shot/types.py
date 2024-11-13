from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from typing import TypeVar
from uuid import uuid5, NAMESPACE_OID

import ujson


IO = TypeVar("IO", bound=dict | str)
Datum = TypeVar("Datum", bound=tuple[IO, IO] | tuple[IO, IO, str])
Embedding = TypeVar("Embedding", bound=list[float])


def is_io_value(value) -> bool:
    return isinstance(value, (dict, str))


def parse_io_value(value: str) -> IO:
    try:
        return ujson.loads(value)
    except ujson.JSONDecodeError:
        return value


def dump_io_value(data: IO) -> str:
    if isinstance(data, str):
        return data
    return ujson.dumps(data, sort_keys=True)


def id_io_value(data: IO) -> str:
    return str(uuid5(NAMESPACE_OID, dump_io_value(data)))


@dataclass
class Shot:
    inputs: IO
    outputs: IO
    id: str

    def __init__(self, inputs: IO, outputs: IO, id: str = ""):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.id = id or id_io_value(inputs)

    @cached_property
    def key(self) -> str:
        return dump_io_value(self.inputs)


ShotWithSimilarity = TypeVar("ShotWithSimilarity", bound=tuple[Shot, float])
