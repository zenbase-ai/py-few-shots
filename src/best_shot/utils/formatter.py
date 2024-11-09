from itertools import chain
from typing import Iterable

import ujson

from best_shot.types import IO, Shot


def flatten(iterable: Iterable[Iterable]) -> list:
    return list(chain.from_iterable(iterable))


def to_str(value: IO) -> str:
    match value:
        case str():
            return value
        case dict():
            return ujson.dumps(value)
        case _:
            raise ValueError(f"Unsupported value type: {type(value)}")


def shots_to_messages(shots: list[Shot]) -> list[dict]:
    return flatten(
        [
            [
                {"role": "user", "content": to_str(shot.inputs)},
                {"role": "assistant", "content": to_str(shot.outputs)},
            ]
            for shot in shots
        ]
    )
