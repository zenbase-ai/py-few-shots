from itertools import chain
from typing import Iterable

import ujson

from best_shot.types import IO, Shot


def to_str(value: IO) -> str:
    if isinstance(value, dict):
        return ujson.dumps(value)
    return str(value)


def flatten(iterable: Iterable[Iterable]) -> list:
    return list(chain.from_iterable(iterable))


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
