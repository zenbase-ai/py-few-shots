from itertools import chain
from typing import Iterable

from best_shot.types import Shot


def flatten(iterable: Iterable[Iterable]) -> list:
    return list(chain.from_iterable(iterable))


def shots_to_messages(shots: list[Shot]) -> list[dict]:
    return flatten(
        [
            [
                {"role": "user", "content": str(shot.inputs)},
                {"role": "assistant", "content": str(shot.outputs)},
            ]
            for shot in shots
        ]
    )
