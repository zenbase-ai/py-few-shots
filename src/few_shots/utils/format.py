from itertools import chain
from typing import overload

from few_shots.types import ScoredShot, Shot, dump_io_value


@overload
def shots_to_messages(scored_shots: list[ScoredShot]) -> list[dict]: ...


@overload
def shots_to_messages(shots: list[Shot]) -> list[dict]: ...


def shots_to_messages(shots: list[ScoredShot] | list[Shot]) -> list[dict]:
    if not shots:
        return []

    sample = shots[0]
    if isinstance(sample, ScoredShot):
        shots = [shot for shot, _ in shots]

    return list(
        chain.from_iterable(
            [
                {"role": "user", "content": dump_io_value(shot.inputs)},
                {"role": "assistant", "content": dump_io_value(shot.outputs)},
            ]
            for shot in shots
        )
    )
