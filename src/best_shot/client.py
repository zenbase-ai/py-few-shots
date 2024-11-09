from dataclasses import dataclass
from typing import overload

from best_shot.types import Shot, data_key, Datum, IO, ShotWithSimilarity, is_io_value

from .embed.base import Embedder
from .store.base import Store


@dataclass
class BestShots:
    embed: Embedder
    store: Store

    @overload
    def add(
        self,
        inputs: IO,
        outputs: IO,
        *,
        id: str = "",
        namespace: str = "default",
    ) -> str: ...

    @overload
    def add(
        self,
        data: list[Datum],
        *,
        namespace: str = "default",
    ) -> list[str]: ...

    def add(
        self,
        maybe_inputs: IO | list[Datum],
        maybe_outputs: IO | None = None,
        *,
        id: str = "",
        namespace: str = "default",
    ) -> str | list[str]:
        is_io_args = is_io_value(maybe_inputs) and is_io_value(maybe_outputs)
        data: list[Datum] = (
            [(maybe_inputs, maybe_outputs, id)] if is_io_args else maybe_inputs
        )
        shots = [Shot(*datum) for datum in data]
        embeddings = self.embed([shot.key for shot in shots])
        self.store.add(shots, embeddings, namespace)

        ids = [shot.id for shot in shots]
        return ids[0] if is_io_args else ids

    @overload
    def remove(
        self,
        ids: list[str],
        *,
        namespace: str = "default",
    ): ...

    @overload
    def remove(
        self,
        inputs: dict,
        outputs: dict,
        *,
        id: str = "",
        namespace: str = "default",
    ): ...

    @overload
    def remove(
        self,
        data: list[Datum],
        *,
        namespace: str = "default",
    ): ...

    def remove(
        self,
        maybe_inputs: IO | list[Datum],
        maybe_outputs: IO | None = None,
        *,
        id: str = "",
        namespace: str = "default",
    ):
        is_io_args = is_io_value(maybe_inputs) and is_io_value(maybe_outputs)
        data: list[Datum] = (
            [(maybe_inputs, maybe_outputs, id)] if is_io_args else maybe_inputs
        )
        ids = data if isinstance(data[0], str) else [Shot(*datum).id for datum in data]
        self.store.remove(ids, namespace)

    def clear(self, namespace: str = "default"):
        self.store.clear(namespace)

    def list(
        self,
        inputs: IO,
        *,
        namespace: str = "default",
        limit: int = 5,
    ) -> list[ShotWithSimilarity]:
        [embedding] = self.embed([data_key(inputs)])
        return self.store.list(embedding, namespace, limit)
