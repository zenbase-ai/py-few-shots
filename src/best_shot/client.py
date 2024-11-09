from dataclasses import dataclass
from typing import TypeVar, overload

from best_shot.types import Shot, data_key

from .embed.base import Embedder
from .store.base import ShotWithSimilarity, Store


Datum = TypeVar("Datum", bound=tuple[dict, dict] | tuple[dict, dict, str] | tuple[str, str] | tuple[str, str, str])


@dataclass
class BestShots:
    embedder: Embedder
    store: Store

    @overload
    def add(
        self, inputs: dict | str, outputs: dict | str, id: str = "", namespace: str = "default"
    ) -> str: ...

    @overload
    def add(self, data: list[Datum], namespace: str = "default") -> list[str]: ...

    def add(
        self,
        maybe_inputs,
        maybe_outputs: dict | str | None = None,
        maybe_id: str = "",
        namespace: str = "default",
    ) -> str | list[str]:
        is_io = (isinstance(maybe_inputs, dict) and isinstance(maybe_outputs, dict)) or (
            isinstance(maybe_inputs, str) and isinstance(maybe_outputs, str)
        )
        data: list[tuple[dict, dict, str]] | list[tuple[str, str, str]] = (
            [(maybe_inputs, maybe_outputs, maybe_id)] if is_io else maybe_inputs
        )
        shots = [Shot(*datum) for datum in data]
        embeddings = self.embedder.embed([shot.key for shot in shots])
        self.store.add(shots, embeddings, namespace)

        ids = [shot.id for shot in shots]
        return ids[0] if is_io else ids

    @overload
    def remove(self, id: str, namespace: str = "default"): ...

    @overload
    def remove(self, ids: list[str], namespace: str = "default"): ...

    @overload
    def remove(
        self, inputs: dict | str, outputs: dict | str, id: str = "", namespace: str = "default"
    ): ...

    @overload
    def remove(self, data: list[Datum], namespace: str = "default"): ...

    def remove(
        self,
        maybe_inputs,
        maybe_outputs: dict | str | None = None,
        id: str = "",
        namespace: str = "default",
    ):
        is_io = isinstance(maybe_inputs, dict) and isinstance(maybe_outputs, dict) or (
            isinstance(maybe_inputs, str) and isinstance(maybe_outputs, str)
        )
        data: list[tuple[dict, dict, str]] | list[tuple[str, str, str]] = (
            [(maybe_inputs, maybe_outputs, id)] if is_io else maybe_inputs
        )
        match data:
            case str() as id:
                return self.remove([id], namespace)
            case tuple() as datum:
                return self.remove([Shot(*datum).id], namespace)
            case list():
                is_ids = isinstance(data[0], str)
                ids = data if is_ids else [Shot(*datum).id for datum in data]
                return self.store.remove(ids, namespace)
            case _:
                raise ValueError(f"Invalid data type: {type(data)}")

    def clear(self, namespace: str = "default"):
        self.store.clear(namespace)

    def list(
        self,
        inputs: dict | str,
        namespace: str = "default",
        limit: int = 5,
    ) -> list[ShotWithSimilarity]:
        embedding = self.embedder.embed([data_key(inputs)])[0]
        return self.store.list(embedding, namespace, limit)
