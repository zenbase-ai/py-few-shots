from dataclasses import dataclass
from typing import overload

from few_shots.types import (
    Shot,
    dump_io_value,
    Datum,
    IO,
    ScoredShot,
    is_io_value,
)

from .embed.base import AsyncEmbed
from .store.base import AsyncStore


@dataclass
class AsyncFewShots:
    """
    Client for storing and retrieving few-shot examples.

    Combines an embedding model with a vector store to enable semantic search over examples.

    `add` is used to store examples, `remove` to delete them, `clear` to remove all examples in a namespace, and `list` to find similar examples to an input.
    """

    embed: AsyncEmbed
    store: AsyncStore

    @overload
    async def add(
        self,
        inputs: IO,
        outputs: IO,
        *,
        id: str = "",
        namespace: str = "default",
    ) -> str:
        """Add an example to the store.

        Args:
            inputs: Input to store
            outputs: Output to store
            id: ID for the example
            namespace: Namespace to store the example in
        """

    @overload
    async def add(
        self,
        data: list[Datum],
        *,
        namespace: str = "default",
    ) -> list[str]:
        """Add multiple examples to the store.

        Args:
            data: List of (input, output) or (input, output, id) tuples
            namespace: Namespace to store the examples in
        """

    async def add(
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
        vectors = await self.embed([shot.key for shot in shots])
        await self.store.add(shots, vectors, namespace)

        ids = [shot.id for shot in shots]
        return ids[0] if is_io_args else ids

    @overload
    async def remove(self, ids: list[str], *, namespace: str = "default"):
        """Remove one or more examples from the store.

        Args:
            ids: IDs of examples to remove
            namespace: Namespace to remove examples from
        """

    @overload
    async def remove(
        self,
        inputs: IO,
        outputs: IO | None = None,
        *,
        id: str = "",
        namespace: str = "default",
    ):
        """Remove one example from the store.

        Args:
            inputs: Input of example to remove
            outputs: Output of example to remove
            id: ID of example to remove
            namespace: Namespace to remove the example from
        """

    @overload
    async def remove(
        self,
        data: list[Datum],
        namespace: str = "default",
    ):
        """Remove multiple examples from the store.

        Args:
            data: List of (input, output) or (input, output, id) tuples
            namespace: Namespace to remove the examples from
        """

    async def remove(
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
        await self.store.remove(ids, namespace)

    async def clear(self, namespace: str = "default"):
        """Remove all examples from a namespace.

        Args:
            namespace: Namespace to clear
        """
        await self.store.clear(namespace)

    async def list(
        self,
        inputs: IO,
        *,
        namespace: str = "default",
        limit: int = 5,
    ) -> list[ScoredShot]:
        """Find similar examples to an input.

        Args:
            inputs: Input to find similar examples for
            namespace: Namespace to search in
            limit: Maximum number of examples to return

        Returns:
            List of (example, score) tuples, sorted by distance ascending
        """
        [vector] = await self.embed([dump_io_value(inputs)])
        return await self.store.list(vector, namespace, limit)
