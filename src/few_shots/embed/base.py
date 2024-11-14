from abc import abstractmethod

from few_shots.types import Vector


class Embed:
    @abstractmethod
    def __call__(self, inputs: list[str]) -> list[Vector]: ...


class AsyncEmbed(Embed):
    @abstractmethod
    async def __call__(self, inputs: list[str]) -> list[Vector]: ...
