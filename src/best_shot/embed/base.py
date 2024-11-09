from abc import abstractmethod


class Embedder:
    @abstractmethod
    def embed(self, inputs: list[str]) -> list[list[float]]: ...
