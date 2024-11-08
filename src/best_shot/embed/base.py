from abc import abstractmethod


class Embedder:
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
