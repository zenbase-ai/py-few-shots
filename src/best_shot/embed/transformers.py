from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .base import Embedder


@dataclass
class TransformersEmbedder(Embedder):
    model: SentenceTransformer

    def embed(self, inputs: list[str]) -> list[list[float]]:
        return self.model.encode(inputs).tolist()
