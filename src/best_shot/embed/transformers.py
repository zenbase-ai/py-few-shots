from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .base import Embedder


@dataclass
class TransformersEmbedder(Embedder):
    model: SentenceTransformer

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
