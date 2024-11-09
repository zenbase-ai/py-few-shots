from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from best_shot.utils.asyncify import asyncify_class
from .base import AsyncEmbedder, Embedder


@dataclass
class TransformersEmbedder(Embedder):
    model: SentenceTransformer

    def __call__(self, inputs: list[str]) -> list[list[float]]:
        return self.model.encode(inputs).tolist()


@asyncify_class
class AsyncTransformersEmbedder(TransformersEmbedder, AsyncEmbedder): ...
