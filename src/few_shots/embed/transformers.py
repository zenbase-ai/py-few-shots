from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from few_shots.types import Embedding
from few_shots.utils.asyncio import asyncify_class
from .base import AsyncEmbedder, Embedder


@dataclass
class TransformersEmbedder(Embedder):
    model: SentenceTransformer

    def __call__(self, inputs: list[str]) -> list[Embedding]:
        return self.model.encode(inputs).tolist()


@asyncify_class
class AsyncTransformersEmbedder(TransformersEmbedder, AsyncEmbedder): ...
