from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from few_shots.types import Vector
from .base import Embed


@dataclass
class TransformersEmbed(Embed):
    model: SentenceTransformer

    def __call__(self, inputs: list[str]) -> list[Vector]:
        return self.model.encode(inputs).tolist()
