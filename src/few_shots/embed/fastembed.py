from dataclasses import dataclass
from fastembed import TextEmbedding, SparseTextEmbedding

from few_shots.types import Vector

from .base import Embed


Model = TextEmbedding | SparseTextEmbedding


@dataclass
class FastEmbed(Embed):
    model: Model

    def __call__(self, inputs: list[str]) -> list[Vector]:
        return [v.tolist() for v in self.model.embed(inputs)]
