from fastembed import TextEmbedding, SparseTextEmbedding

from few_shots.types import Vector

from .base import Embed


class FastEmbed(Embed):
    model: TextEmbedding | SparseTextEmbedding

    def __init__(self, model: TextEmbedding | SparseTextEmbedding):
        self.model = model

    def __call__(self, inputs: list[str]) -> list[Vector]:
        return [v.tolist() for v in self.model.embed(inputs)]
