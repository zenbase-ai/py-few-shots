from sentence_transformers import SentenceTransformer

from few_shots.types import Vector
from .base import Embed


class TransformersEmbed(Embed):
    model: SentenceTransformer

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, inputs: list[str]) -> list[Vector]:
        return self.model.encode(inputs).tolist()
