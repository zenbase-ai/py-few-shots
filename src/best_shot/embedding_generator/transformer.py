from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
from scipy.spatial.distance import cosine


from best_shot.embedding_generator.base import BaseEmbeddingGenerator

model = SentenceTransformer('all-MiniLM-L6-v2')


class TransformerEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self):
        pass

    def generate_embedding(self, text: str):
        embedding = model.encode(text)
        return embedding

    def generate_embeddings(self, texts: list[str]):
        embeddings = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))
        return embeddings
