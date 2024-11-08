from typing import List
from openai import OpenAI
import json
import hashlib
import numpy as np
from scipy.spatial.distance import cosine

from best_shot.constant import LIST_OF_EMBEDDING_GENERATORS, LIST_OF_EMBEDDING_STORAGES
from best_shot.embedding_generator.factory import EmbeddingGeneratorFactory
from best_shot.embedding_storage.factory import EmbeddingStorageFactory


class Shot:
    def __init__(self, inputs: dict, outputs: dict):
        self.inputs = inputs
        self.outputs = outputs
        self._id = self._generate_id()

    @property
    def id(self):
        return self._id

    def __repr__(self):
        return f"Shot(id={self.id}, inputs={self.inputs}, outputs={self.outputs})"

    def _generate_id(self):
        inputs_str = json.dumps(self.inputs, sort_keys=True)
        return hashlib.sha256(inputs_str.encode()).hexdigest()


class BestShots:
    def __init__(self, embedding_generator: LIST_OF_EMBEDDING_GENERATORS, embedding_storage: LIST_OF_EMBEDDING_STORAGES):
        self.embedding_generator = EmbeddingGeneratorFactory.get_embedding_generator_class(embedding_generator)()
        self.embedding_storage = EmbeddingStorageFactory.get_embedding_storage_class(embedding_storage)()
        self.shots = {}

    def add(self, shot: Shot, namespace: str):
        if namespace not in self.shots:
            self.shots[namespace] = {}
        if shot.id not in self.shots[namespace]:
            self.shots[namespace][shot.id] = shot
        self.embedding_storage.add(shot.id, self.embedding_generator.generate_embedding(json.dumps(shot.inputs)), namespace)

    def bulk_add(self, shots: List[Shot], namespace: str):
        if namespace not in self.shots:
            self.shots[namespace] = {}
        for shot in shots:
            if shot.id not in self.shots[namespace]:
                self.shots[namespace][shot.id] = shot
            self.embedding_storage.add(shot.id, self.embedding_generator.generate_embedding(json.dumps(shot.inputs)), namespace)

    def remove(self, shot: Shot, namespace: str):
        self.embedding_storage.remove(shot.id, namespace)
        if namespace in self.shots:
            if shot.id in self.shots[namespace]:
                del self.shots[namespace][shot.id]

    def remove_all(self, namespace: str):
        self.embedding_storage.remove_all(namespace)
        if namespace in self.shots:
            self.shots[namespace] = {}

    def get_best_shots(self, query: str, namespace: str, limit: int):
        query_embedding = self.embedding_generator.generate_embedding(query)
        embeddings = self.embedding_storage.get_namespace_embeddings(namespace)
        similarities = np.array([1 - cosine(query_embedding, emb) for shot_id, emb in embeddings.items()])
        shot_ids = list(embeddings.keys())

        # Get indices of the top 'limit' similar sentences
        top_indices = np.argsort(similarities)[-limit:][::-1]
        best_shot_ids = [shot_ids[i] for i in top_indices]
        return [self.shots[namespace][shot_id] for shot_id in best_shot_ids]
