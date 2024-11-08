from best_shot.embedding_storage.base import BaseEmbeddingStorage


class MemoryEmbeddingStorage(BaseEmbeddingStorage):
    def __init__(self):
        self.embedding_storage = {}

    def add(self, shot_id: str, embedding: list, namespace: str):
        if namespace not in self.embedding_storage:
            self.embedding_storage[namespace] = {}
        if shot_id not in self.embedding_storage[namespace]:
            self.embedding_storage[namespace][shot_id] = embedding

    def bulk_add(self, embeddings: dict, namespace: str):
        if namespace not in self.embedding_storage:
            self.embedding_storage[namespace] = {}
        for shot_id, embedding in embeddings.items():
            self.embedding_storage[namespace][shot_id] = embedding

    def get(self, shot_id: str, namespace: str):
        if namespace not in self.embedding_storage:
            return None
        if shot_id not in self.embedding_storage[namespace]:
            return None
        return self.embedding_storage[namespace][shot_id]

    def remove(self, shot_id: str, namespace: str):
        if namespace not in self.embedding_storage:
            return
        if shot_id not in self.embedding_storage[namespace]:
            return
        del self.embedding_storage[namespace][shot_id]

    def remove_all(self, namespace: str):
        if namespace not in self.embedding_storage:
            return
        self.embedding_storage[namespace] = {}

    def get_namespace_embeddings(self, namespace: str):
        if namespace not in self.embedding_storage:
            return {}
        return self.embedding_storage[namespace]

    def clear(self):
        self.embedding_storage = {}
