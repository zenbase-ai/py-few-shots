class BaseEmbeddingStorage:
    def add(self, shot_id: str, embedding: list, namespace: str):
        pass

    def get(self, shot_id: str, namespace: str):
        pass

    def remove(self, shot_id: str, namespace: str):
        pass

    def remove_all(self, namespace: str):
        pass

    def get_namespace_embeddings(self, namespace: str):
        pass

    def clear(self):
        pass
