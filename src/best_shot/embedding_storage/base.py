class BaseEmbeddingStorage:
    def add(self, embedding: list, namespace: str):
        pass

    def get(self, namespace: str):
        pass

    def remove(self, namespace: str):
        pass

    def list(self, namespace: str):
        pass

    def clear(self):
        pass
