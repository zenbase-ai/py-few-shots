from typing import Type

from best_shot.constant import MEMORY, LIST_OF_EMBEDDING_STORAGES
from best_shot.embedding_storage.base import BaseEmbeddingStorage
from best_shot.embedding_storage.memory import MemoryEmbeddingStorage


class EmbeddingStorageFactory:
    list_of_models = {
        MEMORY: MemoryEmbeddingStorage,
    }

    @staticmethod
    def get_embedding_storage_class(model_name: LIST_OF_EMBEDDING_STORAGES) -> Type[BaseEmbeddingStorage]:
        return EmbeddingStorageFactory.list_of_models[model_name]
