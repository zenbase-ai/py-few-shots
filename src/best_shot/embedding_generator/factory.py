from typing import Type

from best_shot.constant import TRANSFORMER, LIST_OF_EMBEDDING_GENERATORS
from best_shot.embedding_generator.base import BaseEmbeddingGenerator
from best_shot.embedding_generator.transformer import TransformerEmbeddingGenerator


class EmbeddingGeneratorFactory:
    list_of_models = {
        TRANSFORMER: TransformerEmbeddingGenerator,
        # "openai": OpenAIEmbeddingGenerator,
    }

    @staticmethod
    def get_embedding_generator_class(model_name: LIST_OF_EMBEDDING_GENERATORS) -> Type[BaseEmbeddingGenerator]:
        return EmbeddingGeneratorFactory.list_of_models[model_name]
