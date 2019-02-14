from tensorflow.python.keras.engine.training import Model
from tensor_builders.deep import DeepTensorBuilder
from tensor_builders.cnn import CNNTensorBuilder
from typing import List


class SingletonClassifierModelBuilder:
    def __init__(self, use_words_feature: bool = True, use_context_feature: bool = True,
                 use_syntactic_feature: bool = True, syntactic_features_num: int = None,
                 embedding_matrix: List[List[int]] = None) -> None:

        self.embedding_matrix = embedding_matrix
        self.syntactic_features_num = syntactic_features_num
        self.use_syntactic_feature = use_syntactic_feature
        self.use_context_feature = use_context_feature
        self.use_words_feature = use_words_feature

        if self.use_syntactic_feature and self.syntactic_features_num is None:
            raise Exception('Specify syntactic features number')

        if (self.use_words_feature or self.use_context_feature) and self.embedding_matrix is None:
            raise Exception('Insert embedding matrix')

        self.deep_tensor_builder = DeepTensorBuilder()
        self.cnn_tensor_builder = CNNTensorBuilder(
            input_length=10, vocab_size=len(embedding_matrix), vector_size=len(embedding_matrix[0]),
            embedding_matrix=embedding_matrix, filter_sizes=[2, 3, 4], num_filters=64, trainable_embedding=True
        )

    def create_model(self) -> Model:
        inputs = []
        tensors = []

        if self.use_words_feature:
            inp, tensor = self.cnn_tensor_builder.create_tensor()
            inputs.append(inp)
            inputs.append(tensor)
