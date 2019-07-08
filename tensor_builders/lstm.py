from typing import List

from tensorflow.keras.layers import Embedding, Dense
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.layers import LSTM

from .base import BaseTensorBuilder


class LSTMTensorBuilder(BaseTensorBuilder):
    variables = ['vocab_size', 'vector_size', 'embedding_matrix', 'trainable_embedding', 'output_size']

    def __init__(self, vocab_size: int = None, vector_size: int = None, embedding_matrix: List[List[int]] = None,
                 trainable_embedding: bool = False, output_size: int = 16, *args, **kwargs):
        super().__init__(input_shape=(None,), *args, **kwargs)

        self.output_size = output_size
        self.trainable_embedding = trainable_embedding
        self.embedding_matrix = embedding_matrix
        self.vector_size = vector_size
        self.vocab_size = vocab_size

    def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
        tensor = Embedding(self.vocab_size, self.vector_size, weights=[self.embedding_matrix],
                           trainable=self.trainable_embedding)(input_tensor)
        tensor = LSTM(self.output_size)(tensor)
        tensor = Dense(self.output_size, activation='relu')(tensor)

        return tensor
