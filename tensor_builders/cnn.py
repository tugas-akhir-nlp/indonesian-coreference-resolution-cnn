from typing import List

from tensorflow.keras.layers import Embedding, Dense, Concatenate, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.python.framework.ops import Tensor

from .base import BaseTensorBuilder


class CNNTensorBuilder(BaseTensorBuilder):
    variables = ['input_length', 'vocab_size', 'vector_size', 'embedding_matrix', 'trainable_embedding',
                 'filter_sizes', 'num_filters', 'output_size']

    def __init__(self, input_length: int = None, vocab_size: int = None, vector_size: int = None,
                 embedding_matrix: List[List[int]] = None, filter_sizes: List[int] = None, num_filters: int = None,
                 trainable_embedding: bool = False, output_size: int = 16, *args, **kwargs):

        if input_length is None:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(input_shape=(input_length,), *args, **kwargs)

        self.output_size = output_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.trainable_embedding = trainable_embedding
        self.embedding_matrix = embedding_matrix
        self.vector_size = vector_size
        self.vocab_size = vocab_size
        self.input_length = input_length

    def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
        tensor = Embedding(self.vocab_size, self.vector_size, weights=[self.embedding_matrix],
                           input_length=self.input_length, trainable=self.trainable_embedding)(input_tensor)
        tensor = Reshape((self.input_length, self.vector_size, 1))(tensor)
        convolution_layers = [
            Conv2D(self.num_filters, kernel_size=(filter_size, self.vector_size), padding='valid',
                   kernel_initializer='normal', activation='relu')(tensor)

            for filter_size in self.filter_sizes
        ]
        max_pool_layers = [
            MaxPool2D(pool_size=(self.input_length - self.filter_sizes[i] + 1, 1), strides=(1, 1),
                      padding='valid')(convolution_layers[i])

            for i in range(len(self.filter_sizes))
        ]

        if len(self.filter_sizes) > 1:
            tensor = Concatenate(axis=1)(max_pool_layers)

        tensor = Flatten()(tensor)
        tensor = Dense(self.output_size, activation='relu')(tensor)

        return tensor
