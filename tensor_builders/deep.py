from typing import List

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.framework.ops import Tensor

from .base import BaseTensorBuilder


class DeepTensorBuilder(BaseTensorBuilder):
    variables = ['layers', 'dropout']

    def __init__(self, layers: List[int] = None, dropout: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.dropout = dropout

    def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
        tensor = input_tensor

        for i in range(len(self.layers)):
            tensor = Dense(self.layers[i], activation='relu')(tensor)

            if i < len(self.layers) - 1 and self.dropout > 0:
                tensor = Dropout(self.dropout)(tensor)

        return tensor
