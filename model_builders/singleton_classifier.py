from typing import List

from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.python.keras.engine.training import Model

from tensor_builders.cnn import CNNTensorBuilder
from tensor_builders.deep import DeepTensorBuilder


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

        if self.use_words_feature or self.use_context_feature:
            self.cnn_tensor_builder = CNNTensorBuilder(
                input_length=10, vocab_size=len(embedding_matrix), vector_size=len(embedding_matrix[0]),
                embedding_matrix=embedding_matrix, filter_sizes=[2, 3, 4], num_filters=64, trainable_embedding=False,
                output_size=16
            )

    def create_model(self, softmax: bool = True) -> Model:
        inputs = []
        tensors = []

        if self.use_words_feature:
            inp, tensor = self.cnn_tensor_builder.create_tensor()
            inputs.append(inp)
            tensors.append(tensor)

        if self.use_context_feature:
            inp_prev, tensor_prev = self.cnn_tensor_builder.create_tensor()
            inp_next, tensor_next = self.cnn_tensor_builder.create_tensor()

            inputs.extend([inp_prev, inp_next])
            tensors.extend([tensor_prev, tensor_next])

        if self.use_syntactic_feature:
            inp, tensor = self.deep_tensor_builder.create_tensor(input_shape=(self.syntactic_features_num,),
                                                                 layers=[32, 16],
                                                                 dropout=0.2)
            inputs.append(inp)
            tensors.append(tensor)

        if len(tensors) > 1:
            tensor = Concatenate()(tensors)
        elif len(tensors) == 1:
            tensor = tensors[0]
        else:
            raise Exception('Should have features')

        _, tensor = self.deep_tensor_builder.create_tensor(layers=[32, 8], dropout=0.2, input_tensor=tensor)

        if softmax:
            tensor = Dense(2, activation='softmax')(tensor)
        else:
            tensor = Dense(1, activation='sigmoid')(tensor)

        model = Model(inputs=inputs, outputs=[tensor])

        if softmax:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
