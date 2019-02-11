from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.layers import Input


class BaseTensorBuilder:
    variables = []

    def __init__(self, input_shape: tuple = None, input_only: bool = False) -> None:
        self.input_shape = input_shape
        self.input_only = input_only

    def create_tensor(self, input_tensor: Tensor = None, **kwargs) -> Tensor:
        self.check_variables()

        temp = {}

        for variable in self.variables:
            if variable in kwargs:
                temp[variable] = getattr(self, variable)
                setattr(self, variable, kwargs[variable])

        tensor = self.create_input_tensor() if input_tensor is None else input_tensor

        if not self.input_only:
            tensor = self.create_processing_tensor(tensor)

        for variable in temp.keys():
            setattr(self, variable, temp[variable])

        return tensor

    def check_variables(self) -> None:
        for variable in self.variables:
            if getattr(self, variable, None) is None:
                raise Exception('Variable %s is None' % variable)

    def create_input_tensor(self) -> Tensor:
        if getattr(self, 'input_shape', None) is None:
            raise Exception('Variable input_shape is None')

        return Input(self.input_shape)

    def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
        raise NotImplementedError()
