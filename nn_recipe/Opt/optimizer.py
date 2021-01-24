from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, y, layer, delta, opt_type: str, batch_size=1):
        pass

    class LearningRateValueError(Exception):
        def __init__(self, learning_rate_value):
            message = "Optimizer learning rate must be greater than zero, current value is " + str(learning_rate_value)
            super().__init__(message)

    class LearningRateTypeError(Exception):
        def __init__(self, learning_rate_type):
            message = "Optimizer learning rate must be a scalar real number current type is " + str(learning_rate_type)
            super().__init__(message)


def MomentumBased():
    pass