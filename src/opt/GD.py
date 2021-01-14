from src.opt.optimizer import Optimizer
import numpy as np


class GCD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        if learning_rate <= 0:
            raise Optimizer.LearningRateValueError(learning_rate)
        if type(learning_rate) is not float and type(learning_rate) is not int:
            raise Optimizer.LearningRateTypeError(type(learning_rate))
        self.__learning_rate = learning_rate

    def optimize(self, layer, delta: np.ndarray) -> None:
        layer.weights = layer.weights - self.__learning_rate * delta * layer.local_grad["dW"]
        layer.bias = layer.bias - self.__learning_rate * delta * layer.local_grad["dB"]

