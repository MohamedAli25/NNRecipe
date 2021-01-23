from .gd import GD
import numpy as np


class GDExpDec(GD):
    def __init__(self, iteration_no, k,*args, **kwargs):
        super(GDExpDec, self).__init__(*args, **kwargs)
        self._iteration = iteration_no
        self._k = k

    def update_delta(self, layer, delta: np.ndarray):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w,delta_b

    def optimize(self, layer, delta: np.ndarray) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
        learning_rate = self._learning_rate * np.exp(-self._k * self._iteration)
        layer.weights = layer.weights - learning_rate * delta_w
        layer.bias = layer.bias - learning_rate * delta_b




