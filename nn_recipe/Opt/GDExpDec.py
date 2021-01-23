from .gd import GD
import numpy as np


class GDExpDec(GD):
    def __init__(self, iteration_no, k,*args, **kwargs):
        super(GDExpDec, self).__init__(*args, **kwargs)
        self._iteration = iteration_no
        self._k = k

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str, batch_size) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type, batch_size)
        learning_rate = self._learning_rate * np.exp(-self._k * self._iteration)
        layer.weights = layer.weights - learning_rate * delta_w
        layer.bias = layer.bias - learning_rate * delta_b




