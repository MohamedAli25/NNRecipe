from .GDExpDec import GDExpDec
import numpy as np


class GDInvDec(GDExpDec):
    def __init__(self, *args, **kwargs):
        super(GDInvDec, self).__init__(*args, **kwargs)

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type)
        learning_rate = self._learning_rate / (1 + self._k * self._iteration_no)
        layer.weights = layer.weights - learning_rate * delta_w
        layer.bias = layer.bias - learning_rate * delta_b




