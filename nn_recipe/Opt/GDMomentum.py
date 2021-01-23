from .gd import GD
import numpy as np


class GDMomentum(GD):
    def __init__(self, beta, *args, **kwargs):
        super(GDMomentum, self).__init__(*args, **kwargs)
        self._beta = beta

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str, batch_size) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type, batch_size)
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)

        layer.v = self._beta * layer.v - self._learning_rate * delta_w
        layer.vo = self._beta * layer.vo - self._learning_rate * delta_b
        layer.weights = layer.weights + layer.v
        layer.bias = layer.bias + layer.vo



