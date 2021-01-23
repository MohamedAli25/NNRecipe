from .gd import GD
import numpy as np


class GDAdaGrad(GD):
    def __init__(self,*args, **kwargs):
        super(GDAdaGrad, self).__init__(*args, **kwargs)

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = layer.a + np.square(delta_w)
        layer.ao = layer.ao + np.square(delta_b)

        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), delta_w)
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), delta_b)