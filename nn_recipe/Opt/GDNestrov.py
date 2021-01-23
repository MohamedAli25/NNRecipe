from .GDMomentum import GDMomentum
import numpy as np

class GDNestrov(GDMomentum):
    def __init__(self, *args, **kwargs):
        super(GDNestrov, self).__init__(*args, **kwargs)

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type)
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)

        layer.v = self._beta * layer.v - self._learning_rate * delta_w * (layer.weights + self._beta * layer.v)
        layer.vo = self._beta * layer.vo - self._learning_rate * delta_b * (layer.bias + self._beta * layer.vo)
        layer.weights = layer.weights + layer.v
        layer.bias = layer.bias + layer.vo
