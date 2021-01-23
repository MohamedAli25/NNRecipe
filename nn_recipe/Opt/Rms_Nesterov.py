from .GDMomentum import GDMomentum
import numpy as np
#ALREADY BROKEN
class GDRmsNestrov(GDMomentum):
    def __init__(self, roh, *args, **kwargs):
        super(GDRmsNestrov, self).__init__(*args, **kwargs)
        self._roh = roh

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type)
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights) #check
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(delta_w * (layer.weights + self._beta * layer.v))
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(delta_b * (layer.bias + self._beta * layer.vo))

        layer.v = self._beta * layer.v - (self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5)) * delta_w * (layer.weights + self._beta * layer.v)
        layer.vo = self._beta * layer.vo - (self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5)) * np.sum(delta)/delta.shape[1] * (layer.bias + self._beta * layer.vo)

        layer.weights = layer.weights + layer.v
        layer.bias = layer.bias + layer.vo