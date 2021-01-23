from .gd import GD
import numpy as np


class GDLeakyAdaGrad(GD):
    def __init__(self,roh,*args, **kwargs):
        super(GDLeakyAdaGrad, self).__init__(*args, **kwargs)
        self.__roh=roh

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self.__roh * layer.a + (1 - self.__roh) * np.square(delta_w)
        layer.ao = self.__roh * layer.ao + (1 - self.__roh) * np.square(delta_b)

        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), delta_w)
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), delta_b)