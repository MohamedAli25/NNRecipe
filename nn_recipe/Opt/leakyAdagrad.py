from .gd import GD
import numpy as np


class GDLeakyAdaGrad(GD):
    def __init__(self,roh,*args, **kwargs):
        super(GDLeakyAdaGrad, self).__init__(*args, **kwargs)
        self.__roh=roh

    def update_delta(self, layer, delta: np.ndarray):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w,delta_b

    def optimize(self, layer, delta: np.ndarray) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self.__roh * layer.a + (1 - self.__roh) * np.square(delta_w)
        layer.ao = self.__roh * layer.ao + (1 - self.__roh) * np.square(delta_b)

        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), delta_w)
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), delta_b)