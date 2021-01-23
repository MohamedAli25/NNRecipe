from .leakyAdagrad import GDLeakyAdaGrad
import numpy as np


class  GDAdam(GDLeakyAdaGrad):
    def __init__(self, iteration_no, beta=0.999, *args, **kwargs):
        super(GDAdam, self).__init__(*args, **kwargs)
        self.__iteration_no = iteration_no
        self.__beta = beta

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str, batch_size) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type, batch_size)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        if not hasattr(layer, "f"):
            layer.f = np.zeros_like(layer.weights)
            layer.fo = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(delta_w)
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(delta_b)

        layer.f = self.__beta * layer.f + (1 - self.__beta) * delta_w
        layer.fo = self.__beta * layer.fo + (1 - self.__beta) * delta_b

        learning_rate = self._learning_rate * np.power(1 - np.power(self._roh,self.__iteration_no), 0.5) / (1 - np.power(self.__beta, self.__iteration_no))

        layer.weights = layer.weights - learning_rate / np.power(layer.a, 0.5) * layer.f
        layer.bias = layer.bias - learning_rate / np.power(layer.ao, 0.5) * layer.fo
