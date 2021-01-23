from.optimizer  import Optimizer
import numpy as np


class GD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, *args, **kwargs):
        if learning_rate <= 0:
            raise Optimizer.LearningRateValueError(learning_rate)
        if type(learning_rate) is not float and type(learning_rate) is not int:
            raise Optimizer.LearningRateTypeError(type(learning_rate))
        self._learning_rate = learning_rate

    def optimize(self, layer, delta: np.ndarray) -> None:
        # (batch_size, n_rows , n_cols, n_c)
        layer.weights = layer.weights - self._learning_rate * np.dot(delta, layer.local_grad["dW"])/layer.weights.shape[1]
        layer.bias = layer.bias - self._learning_rate * np.sum(delta) / delta.shape[1]

    def multi_bipolar(self, Y, layer , delta: np.ndarray) ->None:
        delta_W = np.dot(delta, layer.local_grad["dW"])
        delta_B = np.sum(delta) / delta.shape[1]
        yi = np.argmax(Y, axis=0)   # [1 2 5]
        r = np.argmax(layer._cache)
        if not r == yi:
            x = layer._cache['X']
            delta_W[r]+=x
            delta_B[r]+=x

            delta_W[yi]-=x
            delta_B[yi]-=x
        #TODO check bias dimensions
        layer.weights = layer.weights - self._learning_rate * delta_W
        layer.bias = layer.bias - self._learning_rate * delta_B

        
