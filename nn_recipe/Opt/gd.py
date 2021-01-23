from.optimizer import Optimizer
import numpy as np


class GD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, *args, **kwargs):
        if learning_rate <= 0:
            raise Optimizer.LearningRateValueError(learning_rate)
        if type(learning_rate) is not float and type(learning_rate) is not int:
            raise Optimizer.LearningRateTypeError(type(learning_rate))
        self._learning_rate = learning_rate

    # TODO need to be in a separate class

    def update_delta(self, Y, layer, delta: np.ndarray, opt_type:str):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]

        if opt_type == "multi_bipolar":
            correct_index = np.argmax(Y, axis=0)  # [1 2 5]
            r = np.argmax(layer._cache['output'], axis=0)
            d = (r == correct_index).astype(r.dtype)
            if d == 0:
                x = layer._cache['X']
                delta_w[r] += x
                delta_b[r] += 1

                delta_w[correct_index] -= x
                delta_b[correct_index] -= 1

        return delta_w,delta_b



    def optimize(self, y, layer, delta: np.ndarray, opt_type:str) -> None:
        # (batch_size, n_rows , n_cols, n_c)
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type)
        layer.weights = layer.weights - self._learning_rate * delta_w
        layer.bias = layer.bias - self._learning_rate * delta_b
