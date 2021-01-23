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

    def update_delta(self, y, layer, delta: np.ndarray, opt_type:str, batch_size):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]

        if opt_type == "multi_logistic":
            correct_index = np.argmax(y, axis=0)
            x = layer._cache['X']
            k = layer._cache['output']
            r = np.argmax(layer._cache['output'].T, axis=0)
            d = (r == correct_index).astype(r.dtype)
            for j in range(batch_size):
                for i in range(len(d)):
                    if d[i]:
                        delta[r[i]] -= ((1 - k[correct_index[i]]) * x[j])
                    else:
                        delta[r[i]] += k[r[i]] * x[j]
            return delta_w, delta_b

        return delta_w, delta_b

    def optimize(self, y, layer, delta: np.ndarray, opt_type: str,batch_size) -> None:
        delta_w, delta_b = self.update_delta(y, layer, delta, opt_type, batch_size)
        layer.weights = layer.weights - self._learning_rate * delta_w
        layer.bias = layer.bias - self._learning_rate * delta_b
