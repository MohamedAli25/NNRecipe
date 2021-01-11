from __future__ import annotations

import numpy as np
from .layer import Layer


class Linear(Layer):
    def __init__(self, in_dim, out_dim, batch_size=1):
        super(Linear, self).__init__()
        self._init_weights(in_dim, out_dim, batch_size)
    
    def _init_weights(self, in_dim, out_dim, batch_size):
        factor = np.tanh(1/in_dim)
        self._weights['W'] = np.random.rand(in_dim, out_dim) * factor
        self._weights['b'] = np.random.rand(batch_size, out_dim) * factor

    def _forward(self, x):
        return np.dot(x, self._weights['W']) + self._weights['b']

    def _calc_local_grad(self, x):
        return x
