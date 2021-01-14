from __future__ import annotations

import numpy as np
from .layer import Layer
from src.NN.function import Function


class Linear(Layer):
    def __init__(self, in_dim, out_dim, acivation: Function, batch_size=1, *args, **kwargs):
        super(Linear, self).__init__(in_dim, out_dim)
        self.activation = acivation
        self._init_weights(self.in_dim, self.out_dim, batch_size)
    
    def _init_weights(self, in_dim, out_dim, batch_size):
        factor = np.tanh(1/in_dim)      # 1/sqrt(in_dim)    # 1/sqrt(indim + out_dim)
        self._weights = np.random.rand(out_dim, in_dim) * factor
        # self._bias = np.random.rand(out_dim, batch_size) * factor
        self._bias = np.ones((out_dim, batch_size))

    def _forward(self, x:np.ndarray): 
        """ Returns the output after applying the activation function"""
        return self.activation((np.dot(self._weights, x.T) + self._bias))

    def _calc_local_grad(self, x):
        """
            dy/dz = self.activation.local_grad
            d_z/d_w = x
            d_z/d_in = self.weights        z = np.dot(w, in.T)
            d_z/db = 1
            res = np.dot(dy/dz, dz/d_in)
            np.multiply(sigma, res)     not here
        """
        return {
            'dW': np.dot(self.activation.local_grad, x),
            'dX': np.multiply(self.activation.local_grad, self.weights),
            'dB': self.activation.local_grad
        }
