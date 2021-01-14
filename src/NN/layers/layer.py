from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from src.NN.function import Function

#TODO add default value to activatoin type
class Layer(Function):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super(Layer, self).__init__()
        self._weights: np.ndarray = None
        self._bias: np.ndarray = None
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def _init_weights(self, *args, **kwargs):
        pass
    
    # def update_weights(self, lr):
    #     for w, val in self.__weight.items():
    #         self._weight[w] = self._weight[w] - (lr * self._weight_update[w]) 
    
    @property
    def weights(self, *args, **kwargs):
        """ Layer's weights getter"""
        return self._weights

    @weights.setter
    def weights(self, value, *args, **kwargs):
        """ Layer's weights setter"""
        self._weights = value
