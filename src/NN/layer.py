from __future__ import annotations
import numpy as np
from abc import ABC
from function import Function

# class Function(ABC):
#     def __init__(self, *args, **kwargs):
#         # caching
#         self.cache = {}
#         self.grad = {}

#     def __call__(self, *args, **kwargs):
#         out = self.forward()
#         self._local_grad = self.local_grad(*args, **kwargs)
#         return out

#     def forward(self, *args, **kwargs):
#         pass

#     def backward(self, *args, **kwargs):
#         pass
    
#     @property
#     def local_grad(self):
#         return self._local_grad

#     @local_grad.setter
#     def local_grad(self, *args, **kwargs):
#         # calculate local grad
#         pass


class Layer(Function):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._params = {}
        self.params_update = {}

    def _init_params(self, *args, **kwargs):
        pass
    
    def update_params(self, *args, **kwargs):
        pass
    
    # @property
    def load_params(self, *args, **kwargs):
        pass
 

class Linear(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def _init_params(self, x):
        pass

    def forward(self, x):
        pass

    def backward(self, dL):
        pass

    def local_grad(self, x):
        pass


class Flatten(Function):
    def forward(self, X):
        self._cache['shape'] = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dL):
        return dL.reshape(self._cache['shape'])


class BatchNorm2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    # def _init_params():
    #     pass
    def forward(self, x):
        pass
    def backward(self, dL):
        pass


class MaxPool2D(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def forward(self, x):
        pass
    def backward(self, dL):
        pass
    def local_grad(self):
        pass


class AvgPool2D(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def forward(self, x):
        pass
    def backward(self, dL):
        pass
    def local_grad(self):
        pass


class Conv2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def forward(self, x):
        pass
    def backward(self, dL):
        pass
    def local_grad(self):
        pass
