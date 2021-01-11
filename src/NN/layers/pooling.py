from __future__ import annotations

import numpy as np
from src.NN.function import Function

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
