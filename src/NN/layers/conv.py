from __future__ import annotations

import numpy as np
from .layer import Layer

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
