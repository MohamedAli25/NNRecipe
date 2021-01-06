from layers import Function
import numpy as np

def sigmoid(x):
    """
    Returns the sigmoid funciton @ the input x
    """
    return 1/(1+np.exp(-x))

def sigmoid_drv(x):
    """
    Returns the derivative of the sigmoid function @ input x
    """
    s = sigmoid(x)
    return s * (1 - s)

def ReLU(x):
    return x * (x > 0).astype(x.dtype)  # or x * np.maximum(0, x)

def ReLU_drv(x):
    return (x > 0).astype(x.dtype)      # or np.maximum(0, x)

def LeakyReLU(x):
    return np.where(x > 0, x, x * lr)

def LeakyReLU_drv(x, lr=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = lr
    return dx

def tanh(x):
    return np.tanh(x)

def tanh_drv(x):
    return 1 - tanh(x)**2

def hard_tanh(x):
    x[x > 1] = 1
    x[x < -1] = -1
    return x    # or np.maximum(-1, np.minimum(1, x))

def hard_tanh_drv(x):
    X = np.copy(x)
    X[X<-1 and X>1] = 0
    X[X>=-1 and X<1] = 1
    return X

def Softmax(x):
    total = np.sum(np.exp(x), axis=1, keepdims=True)
    return (np.exp(x) / total)


class Sigmoid(Function):
    def __init__(self, *args, **kwargs):
        pass
    
    def forward(self, x):
        return sigmoid(x)

    def backward(self, x):
        pass

    def local_grad(self, x):
        pass
