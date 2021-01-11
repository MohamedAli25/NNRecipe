from function import Function
from _activations import *
import numpy as np

class Sigmoid(Function):
    """
    Class represents the sigmoid activation function

    >>> x = np.array([1, 2, 3])         # input vector
    >>> f = Sigmoid()                   # creating sigmoid object
    >>> print(f(x))                     # calculating sigmoid of the input
    >>> print(f.local_grad)             # get local_grad of the sigmoid at the input
    """
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def _forward(self, x):
        """
        - Forward pass of the sigmoid function
        - sig(x) = 1 / (1 + exp(-x))
        - visit https://en.wikipedia.org/wiki/Sigmoid_function to get more info about sigmoid
        :param x: input that is wanted to calculate the sigmoid at
        :return: sigmoid value at input x
        :rtype: np.ndarray
        """
        return sigmoid(x)

    def _calc_local_grad(self, x):
        """
        - Backward pass of the sigmoid function
        - ∇ sig(x) = a*(1-a),    a --> sig(x)
        - visit https://en.wikipedia.org/wiki/Sigmoid_function to get more info about sigmoid
        :param x: input that is wanted to calculate the sigmoid at
        :return: sigmoid gradient at input x
        :rtype: np.ndarray
        """
        return {'dY': sigmoid_drv(x)}


class Softmax(Function):
    """
    Class represents the softmax activation function

    >>> x = np.array([1, 2, 3])          # input vector
    >>> f = Softmax(x)                   # creating softmax object
    >>> print(f(x))                      # calculating softmax of the input
    """
    def __init__(self):
        super(Softmax, self).__init__()
    
    def _forward(self, x):
        """
        - Calculates the probabilities of the output nodes
        - softmax(x) = exp(x) / sum(exp(x[i])) i: 0-->N, N: number of classes
        - visit https://en.wikipedia.org/wiki/Softmax_function to get more info about softmax
        :param x: input that is wanted to calculate the softmax at
        :return: softmax value at input x
        :rtype: np.ndarray
        """
        return softmax(x)

        
class ReLU(Function):
    """
    Class represents the relu activation function

    >>> x = np.array([1, 2, 3])         # input vector
    >>> f = ReLU()                      # creating relu object
    >>> print(f(x))                     # calculating relu of the input
    >>> print(f.local_grad)             # get local_grad of the relu at the input
    """
    def __init__(self):
        super(ReLU, self).__init__()
    
    def _forward(self, x):
        """
        - Forward pass of the relu function
        - relu(x) = x * max(0, x)
        - visit https://en.wikipedia.org/wiki/Sigmoid_functionhttps://en.wikipedia.org/wiki/Rectifier_(neural_networks)
         to get more info about relu
        :param x: input that is wanted to calculate the relu at
        :return: relu value at input x
        :rtype: np.ndarray
        """
        return relu(x)
    
    def _calc_local_grad(self, x):
        """
        - Backward pass of the relu function
        - ∇ relu(x) = max(0, x)
        - visit https://en.wikipedia.org/wiki/Sigmoid_functionhttps://en.wikipedia.org/wiki/Rectifier_(neural_networks) 
        to get more info about sigmoid
        :param x: input that is wanted to calculate the relu at
        :return: relu gradient at input x
        :rtype: np.ndarray
        """
        return {'dY': relu_drv(x)}

class LeakyReLU(Function):
    """
    Class represents the leaky relu activation function

    >>> x = np.array([1, 2, 3])           # input vector
    >>> f = LeakyReLU()                   # creating leaky relu object
    >>> print(f(x))                       # calculating relu of the input
    >>> print(f.local_grad)               # get local_grad of the relu at the input
    """
    def __init__(self, lr=0.01):
        super(LeakyReLU, self).__init__()
        self._lr = lr
    
    def _forward(self, x):
        """
        - Forward pass of the leaky relu function
        - leaky_relu(x) = {
            x           for x >= 0
            lr * x      for x < 0
        }
        - visit https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
        to get more info about leaky relu
        :param x: input that is wanted to calculate the relu at
        :return: leakyrelu value at input x
        :rtype: np.ndarray
        """
        return leaky_relu(x, self._lr)
    
    def _calc_local_grad(self, x):
        """
        - Backward pass of the leaky relu function
        - ∇ leaky_relu(x) = {
            1           for x >= 0
            lr          for x < 0
        }
        - visit https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
        to get more info about leaky relu
        :param x: input that is wanted to calculate the leaky relu at
        :return: leaky relu gradient at input x
        :rtype: np.ndarray
        """
        return {'dY': leaky_relu_drv(x, self._lr)}