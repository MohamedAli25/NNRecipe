from src.NN.function import Function
import numpy as np

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

    def _forward(self,x,lr):
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

        return np.where(x > 0, x, x * lr)

    def _calc_local_grad(self, x,lr):
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
        dx = np.ones_like(x)
        dx[x < 0] = lr
        return dx
