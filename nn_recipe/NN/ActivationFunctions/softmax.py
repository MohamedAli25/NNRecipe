from nn_recipe.NN.__function import Function
import numpy as np


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
        expZ = np.exp(x)
        return expZ / np.sum(expZ, axis=0)

    def _calc_local_grad(self, x, *args, **kwargs):
        total_sum = np.sum(np.exp(x))
        return ((total_sum - np.exp(x))*np.exp(x))/(total_sum**2)
