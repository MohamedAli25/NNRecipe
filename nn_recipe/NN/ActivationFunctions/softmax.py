from nn_recipe.NN.__function import Function
import numpy as np


class Softmax(Function):
    """
    Class represents the softmax activation function

    >>> x = np.array([1, 2, 3])          # input vector
    >>> f = Softmax(x)                   # creating softmax object
    >>> print(f(x))                      # calculating softmax of the input

    for more info about softmax implementation visit:   https://deepnotes.io/softmax-crossentropy
https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
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
        expZ = np.exp(x - np.max(x))
        out = expZ / np.sum(expZ, axis=0)
        print("forward of softmax", out)
        return out

    def _calc_local_grad(self, x, *args, **kwargs):
        print("from softmax", x.shape)