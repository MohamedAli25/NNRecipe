from nn_recipe.NN.__function import Function
import numpy as np


class MeanSquaredLoss(Function):
    def _forward(self, Y, Y_Hat):
        """
        Args:
            Y(array(N, 1)): vector of desired output
            Y_Hat(array(N, 1)): vector of actual output
        returns the loss value
        """
        return (1/(2*Y.shape[0]))*np.dot((Y_Hat - Y).T, (Y_Hat - Y))

    def _calc_local_grad(self, Y, Y_Hat):
        """returns the loss derivative wrt. the last output layer"""
        return Y_Hat - Y