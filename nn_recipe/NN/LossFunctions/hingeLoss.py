from nn_recipe.NN.function import Function
import numpy as np




class HingeLoss(Function):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def _forward(self, Y, Y_Hat):
        """
        - computes the hinge_loss
        - hinge_loss (Y,Y_Hat) = max(0,1-Y*Y_Hat)
        :param Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        return np.maximum(0, 1 - (Y * Y_Hat))

    def _calc_local_grad(self, Y, Y_Hat, x):
        """
        - computes the grad of hinge_loss
        - ∇hinge_loss (Y,x) ={
                                0   Y*Y_Hat >= 1
                               -Yx  Y*Y_Hat <  1
        }
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        grad = 0
        V = Y * Y_Hat
        grad += 0 if V > 1 else (-Y * x)
        return grad