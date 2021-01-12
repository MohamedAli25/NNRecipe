from src.NN.function import Function
from _losses import *
import numpy as np


class CrossEntropyLoss(Function):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        """
          - computes the cross entropy loss
          -cross_entropy_loss=−(Ylog(Y_Hat)+(1−Y)log(1−Y_Hat))
          - visit //// to get more info about cross entropy
          :param Y   : numpy.ndarray Should contain class labels for each data point in x.
                Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
          :return: cross_entropy_loss value at input x
          :rtype: np.float
        """
        N = Y_Hat.shape()
        return cross_entropy_loss(Y, Y_Hat)

    def local_grad(self, Y, x):
        """
        - computes the grad of cross entropy loss
        - ∇ cross_entropy_loss_drv(Y,x) =
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
        :return:  cross_entropy_loss gradient at input x ,Y
        :rtype: np.ndarray
        """
        return {'dL': cross_entropy_loss_drv(Y, x) / Y.shape[0]}


class MeanSquaredLoss(Function):
    def __init__(self):
        super(MeanSquaredLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        N = Y.shape[0]
        return mean_squared_loss(Y, Y_Hat) * (1 / N)

    def local_grad(self, Y, Y_Hat, x):
        N = Y.shape[0]
        return {'dL': mean_squared_loss_drv(Y, Y_Hat, x) / N}


class LogLikeHoodLoss(Function):
    def __init__(self):
        super(LogLikeHoodLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        N = Y.shape[0]
        return log_like_hood_loss(Y, Y_Hat) / N


class HingeLoss(Function):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        return hinge_loss(Y, Y_Hat)

    def local_grad(self, Y, Y_Hat, x):
        return {'dL': hinge_loss_drv(Y, Y_Hat, x)}
