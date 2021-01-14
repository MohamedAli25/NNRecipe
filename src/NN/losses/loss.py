from src.NN.function import Function
from _losses import *
import numpy as np

#TODO this file to be depricated
class CrossEntropyLoss(Function):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        """
          - computes the cross entropy loss
          -cross_entropy_loss=−(Ylog(Y_Hat)+(1−Y)log(1−Y_Hat))
          :param Y   : numpy.ndarray Should contain class labels for each data point in x.
                Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
          :return:
          :rtype:
        """
        N = Y_Hat.shape()
        return cross_entropy_loss(Y, Y_Hat)

    def local_grad(self, Y, x):
        """
        - computes the grad of cross entropy loss
        - ∇ cross_entropy_loss_drv(Y,x) =
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
        :return:
        :rtype:
        """
        return {'dL': cross_entropy_loss_drv(Y, x) / Y.shape[0]}


class MeanSquaredLoss(Function):
    def __init__(self):
        super(MeanSquaredLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        """
        - computes the mean_squared_loss
        - mean_squared_loss= 0.5(Y-Y_Hat)**2
        :param Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        N = Y.shape[0]
        return mean_squared_loss(Y, Y_Hat) * (1 / N)

    def local_grad(self, Y, Y_Hat, x):
        """
        - computes the grad of mean_squared_loss
        - ∇mean_squared_loss (Y,x) =(Y_Hat-Y)x
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        N = Y.shape[0]
        return {'dL': mean_squared_loss_drv(Y, Y_Hat, x) / N}


class LogLikeHoodLoss(Function):
    def __init__(self):
        super(LogLikeHoodLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        """
        - computes the grad of mean_squared_loss
        - log_like_hood_loss(Y,Y_Hat) =-log(|(Y/2)-0.5+Y_Hat|)
        :param Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        N = Y.shape[0]
        return log_like_hood_loss(Y, Y_Hat) / N


class HingeLoss(Function):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, Y, Y_Hat):
        """
        - computes the hinge_loss
        - hinge_loss (Y,Y_Hat) = max(0,1-Y*Y_Hat)
        :param Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        return hinge_loss(Y, Y_Hat)

    def local_grad(self, Y, Y_Hat, x):
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
        return {'dL': hinge_loss_drv(Y, Y_Hat, x)}
