from src.NN.function import Function
import numpy as np

#TODO replace function call with math formula
#TODO add normalization type
# regularization

def CrossEntropyLoss(Y, Y_Hat):
    #not finished sum
    #log base e
    if Y.any() == 1:
        return -np.log(np.abs(Y_Hat))
    else:
        return -np.log(1 - Y_Hat)


class CrossEntropyLoss(Function):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def _forward(self, Y, Y_Hat):
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

    def _calc_local_grad(self, Y, x):
        """
        - computes the grad of cross entropy loss
        - ∇ cross_entropy_loss_drv(Y,x) =
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
        :return:
        :rtype:
        """
        return (cross_entropy_loss_drv(Y, x) / Y.shape[0]).item(0)