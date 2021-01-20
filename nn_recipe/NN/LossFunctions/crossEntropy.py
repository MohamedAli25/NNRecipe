from nn_recipe.NN.__function import Function
import numpy as np

#TODO replace function call with math formula
#TODO add normalization type
# regularization




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
        N = Y_Hat.shape[0]
        return - (1/N) * np.sum(Y * np.log(Y_Hat + 1e-9)+(1-Y)*(np.log(1-Y_Hat)))

    def _calc_local_grad(self, Y, x):
        """
        - computes the grad of cross entropy loss
        - ∇ cross_entropy_loss_drv(Y,x) =
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
        :return:
        :rtype:
        """
        grad = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        grad[range(Y.shape[0]), Y] -= 1
        return grad / Y.shape[0]
