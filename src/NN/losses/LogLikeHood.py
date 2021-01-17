from src.NN.function import Function
import numpy as np


def LogLikeHoodLoss_exp(Y, Y_Hat):
    """used with identity activation function
     using regularization
    ∇L=((-y.x.exp(-y.y_hat))/(1+exp(-y.y_hat)))+λw
    """
    return -np.log(1+ np.exp(Y * Y_Hat))



def LogLikeHoodLoss(Y, Y_Hat):
    """if used with sigmoid activation function
    ∇L=(-yx/1+exp(y*wx))
    """
    return -np.log(np.abs((Y/2)-0.5+Y_Hat))


#TODO implement _clac_local_grad
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
        return -np.log(np.abs((Y/2)-0.5+Y_Hat)) / N