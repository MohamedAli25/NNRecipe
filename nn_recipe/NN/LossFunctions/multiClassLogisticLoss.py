from .__loss_function import LossFunction
import numpy as np


class MClassLogisticLoss(LossFunction):
    def _compute_loss(self, Y, Y_hat):
        return -1*np.log(Y[np.argmax(Y_hat, axis=0)][:,0].reshape((1,-1)))


    def _compute_local_grad(self, Y, Y_Hat):
        return 0

"""
-log(YTrue)
"""