from layers import Function
import numpy as np

def MeanSquaredLoss(Y, Y_Hat):
    diff = Y_Hat - Y
    return np.sum(np.dot(diff.T, diff),axis=1, keepdims=True)

def CrossEntropyLoss(Y, Y_Hat):
    #not finished sum
    #log base e
    if Y.any() == 1:
        return -np.log(np.abs(Y_Hat))
    else:
        return -np.log(1 - Y_Hat)


def HingeLoss(Y, Y_Hat):
    return np.maximum(0, 1 - (Y * Y_Hat))


def LogLikeHoodLoss_exp(Y, Y_Hat):
    return -np.log(1+ np.exp(Y * Y_Hat))

def LogLikeHoodLoss(Y, Y_Hat):
    return -np.log(np.abs((Y/2)-0.5+Y_Hat))



class CrossEntropyLoss(Function):
    def forward(self,Y,Y_Hat):
       pass

class MeanSquaredLoss(Function):
    def forward(self, Y, Y_Hat):
        pass

class LogLikeHoodLoss(Function):
    def forward(self, Y, Y_Hat):
        pass

class HingeLoss(Function):
    def forward(self, Y, Y_Hat):
        pass

