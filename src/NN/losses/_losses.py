"""
Helper functions for loss functions
"""
import numpy as np

def hinge_loss(Y, Y_Hat):
    """
       Returns the hinge loss funciton @ the input Y , Y_Hat
    """
    return np.maximum(0, 1 - (Y * Y_Hat))

def hinge_loss_drv(Y,Y_Hat,x):
    grad=0
    V = Y * Y_Hat
    grad += 0 if V > 1 else (-Y*x)
    return grad

def mean_squared_loss(Y, Y_Hat):
    """
       Returns the mean squared loss funciton @ the input Y , Y_Hat
    """
    diff = Y_Hat - Y
    return 0.5*np.sum(np.dot(diff.T, diff),axis=1, keepdims=True)
def mean_squared_loss_drv(Y, Y_Hat,x):

    return np.dot(x.T,Y_Hat)-np.dot(x.T,Y)




def cross_entropy_loss(Y, Y_Hat):
    """
       Returns the cross entropy loss funciton @ the input Y , Y_Hat
       l=−(ylog(p)+(1−y)log(1−p))
    """
    # epsilon = 1e-12 ,Y_Hat = np.clip(Y_Hat, epsilon, 1. - epsilon)
    return -np.sum(Y * np.log(Y_Hat + 1e-9)+(1-Y)*(np.log(1-Y_Hat)))

    # if Y.any() == 1:
    #     return -np.log(np.abs(Y_Hat))
    # else:
    #     return -np.log(1 - Y_Hat)
def cross_entropy_loss_drv(Y,x):
    X = np.copy(x)
    grad= np.exp(X)/np.sum(np.exp(X))
    grad[range(Y.shape[0]), Y] -= 1
    return grad



def log_like_hood_loss_exp(Y, Y_Hat):
    """
       Returns the loglikehood loss funciton in exponential form @ the input Y , Y_Hat
    """
    return -np.log(1+ np.exp(Y * Y_Hat))

def log_like_hood_loss(Y, Y_Hat):
    """
       Returns the loglikehood loss funciton @ the input Y , Y_Hat
    """
    return -np.log(np.abs((Y/2)-0.5+Y_Hat))



