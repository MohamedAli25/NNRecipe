from __future__ import annotations
import numpy as np
from abc import abstractmethod
from src.NN.function import Function
from src.utils.exceptions import ShapeError


class Layer(Function):          # TODO add default value to activation type
    """
    This Class represents a Layer in our Neural Network, the layer have _out_dim neurons and was connected to another
    layer with in_dim neurons

    Layer is responsible for:
        - Calculating the forward path
        - Calculating gradients that will be used to calculate backward path
    """
    def __init__(self, in_dim, out_dim, **kwargs):
        """ Initializes variables that will be used later by the Layer object"""
        # TODO add type checking for constructor input
        super(Layer, self).__init__()                   # calling base class (Function) constructor
        self._weights: np.ndarray = None                # weights matrix
        self._bias: np.ndarray = None                   # bias matrix
        self._in_dim = in_dim                           # input dimensions (number of neurons in the last layer)
        self._out_dim = out_dim                         # output dimensions (number of neuron in the current layer)
        self.__init_params(**kwargs)                    # initializing layer parameters

    @abstractmethod
    def _init_params(self):
        """ Initializing the parameters that will be used by the layer object (bias, weights) """
        pass

    def __init_params(self, **kwargs):
        """
        This function is used to call the init_params function in the sub class then check if there is an initial values
        supplied by the user

        :keyword weights: Initial value for layer weights
        :keyword bias:  Initial value for layer bias
        :raise TypeError: When the given initial data doesn't have the expected type
        :raise ShapeError: When the given initial data doesn't have the expected shape
        """
        self._init_params()
        # Checking for weights initial values
        if "weights" in kwargs:
            weights = kwargs["weights"]
            if type(weights) is not np.ndarray:
                raise TypeError("Required type is numpy.ndarray but the given type is {}".format(str(type(weights))))
            if self.weights.shape != weights.shape:
                raise ShapeError(required_shape=str(self.weights.shape), given_shape=str(weights.shape))
            else:
                self.weights = weights
        # Checking for bias initial values
        if "bias" in kwargs:
            bias = kwargs["bias"]
            if type(bias) is not np.ndarray:
                raise TypeError("Required type is numpy.ndarray but the given type is {}".format(str(type(bias))))
            if self._bias.shape != bias.shape:
                raise ShapeError(required_shape=str(self._bias.shape), given_shape=str(bias.shape))
            else:
                self._bias = bias


    @property
    def weights(self):
        """ Layer's weights getter"""
        return self._weights

    @weights.setter
    def weights(self, value):
        """ Layer's weights setter"""
        # TODO add type checking for weights setter
        self._weights = value

    @property
    def bias(self):
        """ Layer's weights getter"""
        return self._bias

    @bias.setter
    def bias(self, value):
        """ Layer's weights setter"""
        # TODO add type checking for bias setter
        self._bias = value
