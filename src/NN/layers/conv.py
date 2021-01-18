from __future__ import annotations

import numpy as np
from .layer import Layer
from enum import Enum, auto


class PaddingType(Enum):
    SAME = auto()
    VALID = auto()


class ActivationFunction(Enum):
    LINEAR = auto()
    SIGMOID = auto()
    TANH = auto()
    RELU = auto()
    LEAKY_RELU = auto()


class Initializer(Enum):
    RANDOM_NORMAL = auto()
    RANDOM_UNIFORM = auto()
    ZEROS = auto()
    ONES = auto()


class Conv2D(Layer):
    def __init__(self, filters, kernelSize, strides=(1, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.bias = None
        self.weights = None

        if isinstance(kernelSize, int):
            kernelSize = (kernelSize, kernelSize)
        self.kernelSize = kernelSize

        if isinstance(strides, int):
            strides = (strides, strides)
        self.strides = strides

        if "padding" not in kwargs:
            kwargs["padding"] = PaddingType.VALID
        self.padding = kwargs["padding"]

        if "activation" not in kwargs:
            kwargs["activation"] = ActivationFunction.LINEAR
        self.activation = kwargs["activation"]

        if "useBias" not in kwargs:
            kwargs["useBias"] = True
        self.useBias = kwargs["useBias"]

        if "kernelInitializer" not in kwargs:
            kwargs["kernelInitializer"] = True
        self.kernelInitializer = kwargs["kernelInitializer"]

        if "biasInitializer" not in kwargs:
            kwargs["biasInitializer"] = True
        self.biasInitializer = kwargs["biasInitializer"]

        if "inputShape" not in kwargs:
            kwargs["inputShape"] = True
        self.inputShape = kwargs["inputShape"]

    def generate_initial_input_and_output(self, x, kernelSize, outputChannels, strides, padding: PaddingType):
        height, width, numOfChannels = x.shape[0], x.shape[1], outputChannels
        if padding == PaddingType.VALID:
            outputHeight = (height - kernelSize[0]) // strides[0] + 1
            outputWidth = (width - kernelSize[1]) // strides[1] + 1
            outputSize = (outputHeight, outputWidth, numOfChannels)
            newInput = x
        else:
            outputSize = (height, width, numOfChannels)
            paddingHeight = ((strides[0] - 1) * height - strides[0] + kernelSize[0]) // 2
            paddingWidth = ((strides[1] - 1) * width - strides[1] + kernelSize[1]) // 2
            newInput = np.zeros(outputSize)
            newInput[paddingHeight:paddingWidth + height, paddingWidth:paddingWidth + width, :] = x
        output = np.zeros(outputSize)
        return newInput, output

    def forward(self, x):
        newInput, output = self.generate_initial_input_and_output(x, self.kernelSize, self.strides, self.padding)
        outputHeight, outputWidth = output.shape
        # TODO
        for i in range(outputHeight):
            for j in range(outputWidth):
                heightStart = i * self.strides[0]
                heightEnd = heightStart + self.strides[0]
                widthStart = j * self.strides[1]
                widthEnd = widthStart + self.strides[1]
                output[i][j] = np.max(newInput[heightStart:heightEnd, widthStart:widthEnd])


    def backward(self, dL):
        pass

    def calc_local_grad(self):
        pass

    def _init_params(self):
        """ Initializing the parameters that will be used by the layer object (bias, weights) """
        self.bias = np.zeros((self.filters, 1))
        self.weights = np.random.randn((self.filters, self.kernelSize[0], self.kernelSize[1], self.inputShape[2]))
