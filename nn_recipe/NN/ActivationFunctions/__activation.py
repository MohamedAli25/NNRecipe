from nn_recipe.NN.__function import Function
from abc import abstractmethod


class ActivationFunction(Function):
    ID = -1

    def save(self):
        return self.ID
