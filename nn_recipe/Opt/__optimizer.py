from abc import ABC, abstractmethod


class Optimizer(ABC):
    ID = -1

    @abstractmethod
    def optimize(self, y, layer, delta, opt_type: str):
        pass

    class LearningRateValueError(Exception):
        def __init__(self, learning_rate_value):
            message = "Optimizer learning rate must be greater than zero, current value is " + str(learning_rate_value)
            super().__init__(message)

    class LearningRateTypeError(Exception):
        def __init__(self, learning_rate_type):
            message = "Optimizer learning rate must be a scalar real number current type is " + str(learning_rate_type)
            super().__init__(message)

    @abstractmethod
    def _save(self):
        pass

    def save(self):
        out = self._save()
        out["ID"] = self.ID
        return out

    @staticmethod
    @abstractmethod
    def load(data):
        pass