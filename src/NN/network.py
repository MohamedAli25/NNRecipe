from src.NN.layers.layer import Layer
from src.opt.optimizer import Optimizer
from enum import Enum, auto


class OptimizerTypeError(Exception):
    def __init__(self):
        super(OptimizerTypeError, self).__init__("Optimizer object must be an instance of Optimizer class")


class LayerTypeError(Exception):
    def __init__(self):
        super(LayerTypeError, self).__init__("Layer object must be an instance of Layer class")


class Network:
    class TrainingMode(Enum):
        ONLINE_TRAINING = auto()
        BATCH_TRAINING = auto()

    def __init__(self, optimizer: Optimizer, loss_function, train_mode, batch_size=1, *args):
        self.__layers: list[Layer] = []
        self.__opt = None
        self.__loss = None
        self.__train_mode = None
        self.__batch_size = None
        self.set_optimizer(optimizer)
        self.set_loss_function(loss_function)
        self.set_loss_function(train_mode)
        self.set_batch_size(batch_size)
        self.add_layers(*args)

    def add_layers(self, *args):
        for layer in args:
            if not isinstance(layer.__class__, Layer):
                raise LayerTypeError()
            self.__layers.append(layer)

    def set_optimizer(self, optimizer:Optimizer):
        if not isinstance(optimizer.__class__, Optimizer):
            raise Network.OptimizerTypeError()
        self.__opt = optimizer

    def set_loss_function(self, loss_function):
        self.__loss = loss_function

    def set_train_mode(self, train_mode):
        self.__train_mode = train_mode

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size

    def train(self, X, Y, batch_size = None):
        if batch_size is not None:
            self.set_batch_size(batch_size)
        if self.__train_mode == Network.TrainingMode.ONLINE_TRAINING:
            pass

    def save(self):
        pass

    def open(self):
        pass

    def flush(self):
        pass

    def load_parameter(self):
        pass

    def evaluate(self):
        pass
