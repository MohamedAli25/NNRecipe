from .meanSquared import MeanSquaredLoss
from .crossEntropy import CrossEntropyLoss
from .hingeLoss import HingeLoss
from .multiClassLogisticLoss import MClassLogisticLoss

__all__ = [
    "MeanSquaredLoss", "CrossEntropyLoss", "HingeLoss", "MClassLogisticLoss"
]