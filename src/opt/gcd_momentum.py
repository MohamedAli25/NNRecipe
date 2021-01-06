from src.opt.gcd import GCD
import numpy as np


class GCDMomentum(GCD):
    def __init__(self, learning_rate,  alpha, beta):
        super(GCDMomentum, self).__init__(learning_rate)
        pass

    def optimize(self, layer, global_grade: np.ndarray) -> None:
        pass
