from src.opt.gcd import GCD
import numpy as np


class AdaGrad(GCD):
    def __init__(self, learning_rate=0.5):
        super(GCDMomentum, self).__init__(learning_rate)
        pass
         # self.__learning_rate = learning_rate
        

    def optimize(self, layer, global_grade: np.ndarray) -> None:
         """
			from layer : 
		    layer.V : V (t) = (dl/dw(t))^2 + (dl/dw(t-1))^2 + ( dl/dw(t-2))^2 + ...
			aka sum of squares of vectorized weights 
			v , global_grade and layer.weights ve same width

			should be used with mini-batch training
				
        """
        #equivalent to taking diagonal of outer product of grad matrix 
        layer.V = layer.V + np.square(global_grade) 
        #equivalent to taking square root of each alphat
        #should add epsilon(avoid zero in denominator)
        learning_vect=learning_rate * np.power(layer.V,-0.5)

        layer.weights = layer.weights - np.multiply( learning_vect, global_grade)