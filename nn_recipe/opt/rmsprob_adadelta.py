from nn_recipe.opt.gcd import GCD
import numpy as np


class RmsProb(GCD):
    def __init__(self, learning_rate=0.5,beta=0.95):
        super(GCDMomentum, self).__init__(learning_rate)
        pass
         # self.__learning_rate = learning_rate
       	 # self.__beta = beta

    def optimize(self, layer, global_grade: np.ndarray) -> None:
         """
			from layer : 
		    layer.S : S(new) = beta*s(old)+(1-beta)*grad^2
			where beta is averaging /forgetting factor 
			S , global_grade and layer.weights ve same width

			should be used with mini-batch training
				
        """
        #equivalent to taking diagonal of outer product of grad matrix 
        layer.S = self.__beta* layer.S +(1-self.__beta)* np.square(global_grade) 
        #equivalent to taking square root of each alphat
        #should add epsilon(avoid zero in denominator)
        learning_vect=self.__learning_rate * np.power(layer.S,-0.5)

        layer.weights = layer.weights - np.multiply( learning_vect, global_grade)