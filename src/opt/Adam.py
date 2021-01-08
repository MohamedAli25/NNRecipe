from src.opt.gcd import GCD
import numpy as np


class  ADAM(GCD):
    def __init__(self, learning_rate,beta1=.5,beta2=0.95):
        super(GCDMomentum, self).__init__(learning_rate)
        pass
         # self.__learning_rate = learning_rate
       	 # self.__beta2 = beta2
       	  # self.__beta1 = beta1

    def optimize(self, layer, global_grade: np.ndarray) -> None:
         """
				from layer : 
		    layer.S : S(new) = beta2*s(old)+(1-beta2)*grad^2
			where beta is averaging /forgetting factor 
			S , global_grade and layer.weights ve same width
			
			from layer : 
		    layer.V : V(new) = beta1*V(old)+(1-beta1)*grad
			represents adapting momentum with no bias correction 
			v , global_grade and layer.weights ve same width

			should be used with mini-batch training
				
        """
     	layer.V = self.__beta1* layer.V +(1-self.__beta1)* global_grade
        layer.S = self.__beta2* layer.S +(1-self.__beta2)* np.square(global_grade) 
     
       
        learning_vect=self.__learning_rate * np.multiply(np.power(layer.S,-0.5), layer.V)

        layer.weights = layer.weights - np.multiply( learning_vect, global_grade)