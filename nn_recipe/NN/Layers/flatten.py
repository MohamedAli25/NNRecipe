from nn_recipe.NN.__function import Function
class Flatten(Function):
    """Flattening layer for multidimentional input"""
    def _forward(self, X):
        self._cache = X.shape
        return X.reshape(X.shape[0], 1, -1)
    
    def _calc_local_grad(self, dL):
        return dL.reshape(self._cache)
