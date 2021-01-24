from nn_recipe.NN.ActivationFunctions import *
from nn_recipe.NN.LossFunctions import *
from nn_recipe.NN.Layers import *
from nn_recipe.Opt import GD
from nn_recipe.NN import Network
import numpy as np

# net = Network(
#     layers=[
#         Linear(in_dim=784, out_dim=25, activation=Sigmoid()),
#         Linear(in_dim=25, out_dim=10, activation=Identity())
#     ],
#     optimizer=GD(learning_rate=0.001),
#     loss_function=MClassLogisticLoss(sum=True, axis=0),
# )
#
# net.save("C:\\Users\\mgtmP\\Desktop\\mnist_net.net")
from nn_recipe.utility import OneHotEncoder

X = np.loadtxt("C:\\Users\\mgtmP\\Desktop\\NNRecipe\\mnist_1k.csv", delimiter=",")
Y = X[:,0].reshape((-1, 1))
X = X[:,1:]
X = X / 255

net = Network.load("C:\\Users\\mgtmP\\Desktop\\mnist_net.net")
out = net.evaluate(X)

encoder = OneHotEncoder(
    types=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    active_state=1,
    inactive_state=0
)
yhat = encoder.decode(out)
yhat = np.array(yhat).reshape((-1, 1))
print(yhat - Y)
# print("######################################################################################")
# print(Y)