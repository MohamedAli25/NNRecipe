from nn_recipe.NN.Layers import Linear
from nn_recipe.NN.ActivationFunctions import Sigmoid, ReLU, LeakyReLU, Identity, Softmax
from nn_recipe.NN.LossFunctions import CrossEntropyLoss, MeanSquaredLoss
from nn_recipe.Opt import GD
from nn_recipe.NN import Network
from nn_recipe.utility import OneHotEncoder

import numpy as np

# you can download the whole 60K example from http://yann.lecun.com/exdb/mnist/
# i attached bellow a sample from it just a 1000 example
# change the path bellow to mnist_1k.csv
X = np.loadtxt("C:\\Users\\mgtmP\\Desktop\\mnist_1k.txt", delimiter=",")
Y = X[:,0].reshape((-1, 1))
X = X[:,1:]
X = X / 255

encoder = OneHotEncoder(
    types=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    active_state=1,
    inactive_state=0
)
Y = encoder.encode(Y)

net = Network(
    layers=[
        Linear(in_dim=784, out_dim=25, activation=Sigmoid()),
        Linear(in_dim=25, out_dim=10, activation=Softmax())
    ],
    optimizer=GD(learning_rate=0.003),
    loss_function=CrossEntropyLoss(sum=True, axis=1),
)

net.train(X, Y, notify_func=print, batch_size=10000, max_itr=3)

# # TODO Bugs needed to be solved: Batch size spliting array
# # TODO implement softmax layer
#
#
# X = np.array([[1, 1, 1]])
# loss = Softmax()
# print(loss(X))



# OneHotEncoder testing
# Y = np.array([["Mohamed"], ["Ali"], ["Ahmed"]])
# encoder = OneHotEncoder(
#     types=["Mohamed", "Ahmed", "Ali"],
# )
# out = encoder.encode(Y)
# print("encoded", out)
# print("decoded", encoder.decode(out))