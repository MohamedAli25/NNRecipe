from nn_recipe.NN.Layers import Linear
from nn_recipe.NN.ActivationFunctions import Sigmoid, ReLU, LeakyReLU, Identity, Softmax
from nn_recipe.NN.LossFunctions import CrossEntropyLoss, MeanSquaredLoss, MClassLogisticLoss, MClassBipolarPerceptron
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
        Linear(in_dim=25, out_dim=10, activation=Identity())
    ],
    optimizer=GD(learning_rate=0.003),
    loss_function=MClassBipolarPerceptron(sum=True, axis=0),
)
net.train(X, Y, notify_func=print, batch_size=None, max_itr=1)


# # TODO Bugs needed to be solved: Batch size spliting array
# # TODO implement softmax layer
#
#
# X = np.array([[5, 6, 7]])
# loss = MClassBipolarPerceptron()
# # print(loss(X))
# #
# #
# #
# # OneHotEncoder testing
# Y = np.array([["Ahmed"]])
#
# encoder = OneHotEncoder(
#     types=["Mohamed", "Ahmed", "Ali"],
# )
#
# encoded_Y = encoder.encode(Y).T
# loss(encoded_Y, X)
#
