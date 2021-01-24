from nn_recipe.NN.Layers import Linear
from nn_recipe.NN.ActivationFunctions import Sigmoid, ReLU, LeakyReLU, Identity, Softmax
from nn_recipe.NN.LossFunctions import CrossEntropyLoss, MeanSquaredLoss, MClassLogisticLoss, MClassBipolarPerceptron
from nn_recipe.Opt import GD
from nn_recipe.NN import Network
from nn_recipe.utility import OneHotEncoder

import numpy as np

X = np.loadtxt("C:\\Users\\mgtmP\\Downloads\\mnist_train.csv", delimiter=",")
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
    optimizer=GD(learning_rate=0.001),
    loss_function=MClassLogisticLoss(sum=True, axis=0),
)
loss, itr = net.train(X, Y, notify_func=print, batch_size=1, max_itr=100)
print(loss)
print(itr)

# l1 = Linear(in_dim=784, out_dim=10, activation=Identity())
# loss = MClassLogisticLoss()
#
# for example_index in range(1, len(X)):
#     example_x = X[example_index,:].reshape(1, -1)
#     example_y = Y[example_index,:].reshape(1, -1)
#     out = l1(example_x)
#     print(loss(example_y, out))
#     print(loss.local_grad)



# for example in X:
#     example = example.reshape()

# X = np.array([[5, 6, 7],
#               [5, 6, 7]])
# loss = MClassLogisticLoss()
# # print(loss(X))
# #
# #
# #
# # OneHotEncoder testing
# Y = np.array([["Ahmed"],
#               ["Ali"]])
#
# encoder = OneHotEncoder(
#     types=["Mohamed", "Ahmed", "Ali"],
# )
#
# encoded_Y = encoder.encode(Y)
# print(loss(encoded_Y, X))
# print(loss.local_grad)
#
