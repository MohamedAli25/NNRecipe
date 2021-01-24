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
from nn_recipe.DataLoader.mnistDataLoader import MNISTDataLoader

# mnist = MNISTDataLoader(download=False, rootPath="C:\\Users\\mgtmP\\Desktop\\NNRecipe\\mnist")
# mnist.load()
# X = mnist.get_train_data().reshape((-1, 28*28)) / 255
# Y = mnist.get_train_labels().reshape((-1, 1))
# X_validate = mnist.get_validation_data().reshape((-1, 28*28)) / 255
# Y_validate = mnist.get_validation_labels().reshape((-1, 1))

# print(X.shape)
# print(Y.shape)
# print(X_validate.shape)
# print(Y_validate.shape)

X = np.loadtxt("C:\\Users\\mgtmP\\Downloads\\mnist_train.csv", delimiter=",")
Y = X[:,0].reshape((-1, 1))
X = X[:,1:]
X = X / 255


net = Network(
    layers=[
        Linear(in_dim=784, out_dim=25, activation=Sigmoid()),
        Linear(in_dim=25, out_dim=10, activation=Identity())
    ],
    optimizer=GD(learning_rate=0.001),
    loss_function=MClassLogisticLoss(sum=True, axis=0),
)

# out = net.evaluate(X)
#
encoder = OneHotEncoder(
    types=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    active_state=1,
    inactive_state=0
)

# def validate():
#     out = net.evaluate(x_validate)
#     yhat = encoder.decode(out)
#     yhat = np.array(yhat).reshape((-1, 1))
#     print(np.count_nonzero(yhat - y_validate) / X.shape[0])
#
net.train(X, Y, notify_func=print, batch_size=1, max_itr=1)

out = net.evaluate(X)
# yhat = encoder.decode(out)
# print(yhat[0])
print(Y[0])
# yhat = np.array(yhat).reshape((-1, 1))
# print("error is  ", np.count_nonzero(yhat - Y) / X.shape[0])
