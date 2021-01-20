import numpy as np
from nn_recipe.NN.Layers.linear import Linear
from nn_recipe.NN.Layers.pooling import*
from nn_recipe.NN.ActivationFunctions.Sigmoid import Sigmoid
from nn_recipe.NN.LossFunctions.MeanSquared import MeanSquaredLoss
from nn_recipe.Opt.gd import GD

from nn_recipe.NN.network import Network
from PIL import Image

# x = np.array([
#     [1, 1],
#     [2, 3],
#     [4, 7],
#     [-1, -1],
#     [-7, -0.2],
#     [-0.1, -3]
# ])

# y = np.array([
#     [1],
#     [1],
#     [1],
#     [-10],
#     [-10],
#     [-10],
# ])

# l1 = Linear(in_dim=2, out_dim=1, activation=Sigmoid())
# l11 = Linear(in_dim=2, out_dim=1, activation=Sigmoid(), weights=np.copy(l1.weights), bias=np.copy(l1.bias))
# net = Network(
#     Layers=[l11],
#     optimizer=GD(learning_rate=0.1),

# )
# loss, it_no = net.train(x, y, epsilon=0.1)
# print(loss)
# print(net.evaluate([-7, -0.2]))

# print("####################################################################################################")
# Opt = GD(learning_rate=0.1)
# msl = MeanSquaredLoss()
# for a in range(5):
#     out = l1(x)
#     loss = msl(y, out)
#     print("{}".format(loss))
#     delta = msl.local_grad    # dL/dy (last layer)
#     # print("delta", delta)
#     delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
#     Opt.optimize(l1, delta)
#     delta = np.dot(delta.T, l1.local_grad["dX"])
#
#     # delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
#     # Opt.optimize(l1, delta)

# print(l1.weights)
# print(l1.bias)
# print(l2.weights)
# print(l2.bias)
# print(l2(l1(np.array([[5,5]]))))

# img = Image.open(r'E:\\Engineering_courses\\Senior\\NN\\Project\\andrew2.jpeg') #.convert('LA')
# p1 = MaxPool2D(kernelSize=3, strides=2, padding=PaddingType.SAME)
# # print(np.array(img).shape)
# p1_out = p1(np.array(img))
# # in_p = np.arange(1, 10).reshape((3,3,1))
# # print(in_p)
# # print(p1_out)
# Image.fromarray(np.uint8(p1_out)).show()

from nn_recipe.NN import ActivationFunctions