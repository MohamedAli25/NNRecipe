import numpy as np
from src.NN.layers.linear import Linear
from src.NN.activations.Sigmoid import Sigmoid
from src.NN.losses.MeanSquared import MeanSquaredLoss
from src.opt.GD import GD

from src.NN.network import Network

x = np.array([
    [1, 1],
    [2, 3],
    [4, 7],
    [-1, -1],
    [-7, -0.2],
    [-0.1, -3]
])

y = np.array([
    [1],
    [1],
    [1],
    [-10],
    [-10],
    [-10],
])

l1 = Linear(in_dim=2, out_dim=1, activation=Sigmoid())
l11 = Linear(in_dim=2, out_dim=1, activation=Sigmoid(), weights=np.copy(l1.weights), bias=np.copy(l1.bias))
net = Network(
    layers=[l11],
    optimizer=GD(learning_rate=0.1),

)
loss, it_no = net.train(x, y, epsilon=0.1)
print(loss)
print(net.evaluate([-7, -0.2]))
# print("####################################################################################################")
# opt = GD(learning_rate=0.1)
# msl = MeanSquaredLoss()
# for a in range(5):
#     out = l1(x)
#     loss = msl(y, out)
#     print("{}".format(loss))
#     delta = msl.local_grad
#     # print("delta", delta)
#     delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
#     opt.optimize(l1, delta)
#     delta = np.dot(delta.T, l1.local_grad["dX"])
#
#     # delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
#     # opt.optimize(l1, delta)

# print(l1.weights)
# print(l1.bias)
# print(l2.weights)
# print(l2.bias)
# print(l2(l1(np.array([[5,5]]))))