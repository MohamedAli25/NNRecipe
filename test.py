import numpy as np
from src.NN.layers.linear import Linear
from src.NN.layers.pooling import*
from src.NN.activations.Sigmoid import Sigmoid
from src.NN.losses.MeanSquared import MeanSquaredLoss
from src.opt.GD import GCD

# x = np.array([[1, 0.1]])
# y = np.array([[0.6], [0.01]])
# opt = GCD(learning_rate=1)
# msl = MeanSquaredLoss()
# l1 = Linear(in_dim=2, out_dim=2, activation=Sigmoid(), weights=np.eye(2), bias=np.array([[1], [1]]))
# l2 = Linear(in_dim=2, out_dim=2, activation=Sigmoid(), weights=np.eye(2), bias=np.array([[1], [1]]))

# for a in range(20):
#     out = l2(l1(x).T)
#     msl(y, out)
#     print("loss {}:".format(str(a)), msl(y, out))
#     delta = np.dot(l2.local_grad["dX"].T, msl.local_grad)
#     opt.optimize(l2, delta)
#     delta = np.dot(l1.local_grad["dX"].T, delta)
#     opt.optimize(l1, delta)

"""Pooling testing"""
in_p = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
p1 = MaxPool2D(kernelSize=2, strides=2, padding=PaddingType.SAME)
p1_out = p1(in_p)
pool_out = np.reshape(p1_out, (1, -1))
l1 = Linear(in_dim=9, out_dim=3, activation=Sigmoid(), weights=np.ones((3, 9)), bias=np.array([[1], [1], [1]]))
l2 = Linear(in_dim=3, out_dim=1, activation=Sigmoid(), weights=np.array([[1, 2, 3]]), bias=np.array([[1]]))
print(l2(l1(pool_out)))
