import numpy as np
from src.NN.layers.linear import Linear
from src.NN.activations.activation import Sigmoid
from src.NN.losses.MeanSquared import MeanSquaredLoss
from src.opt.GD import GCD

x = np.array([[1, 0.1]])
y = np.array([[0.6], [0.01]])
opt = GCD(learning_rate=1)
msl = MeanSquaredLoss()
l1 = Linear(in_dim=2, out_dim=2, activation=Sigmoid(), weights=np.eye(2), bias=np.array([[1], [1]]))
l2 = Linear(in_dim=2, out_dim=2, activation=Sigmoid(), weights=np.eye(2), bias=np.array([[1], [1]]))

for a in range(20):
    out = l2(l1(x).T)
    msl(y, out)
    print("loss {}:".format(str(a)), msl(y, out))
    delta = np.dot(l2.local_grad["dX"].T, msl.local_grad)
    opt.optimize(l2, delta)
    delta = np.dot(l1.local_grad["dX"].T, delta)
    opt.optimize(l1, delta)
